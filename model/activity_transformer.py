"""Activity-sequence transformer for the hybrid robot policy.

Predicts upcoming activities from the LLM-curated activity stream. Activity
names are an OPEN vocabulary, so tokens are built from frozen language
embeddings of the names (OpenAI text-embedding-3-small, fetched by the
curator into embeddings.json) passed through a learned linear projection.

Two config-switchable input representations sharing the same trunk:
  event    — one token per tagged segment: [proj(name_emb), tau(start),
             tau(end), dur/1440]; targets = next segment's activity + the
             gap to its start (SmoothL1). Primary mode.
  interval — one token per 10-min tick: [proj(name_emb_at_tick), tau(t)];
             target = activity at the next tick, 5x loss weight on ticks
             where the activity changes (else the model learns copy-previous).

Loss is an embedding-anchored softmax: logits_i = <W h, proj(e_i)> / tau over
the curated vocab — no per-class free parameters, so vocabulary edits don't
change the architecture, and confidences are calibrated for the policy's
action-emission threshold.

Time prediction (event mode) uses the same anchored-softmax idea over the
clock: the head emits a query in sinusoidal time space and scores it against
the time encodings of every 10-min tick of the day (positional-embedding
style), cross-entropy on the next activity's actual start tick. Absolute
clock anchoring beats gap regression because household routines are
clock-anchored (breakfast at ~8:15), not interval-anchored.

Plain-PyTorch training loop (the model is ~250k params on ~1.3k tokens;
Lightning would be overhead). Honors MAX_TRAIN_EPOCHS env var.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same sinusoidal formula as model/embeddings.py, but device-agnostic.
_TIME_PERIODS = [i * 60 for i in [24, 18, 12, 8, 4, 2, 1]]
D_TAU = 2 * len(_TIME_PERIODS)  # 14

DAY_START_MIN = 360.0
DAY_END_MIN = 1440.0
TICK_MIN = 10.0


def time_encoding(t_minutes: torch.Tensor) -> torch.Tensor:
    """Sinusoidal time-of-day encoding. t_minutes: (...,) -> (..., D_TAU)."""
    if not isinstance(t_minutes, torch.Tensor):
        t_minutes = torch.tensor(t_minutes, dtype=torch.float32)
    t = t_minutes.float().unsqueeze(-1)
    args = 2.0 * math.pi * t / torch.tensor(_TIME_PERIODS, dtype=torch.float32,
                                            device=t.device)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ActivityTransformer(nn.Module):
    """Tiny causal transformer over activity tokens.

    Vocab index convention: 0..V-1 = curated activity names (order of the
    `vocab` list). Interval mode adds IDLE as class V (learned vector, no
    name embedding). BOS is an input-only learned vector (event mode).
    """

    def __init__(self, vocab: list, name_embeddings: torch.Tensor, mode: str,
                 d_proj: int = 64, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 2, dim_feedforward: int = 192,
                 dropout: float = 0.2, logit_tau: float = 1.0,
                 max_len: int = 160):
        super().__init__()
        assert mode in ('event', 'interval')
        assert len(vocab) == name_embeddings.shape[0]
        self.vocab = list(vocab)
        self.mode = mode
        self.logit_tau = logit_tau
        self.register_buffer('name_emb', name_embeddings.float())  # (V, D_e)
        self.proj = nn.Linear(name_embeddings.shape[1], d_proj)
        self.idle_vec = nn.Parameter(torch.randn(d_proj) * 0.02)
        self.bos_vec = nn.Parameter(torch.randn(d_proj) * 0.02)
        in_dim = d_proj + (2 * D_TAU + 1 if mode == 'event' else D_TAU)
        self.in_head = nn.Linear(in_dim, d_model)
        # No learned positional embedding: each token already carries its
        # absolute clock position via the sinusoidal time features, and the
        # causal mask preserves order. A learned index embedding would make
        # the model sensitive to how many segments happen to precede.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_head = nn.Linear(d_model, d_proj)
        # Activity-CONDITIONED time head (event mode): for every candidate
        # activity in parallel, emit a query in sinusoidal clock space from
        # [hidden state ; that activity's projected name embedding], scored
        # against the time encodings of every 10-min tick (anchored softmax
        # over absolute clock positions). Answers "when would activity i
        # start" rather than "when does whatever-is-next start"; trained only
        # on the activity that actually came next.
        self.time_head = nn.Linear(d_model + d_proj, D_TAU)
        day_ticks = torch.arange(DAY_START_MIN, DAY_END_MIN, TICK_MIN)
        self.register_buffer('clock_anchors', time_encoding(day_ticks))  # (108, D_TAU)
        self.register_buffer('clock_ticks', day_ticks)

    @property
    def n_classes(self) -> int:
        return len(self.vocab) + (1 if self.mode == 'interval' else 0)

    @property
    def idle_idx(self) -> int:
        return len(self.vocab)

    @property
    def bos_idx(self) -> int:
        return len(self.vocab) + 1

    def _act_proj(self, act_idxs: torch.Tensor) -> torch.Tensor:
        """(B, T) indices -> (B, T, d_proj). idle_idx/bos_idx use learned
        vectors; real activities go through the frozen-embedding projection."""
        table = torch.cat([self.proj(self.name_emb),
                           self.idle_vec.unsqueeze(0),
                           self.bos_vec.unsqueeze(0)], dim=0)  # (V+2, d_proj)
        return table[act_idxs]

    def _class_anchors(self) -> torch.Tensor:
        anchors = self.proj(self.name_emb)                      # (V, d_proj)
        if self.mode == 'interval':
            anchors = torch.cat([anchors, self.idle_vec.unsqueeze(0)], dim=0)
        return anchors

    def forward(self, act_idxs: torch.Tensor, t_start: torch.Tensor,
                t_end: torch.Tensor = None,
                pad_mask: torch.Tensor = None) -> tuple:
        """act_idxs (B,T) int64, t_start/t_end (B,T) minutes,
        pad_mask (B,T) bool with True = PADDING.
        Returns (logits (B,T,C), gap_pred (B,T) normalized)."""
        act = self._act_proj(act_idxs)
        tau_s = time_encoding(t_start)
        if self.mode == 'event':
            tau_e = time_encoding(t_end)
            dur = ((t_end - t_start) / DAY_END_MIN).unsqueeze(-1)
            x = torch.cat([act, tau_s, tau_e, dur], dim=-1)
        else:
            x = torch.cat([act, tau_s], dim=-1)
        x = self.in_head(x)
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=x.device), diagonal=1)
        h = self.trunk(x, mask=causal, src_key_padding_mask=pad_mask)
        anchors = self._class_anchors()                          # (C, d_proj)
        logits = self.out_head(h) @ anchors.T / self.logit_tau   # (B, T, C)
        # activity-conditioned clock queries, all V activities in parallel:
        # (B, T, V, d_model + d_proj) -> (B, T, V, n_ticks)
        B, T, _ = h.shape
        act_anchors = self.proj(self.name_emb)                   # (V, d_proj)
        V = act_anchors.shape[0]
        hv = h.unsqueeze(2).expand(B, T, V, h.shape[-1])
        av = act_anchors.view(1, 1, V, -1).expand(B, T, V, -1)
        q_time = self.time_head(torch.cat([hv, av], dim=-1))     # (B,T,V,D_TAU)
        time_logits = q_time @ self.clock_anchors.T / self.logit_tau
        return logits, time_logits

    # -------------------------------------------------------------- predict

    @torch.no_grad()
    def predict_next(self, segments: list, now_min: float) -> tuple:
        """Eval-time helper (event mode): segments = the partial day's tagged
        segments [{activity_name, start_min, end_min}], in order. Returns
        (probs dict name->p over the vocab, predicted gap in minutes from the
        last segment's end to the next activity's start)."""
        self.eval()
        name_to_idx = {n: i for i, n in enumerate(self.vocab)}
        # Skip context segments whose activity the transformer never saw — e.g. an
        # activity the curator registered/renamed *after* this checkpoint trained
        # (it lives in the embeddings registry but never entered the training-segment
        # vocab). Dropping it from the context is a strict no-op when every segment
        # is in-vocab (all normal runs); it only prevents a KeyError on the rare
        # registry-only activity instead of crashing the eval.
        known = [s for s in segments if s['activity_name'] in name_to_idx]
        if len(known) != len(segments):
            dropped = sorted({s['activity_name'] for s in segments
                              if s['activity_name'] not in name_to_idx})
            print(f"[transformer] predict_next: dropped {len(segments) - len(known)} "
                  f"out-of-vocab context segment(s): {dropped}")
        segments = known
        if self.mode == 'event':
            idxs = [self.bos_idx] + [name_to_idx[s['activity_name']]
                                     for s in segments]
            starts = [DAY_START_MIN] + [s['start_min'] for s in segments]
            ends = [DAY_START_MIN] + [s['end_min'] for s in segments]
            act = torch.tensor([idxs], dtype=torch.int64)
            ts = torch.tensor([starts], dtype=torch.float32)
            te = torch.tensor([ends], dtype=torch.float32)
            logits, time_logits = self.forward(act, ts, te)
            probs = F.softmax(logits[0, -1], dim=-1)
            # per-activity predicted start = probability-weighted clock
            # position over future ticks (an upcoming activity cannot start
            # in the past)
            tprobs = F.softmax(time_logits[0, -1], dim=-1)   # (V, 108)
            future = self.clock_ticks >= float(now_min)
            gaps = {}
            for i, name in enumerate(self.vocab):
                if future.any():
                    fp = tprobs[i, future] / tprobs[i, future].sum().clamp_min(1e-9)
                    pred_start = float((fp * self.clock_ticks[future]).sum())
                else:
                    pred_start = float(now_min)
                gaps[name] = max(0.0, pred_start - float(now_min))
            gap_min = gaps
        else:
            # Interval mode predicts the NEXT TICK, which is often idle
            # between activities. Roll out greedily until the first non-idle
            # prediction (or the horizon) so the return value matches event
            # mode's "next activity + gap" semantics.
            ticks = torch.arange(DAY_START_MIN, float(now_min) + 1, TICK_MIN)
            tick_act = torch.full((len(ticks),), self.idle_idx,
                                  dtype=torch.int64)
            for s in segments:
                sel = (ticks >= s['start_min']) & (ticks <= s['end_min'])
                tick_act[sel] = name_to_idx[s['activity_name']]
            horizon_ticks = 18  # 3 h
            probs, gap_found = None, None
            for step in range(1, horizon_ticks + 1):
                logits, _ = self.forward(tick_act.unsqueeze(0),
                                         ticks.unsqueeze(0).float())
                probs = F.softmax(logits[0, -1], dim=-1)
                pred = int(probs.argmax())
                if pred != self.idle_idx:
                    gap_found = (step - 1) * TICK_MIN
                    break
                ticks = torch.cat([ticks, ticks[-1:] + TICK_MIN])
                tick_act = torch.cat([tick_act, torch.tensor([pred])])
            gap_min = {n: gap_found for n in self.vocab} \
                if gap_found is not None else {}
        out = {n: float(probs[i]) for i, n in enumerate(self.vocab)}
        if self.mode == 'interval':
            out['__idle__'] = float(probs[self.idle_idx])
        return out, gap_min


# ---------------------------------------------------------------------------
# Dataset assembly from the curator's tagged days
# ---------------------------------------------------------------------------

def _event_sample(day: dict, name_to_idx: dict, bos_idx: int) -> dict:
    """One day -> input positions [BOS, seg_1..seg_{S-1}] predicting
    seg_1..seg_S (+ gap to each target's start)."""
    segs = day['segments']
    idxs = [name_to_idx[s['activity_name']] for s in segs]
    starts = [s['start_min'] for s in segs]
    ends = [s['end_min'] for s in segs]
    def start_tick(t):
        return int(min(max((t - DAY_START_MIN) // TICK_MIN, 0),
                       (DAY_END_MIN - DAY_START_MIN) // TICK_MIN - 1))
    return {
        'act': torch.tensor([bos_idx] + idxs[:-1], dtype=torch.int64),
        't_start': torch.tensor([DAY_START_MIN] + starts[:-1]),
        't_end': torch.tensor([DAY_START_MIN] + ends[:-1]),
        'target': torch.tensor(idxs, dtype=torch.int64),
        # absolute clock tick of each target segment's start
        'target_tick': torch.tensor([start_tick(s) for s in starts],
                                    dtype=torch.int64),
        'weight': torch.ones(len(segs)),
    }


def _interval_sample(day: dict, name_to_idx: dict, idle_idx: int,
                     change_weight: float = 5.0) -> dict:
    last_end = max(s['end_min'] for s in day['segments'])
    ticks = torch.arange(DAY_START_MIN, last_end + TICK_MIN, TICK_MIN)
    tick_act = torch.full((len(ticks),), idle_idx, dtype=torch.int64)
    for s in day['segments']:
        sel = (ticks >= s['start_min']) & (ticks <= s['end_min'])
        tick_act[sel] = name_to_idx[s['activity_name']]
    target = tick_act[1:]
    weight = torch.ones(len(target))
    weight[target != tick_act[:-1]] = change_weight
    return {
        'act': tick_act[:-1],
        't_start': ticks[:-1].float(),
        't_end': None,
        'target': target,
        'target_tick': torch.zeros(len(target), dtype=torch.int64),
        'weight': weight,
    }


def build_samples(tagged_days: list, vocab: list, mode: str) -> list:
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    v = len(vocab)
    samples = []
    for day in tagged_days:
        if not day['segments']:
            raise RuntimeError(f"day {day.get('day')} has no segments")
        if mode == 'event':
            samples.append(_event_sample(day, name_to_idx, bos_idx=v + 1))
        else:
            samples.append(_interval_sample(day, name_to_idx, idle_idx=v))
    return samples


def _collate(samples: list, mode: str) -> dict:
    B = len(samples)
    T = max(len(s['act']) for s in samples)
    batch = {
        'act': torch.zeros(B, T, dtype=torch.int64),
        't_start': torch.zeros(B, T),
        't_end': torch.zeros(B, T),
        'target': torch.zeros(B, T, dtype=torch.int64),
        'target_tick': torch.zeros(B, T, dtype=torch.int64),
        'weight': torch.zeros(B, T),   # 0 on padding -> masked out of loss
        'pad_mask': torch.ones(B, T, dtype=torch.bool),
    }
    for b, s in enumerate(samples):
        L = len(s['act'])
        batch['act'][b, :L] = s['act']
        batch['t_start'][b, :L] = s['t_start']
        if mode == 'event':
            batch['t_end'][b, :L] = s['t_end']
        batch['target'][b, :L] = s['target']
        batch['target_tick'][b, :L] = s['target_tick']
        batch['weight'][b, :L] = s['weight']
        batch['pad_mask'][b, :L] = False
    return batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _losses_and_metrics(model: ActivityTransformer, batch: dict,
                        time_loss_weight: float) -> tuple:
    logits, time_logits = model(batch['act'], batch['t_start'],
                                batch['t_end'] if model.mode == 'event' else None,
                                pad_mask=batch['pad_mask'])
    w = batch['weight']
    valid = w > 0
    ce = F.cross_entropy(logits[valid], batch['target'][valid],
                         reduction='none')
    loss = (ce * w[valid]).sum() / w[valid].sum()
    if model.mode == 'event':
        # train the clock head only on the activity that actually came next:
        # gather (B,T,V,108) at the target activity -> (n_valid, 108)
        tgt_act = batch['target'][valid]                  # (n_valid,)
        tl = time_logits[valid]                           # (n_valid, V, 108)
        tl_at_tgt = tl.gather(
            1, tgt_act.view(-1, 1, 1).expand(-1, 1, tl.shape[-1])).squeeze(1)
        time_ce = F.cross_entropy(tl_at_tgt, batch['target_tick'][valid])
        loss = loss + time_loss_weight * time_ce
    with torch.no_grad():
        pred = logits[valid].argmax(-1)
        tgt = batch['target'][valid]
        top1 = (pred == tgt).float().mean().item()
        k = min(3, logits.shape[-1])
        topk = logits[valid].topk(k, dim=-1).indices
        top3 = (topk == tgt.unsqueeze(-1)).any(-1).float().mean().item()
        if model.mode == 'event':
            tl = time_logits[valid]
            tl_at_tgt = tl.gather(
                1, batch['target'][valid].view(-1, 1, 1)
                .expand(-1, 1, tl.shape[-1])).squeeze(1)
            pred_tick = tl_at_tgt.argmax(-1)
            start_mae = (pred_tick - batch['target_tick'][valid]).abs() \
                .float().mean().item() * TICK_MIN
        else:
            start_mae = None
    return loss, {'top1': top1, 'top3': top3, 'start_mae_min': start_mae}


def _noisy_day(day: dict, vocab: list, rng) -> dict:
    """Tagger-noise augmentation: the deployed model consumes the eval
    tagger's imperfect output, not clean curation. Mimic that noise on a
    training day: drop, mislabel, or time-jitter segments."""
    p = 0.1
    segs = []
    for s in day['segments']:
        r = rng.random()
        if r < p and len(day['segments']) > 2:
            continue                                    # dropped segment
        s = dict(s)
        if r < 2 * p:
            s['activity_name'] = vocab[rng.randrange(len(vocab))]  # mislabel
        jit = (rng.random() - 0.5) * 20                 # +-10 min jitter
        s['start_min'] = s['start_min'] + jit
        s['end_min'] = max(s['end_min'] + jit, s['start_min'] + 1)
        segs.append(s)
    return {**day, 'segments': segs or day['segments']}


# Training hyperparameters. These were config keys that nothing ever varied,
# so they live here instead. Values are exactly what config/base.yaml supplied.
TRAIN_VAL_DAYS = 7
TRAIN_LR = 1e-3
TRAIN_TIME_LOSS_WEIGHT = 0.5
TRAIN_EARLY_STOP_PATIENCE = 50
TRAIN_NOISE_AUG_COPIES = 2   # NB: the old cfg.get() fallback here was 0, but
                             # base.yaml supplied 2 -- 2 is what every run used.


def train_activity_transformer(tagged_days: list, embeddings: dict,
                               cfg: dict, out_path: str,
                               device: str = None,
                               day_weights: list = None,
                               init_state: dict = None) -> dict:
    """Train on curated tagged days; temporal split (last cfg['val_days']
    days = validation). Saves checkpoint to out_path; returns val metrics.

    v2-PF continual learning:
    - day_weights: per-day recency weights aligned with tagged_days (the
      graph decay's gamma^age, floored at 0.1 so old rare activities don't
      vanish). Scales each day's per-position loss weight; validation days
      stay UNWEIGHTED (honest early stopping).
    - init_state: previous round's state_dict for warm start; entries with
      mismatched shapes (vocab-sized buffers after vocabulary growth) are
      skipped loudly."""
    vocab = sorted({s['activity_name'] for d in tagged_days
                    for s in d['segments']})
    missing = [n for n in vocab if n not in embeddings['vectors']]
    if missing:
        raise RuntimeError(f"activity names without embeddings: {missing}")
    name_emb = torch.tensor([embeddings['vectors'][n] for n in vocab])

    mode = cfg['mode']
    model = ActivityTransformer(
        vocab, name_emb, mode,
)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if init_state is not None:
        own = model.state_dict()
        usable = {k: v for k, v in init_state.items()
                  if k in own and own[k].shape == v.shape}
        skipped = [k for k in init_state if k not in usable]
        model.load_state_dict(usable, strict=False)
        if skipped:
            print(f"  [transformer] warm start: skipped shape-mismatched "
                  f"entries (vocab growth): {skipped}")

    val_days = min(cfg.get('val_days', TRAIN_VAL_DAYS), max(1, len(tagged_days) // 3))
    train_days = tagged_days[:-val_days]
    train_w = (list(day_weights[:-val_days])
               if day_weights is not None else None)
    n_aug = int(cfg.get('noise_aug_copies', TRAIN_NOISE_AUG_COPIES))
    if n_aug and mode == 'event':
        import random as _random
        rng = _random.Random(0)   # deterministic (resume-safe)
        aug = [_noisy_day(d, vocab, rng)
               for _ in range(n_aug) for d in train_days]
        print(f"  [transformer] noise augmentation: +{len(aug)} noisy day "
              f"copies (tagger-noise robustness)")
        if train_w is not None:
            train_w = train_w + [w for _ in range(n_aug) for w in train_w]
        train_days = train_days + aug
    train_s = build_samples(train_days, vocab, mode)
    if train_w is not None:
        floor = 0.1   # uniform replay floor: old rare activities survive
        for s, w in zip(train_s, train_w):
            s['weight'] = s['weight'] * max(float(w), floor)
    val_s = build_samples(tagged_days[-val_days:], vocab, mode)
    if not train_s:
        raise RuntimeError("no training days after the validation split")
    train_b = {k: v.to(device) for k, v in _collate(train_s, mode).items()}
    val_b = {k: v.to(device) for k, v in _collate(val_s, mode).items()}

    max_epochs = int(os.environ.get('MAX_TRAIN_EPOCHS', '400'))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', TRAIN_LR))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    time_w = cfg.get('time_loss_weight', TRAIN_TIME_LOSS_WEIGHT)
    patience = cfg.get('early_stop_patience', TRAIN_EARLY_STOP_PATIENCE)
    best = {'val_loss': float('inf'), 'epoch': -1}
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        loss, train_m = _losses_and_metrics(model, train_b, time_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_m = _losses_and_metrics(model, val_b, time_w)
        if float(val_loss) < best['val_loss'] - 1e-5:
            best = {'val_loss': float(val_loss), 'epoch': epoch,
                    'train_loss': float(loss), **{f'val_{k}': v
                                                  for k, v in val_m.items()},
                    **{f'train_{k}': v for k, v in train_m.items()}}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 25 == 0:
            print(f"  [transformer] epoch {epoch}: train_loss={float(loss):.4f} "
                  f"val_loss={float(val_loss):.4f} val_top1={val_m['top1']:.3f} "
                  f"val_top3={val_m['top3']:.3f}")
        if epoch - best['epoch'] >= patience:
            print(f"  [transformer] early stop at epoch {epoch} "
                  f"(best epoch {best['epoch']})")
            break
    if best_state is None:
        raise RuntimeError("training produced no finite validation loss")
    model.load_state_dict(best_state)

    ckpt = {
        'state_dict': best_state,
        'config': cfg,
        'vocab': vocab,
        'name_embeddings': name_emb,
        'embedding_model': embeddings['model'],
        'mode': mode,
        'val_metrics': best,
        'n_train_days': len(train_s),
        'n_val_days': len(val_s),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"  [transformer] saved {out_path}; best={best}")
    return best


def load_activity_transformer(ckpt_path: str,
                              device: str = 'cpu') -> ActivityTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = ActivityTransformer(
        ckpt['vocab'], ckpt['name_embeddings'], ckpt['mode'],
)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model
