"""Goal Proposal Network for the NOPA baseline (Puig et al., ICRA 2023, Fig. 2b).

Learns ``q(g | s0, s_t)``: given the episode's initial state and the current
state, a distribution over the count of every predicate in the day's goal. The
policy samples K goal proposals from it (Alg. 2 line 13).

Architecture mirrors the paper: a 4-layer FC trunk over the flattened
``Delta s_t`` matrix down to ``h = 128``, then a 2-layer MLP head emitting a
per-predicate softmax over count bins. The paper uses ``p = 136`` predicate
types and ``c = 9`` counts; here ``p`` is the observed ``(obj, dest)``
vocabulary (84-109 on HOMER+ A/B/C) and ``c = 4`` (measured max repeats/day = 3).

Per the paper, the GPN is trained on episodes in which **only the main agent
acts** — so that it stays robust when the helper starts perturbing the state.
``RobotPolicyNOPA.train`` enforces that by feeding only warm-start days.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from agents.nopa_utils import N_TIME_FEATURES


class GoalProposalNetwork(nn.Module):
    """4-layer FC trunk + 2-layer MLP count head (paper Fig. 2b)."""

    def __init__(self, n_predicates: int, n_counts: int, hidden: int = 256,
                 h: int = 128, head_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_predicates = n_predicates
        self.n_counts = n_counts
        in_dim = n_predicates * (n_counts + 1) + N_TIME_FEATURES
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(h, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, n_predicates * n_counts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim)`` -> count logits ``(B, P, C)``."""
        return self.head(self.trunk(x)).view(-1, self.n_predicates, self.n_counts)


def _metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """Per-predicate count accuracy plus goal-SET precision/recall/F1.

    The goal-set F1 (``count >= 1`` vs ``true count >= 1``) is the number that
    actually predicts downstream helping quality; count accuracy is inflated by
    the many always-zero predicates.
    """
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    p_pos, y_pos = pred >= 1, y >= 1
    tp = (p_pos & y_pos).sum().item()
    fp = (p_pos & ~y_pos).sum().item()
    fn = (~p_pos & y_pos).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return {'count_acc': acc, 'goal_prec': prec, 'goal_rec': rec,
            'goal_f1': 2 * tp / (2 * tp + fp + fn + 1e-9)}


def count_bin_weights(y: torch.Tensor, n_counts: int) -> torch.Tensor:
    """Inverse-frequency weights over count bins.

    Task-level goals are ~2.6% dense against a ~100-predicate vocabulary (vs
    ~63% for whole-day goals), so unweighted cross-entropy is minimised by
    predicting count 0 everywhere: that scores ~97% bin accuracy and goal-F1
    exactly 0. These weights rebalance the bins so the nonzero counts carry
    comparable gradient.
    """
    cnt = torch.bincount(y.reshape(-1), minlength=n_counts).float()
    w = cnt.sum() / (cnt.clamp(min=1.0) * n_counts)
    # Bins with no training samples would otherwise get an enormous weight
    # (measured 12671 for the empty 2/3+ bins in task mode, where a predicate
    # fires at most once per task). They never contribute to the loss as things
    # stand, but a single repeated predicate would then blow up the gradient.
    w[cnt == 0] = 1.0
    return w


def _loss(logits: torch.Tensor, y: torch.Tensor, mode: str = 'ce',
          weight: torch.Tensor | None = None,
          gamma: float = 2.0) -> torch.Tensor:
    """Loss over (batch, predicate) of the count softmax.

    mode 'ce'          : plain cross-entropy (the original).
    mode 'weighted_ce' : cross-entropy with inverse-frequency bin weights.
    mode 'focal'       : focal loss, (1-p)^gamma * -log p, optionally weighted.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    if mode == 'focal':
        logp = nn.functional.log_softmax(flat_logits, dim=-1)
        lp = logp.gather(1, flat_y.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - lp.exp()) ** gamma) * lp
        if weight is not None:
            loss = loss * weight[flat_y]
        return loss.mean()
    if mode not in ('ce', 'weighted_ce'):
        raise ValueError(f"nopa gpn loss must be ce|weighted_ce|focal, got {mode!r}")
    return nn.functional.cross_entropy(
        flat_logits, flat_y, weight=weight if mode == 'weighted_ce' else None)


# Training hyperparameters -- previously config keys nothing ever varied.
# Values are exactly what config/base.yaml supplied.
GPN_N_COUNTS = 4
GPN_VAL_DAYS = 5
GPN_BATCH_SIZE = 256
GPN_LR = 1e-3
GPN_EARLY_STOP_PATIENCE = 50


def train_gpn(x: torch.Tensor, y: torch.Tensor, day_of_sample: list,
              n_days: int, cfg: dict, out_path: str, extra: dict) -> dict:
    """Fit the GPN and save a self-contained checkpoint at *out_path*.

    Args:
        x: ``(N, in_dim)`` encoded inputs, one row per ``(day, tick)``.
        y: ``(N, P)`` long count labels (constant across the ticks of a day).
        day_of_sample: day index per row, used for the temporal split.
        n_days: number of distinct training days.
        cfg: ``robot.nopa.gpn`` config block.
        extra: fields merged into the checkpoint (predicates, clock, ...).

    The split is temporal (last ``val_days`` days held out), matching
    ``model/activity_transformer.py``. Returns the best-epoch metrics dict.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_pred, n_counts = y.shape[1], int(cfg.get('n_counts', GPN_N_COUNTS))

    val_days = min(int(cfg.get('val_days', GPN_VAL_DAYS)), max(1, n_days // 3))
    split_day = n_days - val_days
    day_arr = torch.tensor(day_of_sample, dtype=torch.long)
    tr_idx = (day_arr < split_day).nonzero(as_tuple=False).squeeze(-1)
    va_idx = (day_arr >= split_day).nonzero(as_tuple=False).squeeze(-1)
    if tr_idx.numel() == 0 or va_idx.numel() == 0:
        raise RuntimeError(
            f'nopa GPN: empty split (n_days={n_days}, val_days={val_days}, '
            f'train={tr_idx.numel()}, val={va_idx.numel()})')

    x_tr, y_tr = x[tr_idx].to(device), y[tr_idx].to(device)
    x_va, y_va = x[va_idx].to(device), y[va_idx].to(device)

    model = GoalProposalNetwork(
        n_pred, n_counts
    ).to(device)

    loss_mode = str(cfg.get('loss', 'ce'))
    focal_gamma = float(cfg.get('focal_gamma', 2.0))
    bin_w = (count_bin_weights(y_tr, n_counts).to(device)
             if loss_mode in ('weighted_ce', 'focal') else None)
    print(f"  [nopa-gpn] loss={loss_mode}"
          + (f" gamma={focal_gamma}" if loss_mode == 'focal' else "")
          + (f" bin_weights={[round(float(w), 2) for w in bin_w]}" if bin_w is not None else ""))

    max_epochs = int(os.environ.get('MAX_TRAIN_EPOCHS', '400'))
    batch_size = int(cfg.get('batch_size', GPN_BATCH_SIZE))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get('lr', GPN_LR)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    patience = int(cfg.get('early_stop_patience', GPN_EARLY_STOP_PATIENCE))
    gen = torch.Generator().manual_seed(0)   # deterministic / resume-safe

    best = {'val_loss': float('inf'), 'epoch': -1}
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(x_tr.shape[0], generator=gen).to(device)
        tot, nb = 0.0, 0
        for s in range(0, perm.numel(), batch_size):
            idx = perm[s:s + batch_size]
            loss = _loss(model(x_tr[idx]), y_tr[idx],
                         loss_mode, bin_w, focal_gamma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(x_va)
            val_loss = float(_loss(va_logits, y_va,
                                   loss_mode, bin_w, focal_gamma))
            val_m = _metrics(va_logits, y_va)
        if val_loss < best['val_loss'] - 1e-5:
            best = {'val_loss': val_loss, 'epoch': epoch,
                    'train_loss': tot / max(nb, 1),
                    **{f'val_{k}': v for k, v in val_m.items()}}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 25 == 0:
            print(f"  [nopa-gpn] epoch {epoch}: train_loss={tot / max(nb, 1):.4f} "
                  f"val_loss={val_loss:.4f} val_goal_f1={val_m['goal_f1']:.3f} "
                  f"val_count_acc={val_m['count_acc']:.3f}")
        if epoch - best['epoch'] >= patience:
            print(f"  [nopa-gpn] early stop at epoch {epoch} "
                  f"(best epoch {best['epoch']})")
            break
    if best_state is None:
        raise RuntimeError('nopa GPN: training produced no finite validation loss')
    model.load_state_dict(best_state)

    ckpt = {
        'state_dict': best_state,
        'config': dict(cfg),
        'n_predicates': n_pred,
        'n_counts': n_counts,
        'val_metrics': best,
        'n_train_samples': int(tr_idx.numel()),
        'n_val_samples': int(va_idx.numel()),
        'n_train_days': split_day,
        'n_val_days': val_days,
        **extra,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"  [nopa-gpn] saved {out_path}; best={best}")
    return best


def load_gpn(ckpt_path: str, device: str = 'cpu') -> tuple:
    """Load ``(model, ckpt)``. The checkpoint carries vocab + clock prior too."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = GoalProposalNetwork(
        ckpt['n_predicates'], ckpt['n_counts'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt
