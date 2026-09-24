"""Hybrid LLM + transformer robot policy.

Training (train(), called once by the closed-loop driver at the warmup
boundary): the LLM memory curator segments and names each warmup day's
object-relocation stream (open vocabulary, no ground-truth labels), code
builds branching per-activity task graphs from the tags, and a small
transformer is trained to predict upcoming activities from language
embeddings of the curated names.

Evaluation (get_actions(), every tick): derive the human's moves from
consecutive history frames (subtracting the robot's own actions), recognize
the current/upcoming activity with a particle filter over the task graphs
(no eval-time LLM), and consult the graphs for the most probable next
action(s) to pre-execute. (The old LLM-tagger recognition was removed — see
docs/failed_experiments.md.)

GT guard: the history shim stores world_state['activity'] (the driver needs
it to write the gen-dataset .pt for post-hoc analysis) but this policy never
reads it for inference.

Driver contract (scripts/closed_loop_eval.py): get_actions returns
(actions, (obj_idxs, loc_idxs, prev_loc_idxs)) in slate index space; the
policy exposes .history, .common_data, ._last_fwd, ._update_history(),
.train(), .data_quantity, .reset().
"""

import json
import os

import torch

from agents.activity_particle_filter import GraphParticleFilter, BG_LABEL
from agents.llm_memory_curator import MemoryCurator
from agents.robot_policy import RobotPolicyBase, RobotMoveAction
from agents.task_graph import TaskGraphSet, START
from agents.slatepro_utils import (
    load_common_data, filter_world_state_to_homer_nodes,
    world_state_to_history_initialization, update_history_edges,
    update_history_activity, update_history_time)
from model.activity_transformer import TRAIN_VAL_DAYS, load_activity_transformer

# v6 learned-lead constants. LEAD_ALPHA = target revert rate among engaged
# actions = the metric break-even (act early enough that ≤1/3 of engaged
# placements get reverted). LEAD_STEP is constant (un-annealed) so the lead
# tracks pace drift instead of freezing.
LEAD_ALPHA = 1.0 / 3.0
LEAD_STEP = 5.0          # minutes per day-update
LEAD_MIN = 0.0
LEAD_MAX = 120.0

# --- Acceptance-learning gate (opt-in via policy.accept_gate) ---
# A human move OFF a robot placement within REACTION_WINDOW_MIN of when the robot
# placed it is treated as a reaction (redirect -> reject); a later move is a
# genuine continuation (accept). The human reviews synchronously, so reactions
# land within ~1-2 ten-minute ticks.
REACTION_WINDOW_MIN = 20.0
ACCEPT_BACKOFF_BASE_MIN = 20.0   # within-day exp backoff: wait after the 1st reject
ACCEPT_BACKOFF_MAX_MIN = 60.0    # doubles per repeat, capped here
ACCEPT_PRIOR = 2.0               # optimistic prior: unseen (obj,dest) -> P_accept=1.0


class RobotPolicyHybrid(RobotPolicyBase):
    """Hybrid policy: LLM-curated task graphs + PF recognition + activity transformer.

    TL;DR: train() curates warmup days; get_actions() diffs human moves, runs PF,
    emits graph-predicted teleports gated by transformer prep + safety stack.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = config['robot']['hybrid']
        household = config['benchmark']['household']
        common_data_path = self.cfg['common_data_path'].format(
            household=household)
        self.common_data = load_common_data(common_data_path)
        self.memory_dir = self.cfg['memory_dir']
        self.cl_cfg = self.cfg.get('continual_learning', {}) or {}
        self.curator = MemoryCurator(self.cfg['llm'], self.cfg['embedding'],
                                     self.memory_dir, cl_cfg=self.cl_cfg)
        self.policy_cfg = self.cfg['policy']
        # Recognition is a particle filter over the task graphs (no eval-time
        # LLM). The old LLM-tagger path was removed — see
        # docs/failed_experiments.md "LLM+transformer Hybrid model".
        self.pf = None
        self.transformer = None
        self.task_graphs = None
        self.data_quantity = None
        self._last_fwd = None
        self.history = None
        # Eval-time tag audit: one JSON line per LLM tagging call. Appends on
        # resume (like llm_audit_*.txt) — delete on a clean code-fix restart.
        self._tag_log_path = os.path.join(self.memory_dir, 'eval_tag_log.jsonl')
        self._diag_log_path = os.path.join(self.memory_dir, 'policy_diag.jsonl')
        self._day_counter = -1
        # v6 learned per-key lead time (act,obj,dest) -> minutes. Persists
        # across days (NOT cleared by _reset_day_state). Online pinball/SGD:
        # converges each key to the largest lead with bounded revert rate.
        self._lead_time = {}
        # Lead-time gate (default ON): a candidate whose expected moment is more
        # than `lead` minutes ahead of now is dropped as too-early (prevents
        # acting hours before the human, which gets undone). Ablation
        # (lead_time_gate=False) removes this timing suppression entirely.
        self._lead_gate = bool(self.policy_cfg.get('lead_time_gate', True))
        # Precedence gate (default ON): suppress a candidate in proportion to its
        # same-object graph prerequisites not yet being satisfied (e.g. don't stow
        # an object that still has a pending use). Ablation (precedence_gate=False)
        # removes this ordering suppression at both call sites (_gate_precedence).
        self._precedence_gate = bool(self.policy_cfg.get('precedence_gate', True))
        # Prep branch (default ON): proactively pre-stage upcoming activities the
        # transformer predicts are imminent. Ablation (prep_branch=False) makes
        # the policy purely reactive (PF recognition of currently-open activities
        # only). NOTE: the activity transformer's only consumer is this branch, so
        # prep_branch=False is also the effective "no transformer" ablation.
        self._prep = bool(self.policy_cfg.get('prep_branch', True))
        # Current branch (default ON): task-graph assistance for the PF-recognized
        # in-progress activity. Ablation (current_branch=False) = "only prep":
        # act solely on transformer-anticipated upcoming activities.
        self._current = bool(self.policy_cfg.get('current_branch', True))
        # PF recognition mode (default full posterior). Ablation (pf_map=True)
        # collapses the open-activity posterior to its MAP: only the single
        # most-likely-open activity contributes, at weight 1.0 — no posterior
        # averaging over competing hypotheses.
        self._pf_map = bool(self.policy_cfg.get('pf_map', False))
        # Acceptance-learning gate. Cross-day gamma-decayed accept/reject weights
        # per (obj_idx, dest_idx); persist across days like _lead_time.
        self._accept_gate = bool(self.policy_cfg.get('accept_gate', False))
        # v2 (build-upon): credit an accept ONLY when the human builds upon the
        # placement ('continued'); a placement that merely stayed put ('active',
        # i.e. the human was indifferent) is neutral, not an accept.
        self._accept_buildon = bool(self.policy_cfg.get('accept_buildon', False))
        self._accept_gamma = float(self.policy_cfg.get('accept_gamma', 0.9))
        self._acc_w = {}
        self._rej_w = {}
        self._cur_t = 0.0
        self._reset_day_state()

    def _reset_day_state(self):
        """Clear per-day emit/revert/lead tracking (called from ``reset``)."""
        self._day_events = []          # curator-format events observed today
        self._emitted_last_tick = set()
        # Per-object stack of robot placements (Option 3 / B15 fix). Each entry:
        # {ev_idx, dest_idx, from_idx, act, status} where status is
        # 'active' | 'superseded' | 'reverted' | 'continued'.
        self._robot_placements: dict[int, list[dict]] = {}
        self._lead_emitted = set()     # (act,obj,dest) emitted today (v6)
        self._lead_reverted = set()    # human-undid (v6)
        self._lead_superseded = set()  # robot moved same object again (v6)
        # Acceptance-gate within-day state (exp backoff + this-day reject counts).
        self._reject_until = {}   # (obj,dest) -> block re-emit until this t_min
        self._reject_n = {}       # (obj,dest) -> rejections so far today

    def _active_placements(self, obj_idx: int) -> list[dict]:
        """All still-active robot placements for *obj_idx* (FIFO order)."""
        return [p for p in self._robot_placements.get(obj_idx, [])
                if p['status'] == 'active']

    def _latest_active_placement(self, obj_idx: int) -> dict | None:
        actives = self._active_placements(obj_idx)
        return actives[-1] if actives else None

    def _supersede_active_placements(self, obj_idx: int) -> None:
        """Mark every active placement on *obj_idx* superseded (robot churn)."""
        for p in self._active_placements(obj_idx):
            p['status'] = 'superseded'
            self._day_events[p['ev_idx']]['superseded'] = True
            act = p.get('act')
            if act is not None:
                self._lead_superseded.add((act, obj_idx, p['dest_idx']))

    def _find_revert_placement(self, obj_idx: int, prev_idx: int,
                                dest_idx: int) -> dict | None:
        """Match human undo: was at robot dest (*prev_idx*), moved back to *dest_idx*."""
        for p in reversed(self._active_placements(obj_idx)):
            if p['dest_idx'] == prev_idx and p['from_idx'] == dest_idx:
                return p
        return None

    def _consume_continuation(self, obj_idx: int, prev_idx: int) -> dict | None:
        """Human moved object onward from a robot placement — accept, not revert."""
        for p in reversed(self._active_placements(obj_idx)):
            if p['dest_idx'] == prev_idx:
                p['status'] = 'continued'
                return p
        return None

    def _register_reject(self, obj_idx: int, dest_idx: int) -> None:
        """Record a human rejection of the robot placing obj@dest: short-lived
        exponential within-day backoff + a same-day reject so the cross-day
        acceptance prob learns it. Uses self._cur_t (set in get_actions)."""
        key = (obj_idx, dest_idx)
        n = self._reject_n.get(key, 0) + 1
        self._reject_n[key] = n
        dt = min(ACCEPT_BACKOFF_MAX_MIN, ACCEPT_BACKOFF_BASE_MIN * (2 ** (n - 1)))
        self._reject_until[key] = self._cur_t + dt

    def _resolve_placements(self, t_now: float) -> None:
        """Acceptance gate: resolve each robot placement one tick after it was
        emitted. The harness suppresses the robot's move when the human undoes/
        redirects, so a placement that did NOT materialize (object not at its
        dest now) was rejected; one that materialized was allowed (accept).
        Resolved at most once per placement."""
        cur = self.history['edges'][0, -1].argmax(dim=-1)
        for obj_idx, plist in self._robot_placements.items():
            for p in plist:
                if p.get('_resolved') or p['status'] != 'active':
                    continue
                if p.get('t_emit', t_now) >= t_now:
                    continue  # emitted this tick; human review not yet reflected
                p['_resolved'] = True
                if int(cur[obj_idx]) != p['dest_idx']:
                    p['status'] = 'rejected'
                    self._register_reject(obj_idx, p['dest_idx'])

    def _p_accept(self, obj_idx: int, dest_idx: int) -> float:
        """Laplace-smoothed, gamma-decayed acceptance probability for obj@dest.
        Optimistic prior: unseen -> 1.0; decays toward 0 as rejects accumulate."""
        key = (obj_idx, dest_idx)
        a = self._acc_w.get(key, 0.0)
        r = self._rej_w.get(key, 0.0)
        return (a + ACCEPT_PRIOR) / (a + r + ACCEPT_PRIOR)

    def _update_accept_counts(self) -> None:
        """Day boundary: fold the day's placement outcomes into the gamma-decayed
        cross-day weights. accept = continued (+ active unless accept_buildon) ;
        reject = reverted+redirected+rejected ; superseded excluded (robot self-
        churn). Under accept_buildon, 'active' (object stayed put but the human
        never built upon it = indifference) is also excluded, not credited."""
        day_acc, day_rej = {}, {}
        for obj_idx, plist in self._robot_placements.items():
            for p in plist:
                key = (obj_idx, p['dest_idx'])
                st = p['status']
                if st == 'continued' or (st == 'active' and not self._accept_buildon):
                    day_acc[key] = day_acc.get(key, 0) + 1
                elif st in ('reverted', 'redirected', 'rejected'):
                    day_rej[key] = day_rej.get(key, 0) + 1
        g = self._accept_gamma
        keys = set(self._acc_w) | set(self._rej_w) | set(day_acc) | set(day_rej)
        for key in keys:
            self._acc_w[key] = self._acc_w.get(key, 0.0) * g + day_acc.get(key, 0)
            self._rej_w[key] = self._rej_w.get(key, 0.0) * g + day_rej.get(key, 0)

    # ------------------------------------------------------------------
    # Driver contract: history shim (same calls as RobotPolicySlatepro)
    # ------------------------------------------------------------------

    def _update_history(self, world_state: dict) -> None:
        """Append one timestep to the SLaTe-format history shim (driver contract)."""
        world_state = filter_world_state_to_homer_nodes(
            world_state, self.common_data)
        if self.history is None:
            self.history = world_state_to_history_initialization(
                world_state, self.common_data)
        else:
            self.history['edges'] = update_history_edges(
                world_state, self.history['edges'],
                self.history['vh_id_to_slate_idx'], self.common_data)
            # activity_features: kept ONLY so the driver can write the GT
            # activity into the gen-dataset .pt (post-hoc analysis). This
            # policy never reads it.
            self.history['activity_features'] = update_history_activity(
                world_state, self.history['activity_features'],
                self.common_data['activities'])
            self.history['time_features'], self.history['times'] = \
                update_history_time(world_state,
                                    self.history['time_features'],
                                    self.history['times'])
            self.history['sequence_length'] += 1

    def _update_lead_times(self):
        """v6: pinball update per emitted key. Success = neither reverted nor superseded."""
        dflt = self.policy_cfg['act_window_min']
        bad = self._lead_reverted | self._lead_superseded
        for key in self._lead_emitted:
            dt = self._lead_time.get(key, dflt)
            if key in bad:
                dt -= LEAD_STEP * (1.0 - LEAD_ALPHA)
            else:
                dt += LEAD_STEP * LEAD_ALPHA
            self._lead_time[key] = min(LEAD_MAX, max(LEAD_MIN, dt))
        if self._lead_emitted:
            n_ok = len(self._lead_emitted - bad)
            pct = 100.0 * n_ok / len(self._lead_emitted)
            print(f"[hybrid] v6 lead day {self._day_counter}: "
                  f"success {n_ok}/{len(self._lead_emitted)} ({pct:.0f}%) "
                  f"(excl reverted+superseded)")

    def reset(self):
        """Day boundary: persist attributed log, update v6 leads, reset PF."""
        # v2-PF: persist the finished day's attributed event log (actor field
        # distinguishes robot moves) for the curator's actor-split counts.
        if self.cl_cfg.get('enabled') and self._day_events:
            d = os.path.join(self.memory_dir, 'attributed_days')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f'{self._day_counter:03d}.json'),
                      'w') as f:
                json.dump(self._day_events, f)
        self._update_lead_times()
        if self._accept_gate:
            self._update_accept_counts()
        self.history = None
        self._last_fwd = None
        self._day_counter += 1
        self._reset_day_state()
        if self.pf is not None:
            self.pf.reset()

    # ------------------------------------------------------------------
    # Training (curate -> embed -> graphs -> transformer)
    # ------------------------------------------------------------------

    def train(self, dataset_path: str, output_dir: str, days: int) -> None:
        """Curate warmup .pt dir → embeddings → graphs → train transformer."""
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')
        done_marker = os.path.join(output_dir, f'hybrid_training_done_day{days}')
        if os.path.isfile(done_marker) and os.path.isfile(weights_path):
            print(f"Hybrid training for day {days} already complete, loading.")
            self.load_artifacts(weights_path)
            self.data_quantity = {'train': -1, 'val': -1}
            return

        pt_dir = os.path.join(dataset_path, 'train')
        print(f"[hybrid] curating warmup days from {pt_dir} "
              f"into {self.memory_dir}")
        self.curator.curate_dataset(pt_dir)
        print(f"[hybrid] curator stats: {self.curator.stats()}")

        tagged_days = self.curator.load_tagged_days()
        embeddings = self.curator.build_embeddings()
        from model.activity_transformer import train_activity_transformer
        day_weights, init_state = None, None
        if self.cl_cfg.get('enabled'):
            # v2-PF: recency weights mirror the graph decay (same gamma,
            # chosen prequentially in build_task_graphs during curation);
            # warm-start from the previous round's checkpoint.
            gamma = self.curator.last_gamma
            # transformer_recency (default on): mirror the graph γ-decay in the
            # transformer's day weights. Toggle off to ablate whether the
            # transformer's recency weighting matters (graph decay unaffected).
            if gamma is not None and self.cl_cfg.get('transformer_recency', True):
                n = len(tagged_days)
                day_weights = [gamma ** (n - 1 - d) for d in range(n)]
            import glob as _glob
            import re as _re

            def _day_num(p):
                m = _re.search(r'weights_day(\d+)\.pt$', p)
                return int(m.group(1)) if m else -1
            prev = sorted(_glob.glob(os.path.join(output_dir,
                                                  'weights_day*.pt')),
                          key=_day_num)
            prev = [p for p in prev if p != weights_path]
            if prev:
                init_state = torch.load(prev[-1], map_location='cpu',
                                        weights_only=False)['state_dict']
                print(f"[hybrid] v2-PF warm start from {prev[-1]}")
        metrics = train_activity_transformer(
            tagged_days, embeddings, self.cfg['transformer'], weights_path,
            day_weights=day_weights, init_state=init_state)
        n_val = min(self.cfg['transformer'].get('val_days', TRAIN_VAL_DAYS),
                    max(1, len(tagged_days) // 3))
        self.data_quantity = {'train': len(tagged_days) - n_val, 'val': n_val}
        with open(done_marker, 'w') as f:
            f.write(str(metrics) + '\n')
        self.load_artifacts(weights_path)

    def load_artifacts(self, weights_path: str) -> None:
        """Load the transformer checkpoint + task graphs (resume path)."""
        self.transformer = load_activity_transformer(weights_path)
        graphs_path = os.path.join(self.memory_dir, 'task_graphs.json')
        self.task_graphs = TaskGraphSet.load(graphs_path)
        print(f"[hybrid] loaded transformer ({self.transformer.mode}, "
              f"vocab={len(self.transformer.vocab)}) from {weights_path} and "
              f"{len(self.task_graphs.graphs)} task graphs from {graphs_path}")
        tagged = self.curator.load_tagged_days()
        self.pf = GraphParticleFilter(
            self.task_graphs, tagged, seed=0,
            p_bg_override=self.policy_cfg.get('pf_p_bg', None),
            robot_opens=bool(self.policy_cfg.get('pf_robot_opens', False)),
            single_belief=bool(self.policy_cfg.get('pf_single_belief', False)))
        print(f"[hybrid] PF recognition (v6): particle filter "
              f"over {len(self.task_graphs.graphs)} graphs, "
              f"{self.pf.n} particles (no eval-time LLM), learned lead")

    # ------------------------------------------------------------------
    # Evaluation-time action selection
    # ------------------------------------------------------------------

    def _movable_mask(self) -> torch.Tensor:
        """Boolean mask of slate object rows that may relocate (dynamic edges)."""
        return self.history['dynamic_edges_mask'][0, 0].sum(-1) > 0

    def _observe_human_moves(self, t_now: float) -> list:
        """Diff the last two history frames into human move events.
        Returns the list of NEW events (also appended to _day_events)."""
        if self.history['sequence_length'] < 2:
            self._emitted_last_tick = set()
            return []
        prev = self.history['edges'][0, -2].argmax(dim=-1)
        cur = self.history['edges'][0, -1].argmax(dim=-1)
        movable = self._movable_mask()
        class_names = self.common_data['node_classes']
        new = []
        for i in ((cur != prev) & movable).nonzero(as_tuple=False).squeeze(-1).tolist():
            dest = int(cur[i])
            if (i, dest) in self._emitted_last_tick:
                continue  # our own action from last tick (already in the log)
            e = {
                'actor': 'human',
                't_min': float(t_now),
                'obj_idx': int(i),
                'dest_idx': dest,
                'prev_idx': int(prev[i]),
                'obj_class': class_names[i],
                'dest_class': class_names[dest],
                'prev_class': class_names[int(prev[i])],
            }
            self._day_events.append(e)
            new.append(e)
        self._emitted_last_tick = set()
        return new

    def _mk_add(self, candidates: dict, diag: dict, t_now: float):
        """The timing-gated candidate collector shared by both recognition
        engines. base_t = expected start time of the (current/prepped)
        segment. A candidate is admitted only if its expected moment (base_t
        + the node's mean offset within the activity) is within
        act_window_min of its reference point: 'now' for the in-progress
        activity, the predicted start for a prepped one (so prep_horizon_min
        alone decides WHEN to initiate prep, and act_window_min decides how
        deep into the activity to pre-place). Overdue moments are allowed
        (timing variance); acting hours early is what gets objects undone."""
        pc = self.policy_cfg

        use_lead = True
        dflt_lead = pc['act_window_min']

        def add(graph, preds, source, base_t, activity, discount=1.0):
            for nid, score in preds:
                key = graph.node_key(nid)
                # v6: per-key learned lead replaces the fixed act_window —
                # how far ahead of the expected human time we may act, learned
                # down per (activity,obj,dest) wherever the human reverts.
                lead = (self._lead_time.get((activity,) + key, dflt_lead)
                        if use_lead else dflt_lead)
                expected_t = base_t + graph.mean_offset(nid)
                if self._lead_gate and expected_t > max(t_now, base_t) + lead:
                    diag['too_early'] += 1
                    continue
                s = min(1.0, score) * discount
                if key not in candidates or s > candidates[key][0]:
                    candidates[key] = (s, source, activity, expected_t)
        return add

    @staticmethod
    def _gate_precedence(g, preds, done_mass):
        """Suppress each node in proportion to its same-object graph
        prerequisites NOT being satisfied: factor = Π_B [1 - P(B on path)·
        (1 - P(B done))]. A stow with a pending use is suppressed; a rare
        cross-branch prereq barely gates. Returns (gated preds, gated mass).
        done_mass={} (prep, activity not started) leaves only prereq-free
        opening nodes — prep stages openers, never stows."""
        out, gated = [], 0.0
        for nid, s in preds:
            factor = 1.0
            for bid, p_path in g.prereq_nodes(nid):
                factor *= 1.0 - p_path * (1.0 - done_mass.get(bid, 0.0))
            gated += s * (1.0 - factor)
            out.append((nid, s * factor))
        return out, gated

    def _prep_branch(self, add, diag, segments: list, t_now: float,
                     use_prec: bool = False) -> None:
        """Prep upcoming activities: the transformer predicts which
        activities come next AND, per activity in parallel, when each would
        start. Every activity that is both probable enough and imminent
        enough gets prepped, its candidates weighted by its probability.

        *segments* empty → BOS-only input (start-of-day / idle): predict the
        first activity from morning routine without waiting for human moves.
        use_prec (v5): gate prep candidates by precedence with an empty
        done-set, so prep only pre-stages prereq-free opening actions."""
        pc = self.policy_cfg
        probs, gaps = self.transformer.predict_next(segments, now_min=t_now)
        real = {n: p for n, p in probs.items() if n != '__idle__'}
        top_name = max(real, key=real.get)
        diag['pred_next'] = {'name': top_name,
                             'p': round(real[top_name], 3),
                             'gap_min': gaps.get(top_name)}
        prepped = []
        for name, p in sorted(real.items(), key=lambda kv: -kv[1]):
            if p < pc['next_act_conf']:
                break
            gap = gaps.get(name)
            if gap is None or gap > pc['prep_horizon_min']:
                continue
            g2 = self.task_graphs.graphs.get(name)
            if g2 is None:
                # Transformer predicted an activity with no task graph (vocab
                # edge). Surface it in diag rather than vanishing silently.
                diag.setdefault('prep_no_graph', []).append(name)
                continue
            preds = g2.predict_next({START: 1.0}, lookahead=2)
            if use_prec and self._precedence_gate:
                preds, _ = self._gate_precedence(g2, preds, {})
            add(g2, preds, 'prep', base_t=t_now + gap, activity=name,
                discount=pc['prep_discount'] * p)
            prepped.append({'name': name, 'p': round(p, 3),
                            'gap_min': round(gap)})
        diag['prep_fired'] = bool(prepped)
        diag['prepped'] = prepped

    def _candidate_actions_pf(self, t_now: float) -> tuple:
        """PF-mode candidates: posterior-weighted, no MAP collapse — every
        open activity contributes, scored P(open) x graph successor mass.
        The same threshold/cap/timing gates downstream consume the scores."""
        pc = self.policy_cfg
        candidates, diag = {}, {'too_early': 0}
        add = self._mk_add(candidates, diag, t_now)
        view = self.pf.actuation_view(pace=True)
        post = self.pf.posterior_last_assign()
        top = next(iter(post.items()), (BG_LABEL, 1.0))
        diag['current_activity'] = None if top[0] == BG_LABEL else top[0]
        # MAP ablation: keep only the single most-likely-open activity, weight 1.0.
        if self._pf_map and view:
            map_act = max(view, key=lambda a: view[a][0])
            _p, _pos, _bt, _dm = view[map_act]
            view = {map_act: (1.0, _pos, _bt, _dm)}
        diag['pf_open'] = {a: round(v[0], 3) for a, v in view.items()}
        diag['current_branch'] = ('pf_map' if self._pf_map else 'pf_posterior') \
            if view else 'skipped'
        prec_gated = 0.0
        # current branch: task-graph successors of the PF-recognized in-progress
        # activity. Ablation (current_branch=False = "only prep") skips it so the
        # planner acts solely on transformer-anticipated upcoming activities.
        if self._current:
            for act, (p_open, pos, base_t, done_mass) in view.items():
                g = self.task_graphs.graphs.get(act)
                if g is None:
                    continue
                preds = g.predict_next(pos, lookahead=pc['lookahead_actions'])
                if self._precedence_gate:
                    preds, gm = self._gate_precedence(g, preds, done_mass)
                else:
                    gm = 0.0
                prec_gated += gm * p_open
                add(g, [(nid, s * p_open) for nid, s in preds],
                    'current', base_t=base_t, activity=act)
        diag['prec_gated_mass'] = round(prec_gated, 2)
        if self._prep:
            self._prep_branch(add, diag, self.pf.map_segments(), t_now,
                              use_prec=True)
        else:
            diag['prep_fired'] = False
        return candidates, diag

    def get_actions(self, node_map, *, world_state: dict = None):
        """Every tick: observe human moves (if any), update PF, score candidates."""
        if world_state is None:
            raise ValueError('world_state is required for the hybrid policy')
        if self.transformer is None or self.task_graphs is None:
            raise RuntimeError(
                'hybrid policy asked to act before train()/load_artifacts()')
        self._update_history(world_state)
        self._last_fwd = None
        t_now = world_state['time']
        self._cur_t = t_now
        if self._accept_gate:
            self._resolve_placements(t_now)

        new_events = self._observe_human_moves(t_now)

        for e in new_events:
            rev = self._find_revert_placement(
                e['obj_idx'], e['prev_idx'], e['dest_idx'])
            if rev is not None:
                # Human moved the object straight BACK to where we took it from.
                self.pf.observe_reaction(e['obj_idx'], rev['dest_idx'])
                rev['status'] = 'reverted'
                self._day_events[rev['ev_idx']]['reverted'] = True
                act = rev.get('act')
                if act is not None:
                    self._lead_reverted.add((act, e['obj_idx'], rev['dest_idx']))
            else:
                self.pf.observe(e['obj_idx'], e['dest_idx'], e['t_min'])
                self._consume_continuation(e['obj_idx'], e['prev_idx'])
        if new_events:
            post = self.pf.posterior_last_assign()
            top_act, top_p = next(iter(post.items()), (BG_LABEL, 1.0))
            with open(self._tag_log_path, 'a') as f:
                f.write(json.dumps({
                    'day_in_process': self._day_counter, 't_min': t_now,
                    'n_events': len(self._day_events),
                    'current_activity':
                        None if top_act == BG_LABEL else top_act,
                    'current_p': round(top_p, 3),
                    'posterior_top3': {a: round(p, 3) for a, p in
                                       list(post.items())[:3]},
                    'engine': 'pf',
                }) + '\n')
        candidates, diag = self._candidate_actions_pf(t_now)

        pc = self.policy_cfg
        cur_locs = self.history['edges'][0, -1].argmax(dim=-1)
        slate_to_vh = {v: k for k, v in
                       self.history['vh_id_to_slate_idx'].items()}
        node_id_to_class = self.history['node_id_to_class']
        class_names = self.common_data['node_classes']

        # --- guards first: build the executable shortlist for the ranker ---
        cand_log, shortlist = [], []
        for (obj_idx, dest_idx), (score, source, activity, expected_t) in sorted(
                candidates.items(), key=lambda kv: -kv[1][0]):
            entry = {'obj': class_names[obj_idx], 'dest': class_names[dest_idx],
                     'score': round(score, 3), 'source': source,
                     'activity': activity}
            cand_log.append(entry)
            if self._accept_gate:
                if self._cur_t < self._reject_until.get((obj_idx, dest_idx),
                                                         float('-inf')):
                    entry['outcome'] = 'reject_backoff'
                    continue
                pa = self._p_accept(obj_idx, dest_idx)
                score *= pa
                entry['p_accept'] = round(pa, 3)
                entry['score'] = round(score, 3)
            if score < pc['prob_threshold']:
                entry['outcome'] = 'below_threshold'
                continue
            if int(cur_locs[obj_idx]) == dest_idx:
                entry['outcome'] = 'already_at_dest'
                continue
            key = (obj_idx, dest_idx)
            # No shortlist cap: candidates are in score order and emission is
            # bounded by max_actions_per_tick below. (The old rank_top_k cap
            # existed only to bound what the removed LLM ranker saw.)
            entry['outcome'] = 'shortlisted'
            shortlist.append({'key': key, 'entry': entry,
                              'obj_class': class_names[obj_idx],
                              'dest_class': class_names[dest_idx],
                              'activity': activity, 'score': score,
                              'expected_hhmm':
                                  f"{int(expected_t) // 60 % 24:02d}:"
                                  f"{int(expected_t) % 60:02d}"})

        # All gate-approved candidates execute (the LLM ranker was removed —
        # it was a recall sink; see docs/failed_experiments.md).
        chosen = list(range(len(shortlist)))

        actions, obj_idxs, loc_idxs, prev_loc_idxs = [], [], [], []
        for ci in chosen[:pc['max_actions_per_tick']]:
            c = shortlist[ci]
            obj_idx, dest_idx = c['key']
            obj_vh, dest_vh = slate_to_vh.get(obj_idx), slate_to_vh.get(dest_idx)
            if obj_vh is None or dest_vh is None:
                raise RuntimeError(
                    f'no VH id for slate idx {obj_idx}/{dest_idx}')
            c['entry']['outcome'] = 'emitted'
            actions.append(RobotMoveAction(
                node_id_to_class.get(obj_vh, 'unknown'), obj_vh,
                node_id_to_class.get(dest_vh, 'unknown'), dest_vh))
            obj_idxs.append(obj_idx)
            loc_idxs.append(dest_idx)
            prev_loc_idxs.append(int(cur_locs[obj_idx]))
            self._emitted_last_tick.add(c['key'])
            from_idx = int(cur_locs[obj_idx])
            # Robot re-staging the same object invalidates earlier active placements.
            self._supersede_active_placements(obj_idx)
            self.pf.observe_robot(obj_idx, dest_idx, t_now)
            act = c['activity']
            self._lead_emitted.add((act, obj_idx, dest_idx))
            self._day_events.append({
                'actor': 'robot', 't_min': float(t_now),
                'obj_idx': obj_idx, 'dest_idx': dest_idx,
                'prev_idx': from_idx,
                'obj_class': class_names[obj_idx],
                'dest_class': class_names[dest_idx],
                'prev_class': class_names[from_idx],
            })
            ev_idx = len(self._day_events) - 1
            self._robot_placements.setdefault(obj_idx, []).append({
                'ev_idx': ev_idx,
                'dest_idx': dest_idx,
                'from_idx': from_idx,
                'act': act,
                'status': 'active',
                't_emit': float(t_now),
            })
        if cand_log:
            with open(self._diag_log_path, 'a') as f:
                f.write(json.dumps({
                    'day_in_process': self._day_counter, 't_min': t_now,
                    **diag, 'candidates': cand_log}) + '\n')
        return actions, (obj_idxs, loc_idxs, prev_loc_idxs)
