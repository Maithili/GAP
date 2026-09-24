"""NOPA baseline — Neurally-guided Online Probabilistic Assistance.

Puig, Shu, Tenenbaum & Torralba, "NOPA: Neurally-guided Online Probabilistic
Assistance for Building Socially Intelligent Home Assistants", ICRA 2023.

Three components, mapped onto this benchmark:

1. **Goal Proposal Network** ``q(g | s0, s_t)`` (:mod:`model.nopa_gpn`) — samples
   K goal proposals. A goal here is the multiset of ``(obj, dest)`` predicates
   the human realizes during the day; one simulated day = one NOPA episode.
2. **Inverse-planning particle filter** (paper Alg. 2) — each particle carries a
   goal plus a predicted future trajectory. An observed human relocation that is
   not in a particle's predicted plan rejects that particle; when the set empties
   (or the prediction horizon ``T_prop`` elapses) K fresh goals are resampled.
3. **Uncertainty-aware helping planner** (paper Alg. 3) — scores candidate
   subgoal edges by

       V(s_t, e) = w_r*p(e)*|L_M(e) - L_H(e)|+ - w_c*L_H(e)
                   - w_m*(D(s0, s_hat(e)) - D(s0, s_t))

   and executes the argmax.

Deviations from the paper, all forced by this domain (see DESIGN.md 11a):

* **MCTS -> sampled arrival-clock plan.** The simulator charges no navigation or
  manipulation cost (``environment/world.py``: a teleport is one tick), so search
  over motion plans would be uninformative. A plan here is a *schedule*: each
  remaining predicate draws an arrival time from a per-predicate Gaussian fitted
  on the training days, truncated at the current time. Per-particle sampling is
  the bounded rationality the paper gets from limited MCTS rollouts.
* **``L_H(e) == 1`` for every edge**, so ``-w_c*L_H`` is a constant that can
  never change ``argmax V``. ``w_c`` is an inert hyperparameter here.
* **Predicted-intermediate-state edges collapse into predicted-goal edges** — a
  plan prefix here *is* a subset of the goal. Their only residual effect is the
  ``L_M`` vs ``L_max`` split at the ``T_prop`` horizon.
* **Count bins 9 -> 4** (measured max repeats of one predicate per day is 3).
* **Restore edges are limited to the robot's own placements.** Read literally,
  Alg. 3 admits every initial-state edge, which here would have the robot undo
  the *human's* work all afternoon. Fig. 6b is explicit that the helper puts back
  "the fork **it grabbed**". The literal reading is available behind
  ``policy.restore_all_initial``.
* **Out-of-vocabulary observations are excluded from Alg-2 rejection** — a closed
  predicate vocabulary cannot represent them. Rate logged as ``n_oov``.
* **Plan-order gate on same-object predicates.** NOPA's goals are consistent
  *final states*: in O-WAH each object has exactly one target location, so Alg. 3
  can score every goal edge independently. A household day is a *sequence* — 21
  of 29 moved objects here visit two or more destinations in the same day (e.g.
  chessboard -> table in the morning, chessboard -> bookshelf at night). Scoring
  both simultaneously makes the helper ping-pong the object between them, since
  achieving the later predicate destroys the earlier one. The gate keeps only
  each object's *earliest remaining* predicate, ordered by the particles' own
  sampled plans — so it is read off Gamma_hat_k, not bolted on. Disable with
  ``policy.plan_order_gate: false`` for the literal reading.

No destination masking (``valid_locations`` / ``containment_ban``) is applied:
every vocabulary edge is a relocation a human was observed to perform, so
ancestor cycles are impossible by construction.
"""

from __future__ import annotations

import glob
import json
import os
import random

import torch

from agents.llm_memory_curator import extract_events_from_pt
from model.nopa_gpn import GPN_N_COUNTS
from agents.nopa_utils import (
    TICK_MIN, build_task_samples, fit_task_clock_prior, segment_tasks, build_day_samples, build_predicate_vocab, encode_gpn_input,
    fit_clock_prior, s0_satisfied, sample_truncnormal)
from agents.robot_policy import RobotPolicyBase, RobotMoveAction
from agents.slatepro_utils import (
    homers_script_path_to_processed_pt,
    load_common_data, filter_world_state_to_homer_nodes,
    world_state_to_history_initialization, update_history_edges,
    update_history_activity, update_history_time)
from model.nopa_gpn import load_gpn, train_gpn

# A human move reverting a robot placement within this many minutes is read as a
# reaction to the robot, not as evidence about the human's goal. Matches the
# REACTION_WINDOW_MIN convention used elsewhere in the repo.
REACTION_WINDOW_MIN = 20.0


class RobotPolicyNOPA(RobotPolicyBase):
    """NOPA helper agent as a closed-loop robot policy."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = config['robot']['nopa']
        self.gpn_cfg = self.cfg.get('gpn') or {}
        self.pol = self.cfg['policy']
        self.household = config['benchmark']['household']
        self.common_data = load_common_data(
            self.cfg['common_data_path'].format(household=self.household))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Learned artifacts (populated by train() / load_artifacts()).
        self.gpn = None
        self.predicates = None
        self._pid_map = None
        self._script_path = None
        # Task episodes (default): one GT activity segment = one NOPA episode,
        # matching O-WAH's short single-task episodes. False restores the
        # whole-day-as-one-goal formulation so both granularities stay runnable.
        self._task_episodes = bool(self.cfg.get('task_episodes', True))
        # task_split='prep_cleanup': also cut each GT activity where the human's
        # script relocates an object a second time (prep vs cleanup), in train
        # AND at test, so no episode's goal holds two destinations for one object.
        self._task_split = self.cfg.get('task_split', 'none')
        if self._task_split not in ('none', 'prep_cleanup'):
            raise ValueError(f"nopa.task_split must be none|prep_cleanup, got {self._task_split!r}")
        if self._task_split != 'none' and not self._task_episodes:
            raise ValueError("nopa.task_split requires task_episodes: true")
        self._split = self._task_split == 'prep_cleanup'
        self._tasks = []            # [(aid, t0, t1, i0, i1)] for today
        self._task_i = -1           # index of the task currently open
        self._pid = None
        self.clock = None
        self.n_counts = int(self.gpn_cfg.get('n_counts', GPN_N_COUNTS))

        self.history = None
        self._last_fwd = None
        self.data_quantity = None
        self._diag_path = None
        # Process-local day counter (matches the hybrid's 'day_in_process'):
        # it restarts at 0 when a run resumes from a checkpoint, so it is a
        # diagnostic label only -- never join it to harness day indices.
        self._day_counter = -1

        seed = int(self.pol['seed'])
        self.rng = random.Random(seed)
        self._tgen = torch.Generator().manual_seed(seed)
        self._reset_day_state()

    # ------------------------------------------------------------------
    # Driver contract
    # ------------------------------------------------------------------

    def _reset_day_state(self):
        self._s0_locs = None        # LongTensor[N]: day-start parent of each node
        self._s0_sat = None         # bool[P]: predicates already true at day start
        self._counts = None         # LongTensor[P]: human realizations so far today
        self._particles = []        # [{counts, plan, tau, prefix}]
        self._l = 0                 # main-agent steps since the last resample
        self._emitted_last_tick = set()
        self._placements = []       # {obj, dest, prev, t_emit, status}
        self._oracle_counts = None  # cached GT goal for today (goal_source='oracle')
        self._task_t0 = 0.0         # start clock of the open task (0 in day mode)
        self._n_oov = 0
        self._n_resample = 0
        self._resample_reasons = {}
        self._emit_hist = {}

    def _update_history(self, world_state: dict) -> None:
        world_state = filter_world_state_to_homer_nodes(
            world_state, self.common_data)
        if self.history is None:
            self.history = world_state_to_history_initialization(
                world_state, self.common_data)
        else:
            self.history['edges'] = update_history_edges(
                world_state, self.history['edges'],
                self.history['vh_id_to_slate_idx'], self.common_data)
            self.history['activity_features'] = update_history_activity(
                world_state, self.history['activity_features'],
                self.common_data['activities'])
            self.history['time_features'], self.history['times'] = \
                update_history_time(world_state,
                                    self.history['time_features'],
                                    self.history['times'])
            self.history['sequence_length'] += 1

    def reset(self, script_path: str | None = None):
        """New episode: a fresh day is a fresh goal (paper: one task episode).

        *script_path* is only used by ``goal_source='oracle'`` (a measurement
        instrument, not a NOPA variant: it reads the day's GT script).
        """
        self._script_path = script_path
        self._tasks, self._task_i = [], -1
        if self._task_episodes and script_path:
            pt = homers_script_path_to_processed_pt(script_path)
            if os.path.isfile(pt):
                self._tasks = segment_tasks(pt, self._split)
        if self._day_counter >= 0 and self._diag_path:
            self._log_day_summary()
        self.history = None
        self._last_fwd = None
        self._day_counter += 1
        self._reset_day_state()

    # ------------------------------------------------------------------
    # Training  (paper: the GPN is fit on episodes where only the main agent acts)
    # ------------------------------------------------------------------

    def train(self, dataset_path: str, output_dir: str, days: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')
        if os.path.isfile(weights_path):
            print(f'[NOPA] weights for day {days} already exist; loading.')
            self.load_artifacts(weights_path)
            self.data_quantity = {'train': -1, 'val': -1}
            return

        solo_days = self.cfg['solo_days']
        if solo_days is None:
            raise RuntimeError(
                'robot.nopa.solo_days is unset. NOPA\'s goal proposal network '
                'must be trained on main-agent-alone episodes only (paper 3-C); '
                'set it to --warmup_days.')
        solo_days = int(solo_days)

        train_dir = os.path.join(dataset_path, 'train')
        all_files = sorted(glob.glob(os.path.join(train_dir, '*.pt')))
        if len(all_files) < solo_days:
            raise RuntimeError(
                f'[NOPA] need {solo_days} main-agent-alone day files in '
                f'{train_dir}, found {len(all_files)}')
        # Train on every observed day written so far, the same protocol as
        # SLaTe-PRO (RoutinesDataset over the whole dataset_path) and STREAK.
        # Truncating to the first solo_days made every retrain round re-fit on
        # days 0-4 -- a silent no-op that froze the goal model in the retrain
        # setting while the other baselines kept adapting.
        files = all_files
        print(f'[NOPA] GPN training days ({len(files)} observed): '
              f'{[os.path.basename(f) for f in files]}')

        n_prev = len(self.predicates) if self.predicates else 0
        predicates, pid_map = build_predicate_vocab(files)
        if self._task_episodes:
            clock = fit_task_clock_prior(files, pid_map, float(self.pol['sigma_floor_min']),
                                         split=self._split)
        else:
            clock = fit_clock_prior(files, pid_map, float(self.pol['sigma_floor_min']))
        print(f"[NOPA] episodes: {'TASK (GT activity segments)' if self._task_episodes else 'DAY'}"
              f"{' split prep/cleanup' if self._split else ''}; "
              f"clock prior is {'task-relative' if self._task_episodes else 'absolute'}")
        missing = set(range(len(predicates))) - set(clock)
        if missing:
            raise RuntimeError(f'[NOPA] clock prior missing for pids {sorted(missing)}')
        print(f'[NOPA] predicate vocabulary |P| {n_prev} -> {len(predicates)}')

        xs, ys, day_of = [], [], []
        for di, path in enumerate(files):
            if self._task_episodes:
                x, y, n = build_task_samples(path, predicates, pid_map, self.n_counts,
                                             split=self._split)
            else:
                x, y, n = build_day_samples(path, predicates, pid_map, self.n_counts)
            xs.append(x)
            ys.append(y)
            day_of.extend([di] * n)
        x_all, y_all = torch.cat(xs), torch.cat(ys)

        best = train_gpn(
            x_all, y_all, day_of, len(files), self.gpn_cfg, weights_path,
            extra={'predicates': [list(p) for p in predicates],
                   'clock': clock, 'solo_days': solo_days,
                   'household': self.household,
                   'n_train_days': len(files)})
        self.data_quantity = {'train': int(x_all.shape[0]), 'val': -1,
                              'goal_f1': best.get('val_goal_f1')}
        self.load_artifacts(weights_path)

    def load_artifacts(self, weights_path: str) -> None:
        self.gpn, ckpt = load_gpn(weights_path, self.device)
        self.predicates = [tuple(p) for p in ckpt['predicates']]
        self._pid = {p: i for i, p in enumerate(self.predicates)}
        self.clock = ckpt['clock']
        self.n_counts = int(ckpt['n_counts'])
        self._diag_path = os.path.join(os.path.dirname(weights_path),
                                       'nopa_diag.jsonl')
        print(f'[NOPA] loaded {weights_path}: |P|={len(self.predicates)} '
              f'C={self.n_counts} val={ckpt["val_metrics"]}')

    # ------------------------------------------------------------------
    # Observation  (Alg. 2 lines 4-10: reject particles inconsistent with a_M)
    # ------------------------------------------------------------------

    def _movable_mask(self) -> torch.Tensor:
        return self.history['dynamic_edges_mask'][0, 0].sum(-1) > 0

    def _observe_human_moves(self, t_now: float) -> list:
        """Diff the last two history frames into human relocation events."""
        if self.history['sequence_length'] < 2:
            self._emitted_last_tick = set()
            return []
        prev = self.history['edges'][0, -2].argmax(dim=-1)
        cur = self.history['edges'][0, -1].argmax(dim=-1)
        movable = self._movable_mask()
        new = []
        for i in ((cur != prev) & movable).nonzero(
                as_tuple=False).squeeze(-1).tolist():
            dest = int(cur[i])
            if (i, dest) in self._emitted_last_tick:
                continue          # our own placement from last tick
            new.append({'t_min': float(t_now), 'obj_idx': int(i),
                        'dest_idx': dest, 'prev_idx': int(prev[i])})
        self._emitted_last_tick = set()
        return new

    def _is_undo(self, event: dict, t_now: float) -> bool:
        """True if this move reverts a recent robot placement.

        A robot-induced revert says nothing about the human's goal; feeding it to
        Alg. 2 would reject every particle. An LLM-human *redirect* moves the
        object somewhere new (``dest != prev``) and correctly is not an undo.
        """
        for p in self._placements:
            if (p['obj'] == event['obj_idx']
                    and p['status'] == 'active'
                    and t_now - p['t_emit'] <= REACTION_WINDOW_MIN
                    and event['dest_idx'] == p['prev']):
                p['status'] = 'reverted'
                return True
        return False

    # ------------------------------------------------------------------
    # Proposal  (Alg. 2 lines 11-17)
    # ------------------------------------------------------------------

    def _current_task_index(self, t_now: float) -> int:
        """Index of the GT task segment covering *t_now*, or -1 (day mode)."""
        if not self._task_episodes or not self._tasks:
            return -1
        for i, (_aid, t0, t1, _i0, _i1) in enumerate(self._tasks):
            if t0 <= t_now <= t1:
                return i
        return self._task_i          # between segments: keep the open episode

    def _episode_clock(self, t_now: float) -> float:
        """Time fed to the GPN / clock priors: relative to task start in task
        mode, absolute wall clock in day mode."""
        return t_now - self._task_t0 if self._task_episodes else t_now

    def _oracle_goal_counts(self) -> torch.Tensor:
        """True per-predicate counts for today, read from the GT ``.pt``.

        Measurement instrument (cf. the P0 oracle), not a NOPA variant: it
        removes goal-IDENTITY uncertainty only. Timing still comes from the
        fitted clock priors, so this isolates how much of NOPA's deficit is
        goal inference versus the helping planner.
        """
        if self._oracle_counts is not None:
            return self._oracle_counts
        if not self._script_path:
            raise RuntimeError(
                "nopa goal_source='oracle' needs reset(script_path=...); "
                "the harness passes it for policies whose reset accepts it")
        pt = homers_script_path_to_processed_pt(self._script_path)
        if not os.path.isfile(pt):
            raise RuntimeError(f'nopa oracle: no GT file {pt} for {self._script_path}')
        if self._pid_map is None:
            self._pid_map = {tuple(p): i for i, p in enumerate(self.predicates)}
        counts = torch.zeros(len(self.predicates), dtype=torch.long)
        for e in extract_events_from_pt(pt)['events']:
            pid = self._pid_map.get((int(e['obj_idx']), int(e['dest_idx'])))
            if pid is None:
                continue          # out-of-vocab, same treatment as the observation path
            counts[pid] = min(int(counts[pid]) + 1, self.n_counts - 1)
        self._oracle_counts = counts
        return counts

    def _goal_probs(self, t_now: float) -> torch.Tensor:
        """``(P, C)`` count distribution from the GPN (or uniform for OursRG)."""
        if self.pol['goal_source'] == 'oracle':
            probs = torch.zeros(len(self.predicates), self.n_counts)
            probs[torch.arange(len(self.predicates)), self._oracle_goal_counts()] = 1.0
            return probs
        if self.pol['goal_source'] == 'uniform':
            return torch.full((len(self.predicates), self.n_counts),
                              1.0 / self.n_counts)
        x = encode_gpn_input(self._counts, self._s0_sat,
                             self._episode_clock(t_now),
                             self.n_counts).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.gpn(x)[0]
        return torch.softmax(logits, dim=-1).cpu()

    def _make_plan(self, counts: torch.Tensor, t_now: float) -> dict:
        """Sampled arrival-clock trajectory for one goal (the MCTS stand-in)."""
        remaining = (counts > self._counts).nonzero(
            as_tuple=False).squeeze(-1).tolist()
        tau = {}
        for pid in remaining:
            mu, sigma, _ = self.clock[pid]
            tau[pid] = sample_truncnormal(mu, sigma,
                                          self._episode_clock(t_now), self.rng)
        plan = sorted(tau.items(), key=lambda kv: kv[1])
        t_prop = int(self.pol['T_prop'])
        return {'counts': counts, 'plan': plan, 'tau': tau,
                'rank': {pid: i for i, (pid, _) in enumerate(plan)},
                'prefix': {pid for pid, _ in plan[:t_prop]}}

    def _resample(self, t_now: float, reason: str) -> None:
        probs = self._goal_probs(t_now)
        if self.pol['argmax_goal']:
            samples = probs.argmax(dim=-1).unsqueeze(0)            # HPGPN: K=1
        else:
            k = int(self.pol['K'])
            samples = torch.multinomial(
                probs, k, replacement=True, generator=self._tgen).T
        # A goal may not contradict what has already been observed today.
        samples = torch.maximum(samples, self._counts.unsqueeze(0))
        self._particles = [self._make_plan(samples[i].clone(), t_now)
                           for i in range(samples.shape[0])]
        self._l = 0
        self._n_resample += 1
        self._resample_reasons[reason] = self._resample_reasons.get(reason, 0) + 1

    def _consume(self, pid: int) -> None:
        """Drop a realized predicate from every surviving particle's plan."""
        t_prop = int(self.pol['T_prop'])
        for k in self._particles:
            if pid in k['tau']:
                del k['tau'][pid]
                k['plan'] = [s for s in k['plan'] if s[0] != pid]
                k['rank'] = {p: i for i, (p, _) in enumerate(k['plan'])}
                k['prefix'] = {p for p, _ in k['plan'][:t_prop]}

    # ------------------------------------------------------------------
    # Helping planner  (Alg. 3)
    # ------------------------------------------------------------------

    def _candidate_edges(self, cur: torch.Tensor, t_now: float,
                         movable: torch.Tensor) -> list:
        """Candidate subgoal edges with their ``p(e)`` and ``L_M(e)``."""
        n_part = len(self._particles)
        l_max = float(self.pol['L_max'])
        big = float('inf')
        unit = self.pol['lm_units']
        # Horizon fallback (paper): a predicate past the plan's first T_prop
        # entries gets L_M = L_max instead of its real position, because the
        # paper's plan is an MCTS rollout only simulated that deep. Our plan is
        # predicates sorted by sampled arrival time, so `rank` holds the true
        # position for every predicate and the fallback discards information we
        # have -- measured driving 67% of actions, with L_max=100 dwarfing real
        # positions of 1-15.
        use_fallback = bool(self.pol.get('lm_horizon_fallback', True))
        if unit not in ('ticks', 'steps'):
            raise ValueError(f"robot.nopa.policy.lm_units must be ticks|steps, "
                             f"got {unit!r}")
        cands, wanted_objs = {}, set()

        for k in self._particles:
            for pid in k['tau']:
                obj, dest = self.predicates[pid]
                wanted_objs.add(obj)
                c = cands.setdefault(
                    (obj, dest), {'kind': 'goal', 'pid': pid, 'n': 0,
                                  'L_M': big, 't_first': big})
                c['n'] += 1
                c['t_first'] = min(c['t_first'], k['tau'][pid])
                if use_fallback and pid not in k['prefix']:
                    steps = l_max          # in the goal, beyond the horizon
                elif unit == 'ticks':
                    steps = (k['tau'][pid] - t_now) / TICK_MIN
                else:
                    steps = float(k['rank'][pid] + 1)   # literal-paper plan index
                c['L_M'] = min(c['L_M'], steps)
        for c in cands.values():
            c['p_e'] = c['n'] / max(n_part, 1)

        if self.pol['plan_order_gate']:
            # An object's later relocations are only reachable after its earlier
            # ones; helping with a later one first destroys the earlier. Keep
            # each object's earliest remaining predicate, per the sampled plans.
            earliest = {}
            for key, c in cands.items():
                cur_best = earliest.get(key[0])
                if cur_best is None or c['t_first'] < cands[cur_best]['t_first']:
                    earliest[key[0]] = key
            cands = {k_: cands[k_] for k_ in earliest.values()}

        if not self.pol['restore']:
            return list(cands.items())

        # Restore edges (Alg. 3 lines 8-10): p(e)=1, L_M(e)=0.
        if self.pol['restore_all_initial']:
            objs = [i for i in range(cur.shape[0])
                    if bool(movable[i]) and int(cur[i]) != int(self._s0_locs[i])]
        else:
            objs = [p['obj'] for p in self._placements if p['status'] == 'active'
                    and int(cur[p['obj']]) != int(self._s0_locs[p['obj']])
                    and p['obj'] not in wanted_objs]
        for obj in dict.fromkeys(objs):
            key = (obj, int(self._s0_locs[obj]))
            if key in cands:
                continue           # the goal already wants it back there
            cands[key] = {'kind': 'restore', 'pid': None, 'n': n_part,
                          'p_e': 1.0, 'L_M': 0.0, 't_first': 0.0}
        return list(cands.items())

    def _value(self, obj: int, dest: int, info: dict, cur: torch.Tensor) -> float:
        """Eq. (1). ``L_H`` is 1 tick for every edge: the robot teleports."""
        l_h = 1.0
        s0 = int(self._s0_locs[obj])
        if int(cur[obj]) == s0:
            d_delta = 1.0          # moving an object off its initial location
        elif dest == s0:
            d_delta = -1.0         # putting it back
        else:
            d_delta = 0.0
        benefit = float(self.pol['w_r']) * info['p_e'] * max(0.0, info['L_M'] - l_h)
        w_m = 0.0 if not self.pol['restore'] else float(self.pol['w_m'])
        return benefit - float(self.pol['w_c']) * l_h - w_m * d_delta

    # ------------------------------------------------------------------
    # Eval  (Alg. 1)
    # ------------------------------------------------------------------

    def get_actions(self, node_map, *, world_state: dict = None):
        if world_state is None:
            raise ValueError('world_state is required for the NOPA policy')
        self._update_history(world_state)
        self._last_fwd = None
        if self.gpn is None:
            raise RuntimeError(
                '[NOPA] get_actions called before the goal proposal network was '
                'trained or loaded — train() must run at the warm-start boundary')

        t_now = float(world_state['time'])
        cur = self.history['edges'][0, -1].argmax(dim=-1)
        # Episode boundary. In task mode a new GT activity segment starts a
        # fresh episode: s0 is the CURRENT (possibly robot-perturbed) state,
        # counts reset, particles resample. Boundaries only — the activity id
        # is never shown to the policy, since inferring the goal is its job.
        ti = self._current_task_index(t_now)
        new_episode = self._s0_locs is None or (
            self._task_episodes and ti != self._task_i and ti >= 0)
        if new_episode:
            reason = 'day_start' if self._s0_locs is None else 'task_start'
            if self._task_episodes and ti >= 0:
                self._task_i = ti
                self._task_t0 = self._tasks[ti][1]
            self._s0_locs = cur.clone()
            self._s0_sat = s0_satisfied(self._s0_locs, self.predicates)
            self._counts = torch.zeros(len(self.predicates), dtype=torch.long)
            self._particles = []
            self._l = 0
            self._resample(t_now, reason)

        # --- Alg. 2: reject particles inconsistent with the observed action ---
        for e in self._observe_human_moves(t_now):
            if self._is_undo(e, t_now):
                continue
            pid = self._pid.get((e['obj_idx'], e['dest_idx']))
            if pid is None:
                self._n_oov += 1
                continue
            self._counts[pid] = min(int(self._counts[pid]) + 1,
                                    self.n_counts - 1)
            self._l += 1
            if self.pol['inverse_planning']:
                self._particles = [k for k in self._particles
                                   if pid in k['prefix']]
            self._consume(pid)

        if not self._particles:
            self._resample(t_now, 'empty')
        elif self._l >= int(self.pol['T_prop']):
            self._resample(t_now, 'horizon')

        # --- Alg. 3: score candidate subgoals, execute the argmax ---
        movable = self._movable_mask()
        rule = self.pol.get('action_rule', 'value')
        if rule not in ('value', 'most_likely'):
            raise ValueError(f"nopa action_rule must be value|most_likely, got {rule!r}")
        scored = []
        for (obj, dest), info in self._candidate_edges(cur, t_now, movable):
            if int(cur[obj]) == dest or not bool(movable[obj]):
                continue
            if rule == 'most_likely':
                # Rank purely by posterior support p(e): the fraction of
                # surviving particles whose goal contains this edge. Drops Eq.(1)
                # entirely -- no earliness term, no w_c cost, no w_m disturbance
                # penalty, no idle threshold. Motivated by the measurement that
                # 52% of value-ranked actions were driven by the L_max horizon
                # placeholder rather than by any earliness estimate.
                scored.append((info['p_e'], obj, dest, info))
            else:
                scored.append((self._value(obj, dest, info, cur), obj, dest, info))
        scored.sort(key=lambda s: -s[0])

        if rule == 'most_likely':
            # act on the single most probable edge; no value gate
            chosen = scored[:int(self.pol['max_actions'])]
        else:
            min_value = float(self.pol['min_value'])
            chosen = [s for s in scored[:int(self.pol['max_actions'])]
                      if s[0] > min_value]
        return self._emit(chosen, cur, t_now, len(scored))

    def _emit(self, chosen: list, cur: torch.Tensor, t_now: float,
              n_cands: int):
        slate_to_vh = {v: k for k, v in
                       self.history['vh_id_to_slate_idx'].items()}
        node_id_to_class = self.history['node_id_to_class']
        actions, obj_idxs, loc_idxs, prev_loc_idxs, log = [], [], [], [], []
        for value, obj, dest, info in chosen:
            obj_vh, dest_vh = slate_to_vh.get(obj), slate_to_vh.get(dest)
            if obj_vh is None or dest_vh is None:
                raise RuntimeError(f'[NOPA] no VH id for slate idx {obj}/{dest}')
            prev = int(cur[obj])
            actions.append(RobotMoveAction(
                node_id_to_class.get(obj_vh, 'unknown'), obj_vh,
                node_id_to_class.get(dest_vh, 'unknown'), dest_vh))
            obj_idxs.append(obj)
            loc_idxs.append(dest)
            prev_loc_idxs.append(prev)
            self._emitted_last_tick.add((obj, dest))
            for p in self._placements:
                if p['obj'] == obj and p['status'] == 'active':
                    p['status'] = 'superseded'
            self._placements.append({'obj': obj, 'dest': dest, 'prev': prev,
                                     't_emit': t_now, 'status': 'active'})
            self._emit_hist[(obj, dest)] = self._emit_hist.get((obj, dest), 0) + 1
            log.append({'obj': self.common_data['node_classes'][obj],
                        'dest': self.common_data['node_classes'][dest],
                        'kind': info['kind'], 'V': round(value, 3),
                        'p_e': round(info['p_e'], 3),
                        'L_M': round(info['L_M'], 2)})
        if self._diag_path and log:
            with open(self._diag_path, 'a') as f:
                f.write(json.dumps({
                    'day_in_run': self._day_counter, 't_min': t_now,
                    'n_particles': len(self._particles), 'n_candidates': n_cands,
                    'emitted': log}) + '\n')
        return actions, (obj_idxs, loc_idxs, prev_loc_idxs)

    def _log_day_summary(self) -> None:
        top = max(self._emit_hist.values()) if self._emit_hist else 0
        with open(self._diag_path, 'a') as f:
            f.write(json.dumps({
                'day_in_run': self._day_counter, 'summary': True,
                'n_oov': self._n_oov, 'n_resample': self._n_resample,
                'resample_reasons': self._resample_reasons,
                'n_robot_actions': sum(self._emit_hist.values()),
                'max_repeats_one_edge': top}) + '\n')
