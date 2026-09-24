"""Data helpers for the NOPA baseline (Puig et al., ICRA 2023).

Parallel to :mod:`agents.slatepro_utils`, but for NOPA's *predicate* space
rather than SLaTe's tensor space.

NOPA represents a goal as a set of predicates with counts. In O-WAH a predicate
is e.g. ``ON(plate, kitchentable)`` with a count (how many plates). Here object
instances are unique, so the analogous predicate is the pair
``(obj_idx, dest_idx)`` in slate index space, and its "count" is how many times
the human realizes that relocation during the day (measured max = 3, so the
count head uses 4 bins ``{0, 1, 2, 3+}`` in place of the paper's 9).

Everything here reads the same processed ``.pt`` day files the rest of the repo
uses, via :func:`agents.llm_memory_curator.extract_events_from_pt` (a pure
tensor reader — the LLM curator itself is deliberately NOT used: NOPA must be an
independent baseline, not a variant of the PF-hybrid).
"""

from __future__ import annotations

import math
import os
import re

import torch

from agents.llm_memory_curator import extract_events_from_pt

# Day grid: HOMER+ days run 06:00 -> 24:00 in 10-min ticks.
DAY_START_MIN = 360.0
DAY_END_MIN = 1440.0
TICK_MIN = 10.0
N_TIME_FEATURES = 4


# ---------------------------------------------------------------------------
# Predicate vocabulary
# ---------------------------------------------------------------------------

def build_predicate_vocab(pt_files: list) -> tuple:
    """Closed predicate vocabulary over the given day files.

    Returns ``(predicates, pid_map)`` where *predicates* is a sorted list of
    ``(obj_idx, dest_idx)`` tuples and *pid_map* maps each pair to its index.

    Sorted (not insertion-ordered) so the vocabulary — and therefore the GPN's
    input/output layout — is reproducible across runs and resumes.
    """
    if not pt_files:
        raise ValueError('build_predicate_vocab: no .pt files given')
    seen = set()
    for path in pt_files:
        for ev in extract_events_from_pt(path)['events']:
            seen.add((int(ev['obj_idx']), int(ev['dest_idx'])))
    predicates = sorted(seen)
    return predicates, {p: i for i, p in enumerate(predicates)}


def day_count_vector(events: list, pid_map: dict, n_counts: int) -> torch.Tensor:
    """Per-predicate realization count for one day, clipped to ``n_counts - 1``."""
    counts = torch.zeros(len(pid_map), dtype=torch.long)
    for ev in events:
        pid = pid_map.get((int(ev['obj_idx']), int(ev['dest_idx'])))
        if pid is None:
            continue          # out-of-vocabulary; caller tracks the rate
        counts[pid] = min(int(counts[pid]) + 1, n_counts - 1)
    return counts


# ---------------------------------------------------------------------------
# GPN input encoding  (paper Fig. 2b: Delta s_t, one row per predicate)
# ---------------------------------------------------------------------------

def time_features(t_min: float) -> torch.Tensor:
    """Clock encoding appended to Delta s_t.

    The paper's O-WAH episodes are short and clock-free; a household routine is
    not, so the current time is part of ``s_t`` here. Documented as a deviation.
    """
    frac = float(t_min) / DAY_END_MIN
    span = (float(t_min) - DAY_START_MIN) / (DAY_END_MIN - DAY_START_MIN)
    return torch.tensor([math.sin(2 * math.pi * frac),
                         math.cos(2 * math.pi * frac),
                         frac, span], dtype=torch.float32)


def encode_gpn_input(counts_today: torch.Tensor, s0_sat: torch.Tensor,
                     t_min: float, n_counts: int) -> torch.Tensor:
    """Flatten ``Delta s_t`` into the GPN input vector.

    Per predicate: ``one_hot(count_so_far, n_counts)`` concatenated with a
    binary flag for "already satisfied in ``s0``". Plus 4 global clock dims.

    Shape: ``(P * (n_counts + 1) + 4,)``.
    """
    counts_today = counts_today.long().clamp_(0, n_counts - 1)
    onehot = torch.nn.functional.one_hot(counts_today, n_counts).float()
    per_pred = torch.cat([onehot, s0_sat.float().unsqueeze(-1)], dim=-1)
    return torch.cat([per_pred.reshape(-1), time_features(t_min)])


def s0_satisfied(s0_locs: torch.Tensor, predicates: list) -> torch.Tensor:
    """Binary flag per predicate: does the day-start state already satisfy it?"""
    obj = torch.tensor([p[0] for p in predicates], dtype=torch.long)
    dest = torch.tensor([p[1] for p in predicates], dtype=torch.long)
    return (s0_locs[obj] == dest)


# ---------------------------------------------------------------------------
# Arrival-clock prior  (this domain's stand-in for NOPA's MCTS rollout)
# ---------------------------------------------------------------------------

def fit_clock_prior(pt_files: list, pid_map: dict,
                    sigma_floor_min: float = 20.0) -> dict:
    """Per-predicate Gaussian over arrival time, fitted on main-agent-alone days.

    Returns ``{pid: (mu, sigma, n)}``. Predicates seen once get the floor sigma;
    the floor mirrors the ``STD_FLOOR_MIN`` convention in
    ``activity_particle_filter.derive_priors``.

    This replaces NOPA's MCTS rollout: the simulator charges no navigation or
    manipulation cost, so a plan here is a *schedule*, not a cost-to-go.
    """
    samples = {}
    for path in pt_files:
        for ev in extract_events_from_pt(path)['events']:
            pid = pid_map.get((int(ev['obj_idx']), int(ev['dest_idx'])))
            if pid is None:
                continue
            samples.setdefault(pid, []).append(float(ev['t_min']))
    prior = {}
    for pid, ts in samples.items():
        n = len(ts)
        mu = sum(ts) / n
        if n > 1:
            var = sum((t - mu) ** 2 for t in ts) / (n - 1)
            sigma = max(math.sqrt(var), sigma_floor_min)
        else:
            sigma = sigma_floor_min
        prior[pid] = (mu, sigma, n)
    return prior


# ---------------------------------------------------------------------------
# Task episodes  (one GT activity segment = one NOPA episode)
# ---------------------------------------------------------------------------

def _activity_runs(act: list) -> list:
    """Contiguous runs of the same activity id: ``[(aid, i0, i1)]``."""
    runs, s = [], 0
    for i in range(1, len(act) + 1):
        if i == len(act) or act[i] != act[s]:
            runs.append((int(act[s]), s, i))
            s = i
    return runs


def _gt_counterpart(pt_path: str) -> tuple:
    """GT script file for a day file, and the tick offset from GT to it.

    A GT file (``HOMER_PLUS``) is its own counterpart, offset 0. An observed
    file written by the closed loop (``external/gen_<tag>/...``) has the same
    household/split/day path under ``HOMER_PLUS``, and is stored ONE TICK LATE:
    its states and activity labels at index ``i`` are GT's at ``i - 1``
    (measured on all 75 days of a run: 107/108 labels match at that shift, the
    unmatched one being index 0). That offset is checked exactly here, per
    file, and anything else raises rather than mis-placing a cut.
    """
    if re.search(r'(^|/)HOMER_PLUS/', pt_path):
        return pt_path, 0
    gt = re.sub(r'(^|/)gen_[^/]+/', r'\1HOMER_PLUS/', pt_path, count=1)
    if gt == pt_path or not os.path.isfile(gt):
        raise RuntimeError(f'[NOPA task_split] no GT counterpart for {pt_path} (tried {gt})')
    a = torch.load(pt_path, map_location='cpu', weights_only=False)['activity'].tolist()
    g = torch.load(gt, map_location='cpu', weights_only=False)['activity'].tolist()
    # The ONE tolerated label difference: a sub-tick activity (come_home,
    # leave_home, taking_medication, ...) inside an idle gap. The closed loop
    # labels that single tick with the short activity; the GT pipeline labels it
    # idle (activity id 1 == 'idle' in all three households' common_data). The
    # time grid is unaffected. Measured on complete 75-day runs: A 0, B 12, C 32
    # such ticks, and no mismatch of any other kind.
    IDLE = 1
    bad = []
    if len(a) == len(g) - 1:
        for i in range(1, len(a)):
            if a[i] == g[i - 1]:
                continue
            single = a[i - 1] != a[i] and (i + 1 >= len(a) or a[i + 1] != a[i])
            if not (g[i - 1] == IDLE and single):
                bad.append(i)
    if len(a) != len(g) - 1 or bad:
        raise RuntimeError(f'[NOPA task_split] {pt_path} is not GT {gt} lagged by one tick '
                           f'(len {len(a)} vs {len(g)}, mismatched ticks {bad[:10]}); '
                           f'refusing to map split points')
    return gt, 1


def prep_cleanup_cut_ticks(gt_path: str) -> set:
    """Ticks (GT indices) where a GT activity splits into prep / cleanup parts.

    Only the GT script's relocations are used -- the human's plan, never robot
    moves. Within each activity, a new sub-task starts at the first tick that
    relocates an object already relocated earlier in the current sub-task
    (e.g. bowl cupboard->table ... bowl table->sink). Repeats inside a single
    10-min tick are invisible: the dataset stores one net move per tick.
    """
    d = torch.load(gt_path, map_location='cpu', weights_only=False)
    tick_of = {float(t): i for i, t in enumerate(d['times'].tolist())}
    objs_at = {}
    for e in extract_events_from_pt(gt_path)['events']:
        ti = tick_of.get(float(e['t_min']))
        if ti is None:
            raise RuntimeError(f'[NOPA task_split] event at t={e["t_min"]} has no tick in {gt_path}')
        objs_at.setdefault(ti, set()).add(int(e['obj_idx']))
    cuts = set()
    for _aid, i0, i1 in _activity_runs(d['activity'].tolist()):
        moved = set()
        for ti in range(i0, i1):
            objs = objs_at.get(ti, set())
            if objs & moved:
                cuts.add(ti)
                moved = set()
            moved |= objs
    return cuts


def segment_tasks(pt_path: str, split: bool = False) -> list:
    """GT activity segments for one day: ``[(activity_id, t0, t1, i0, i1)]``.

    Boundaries come from the dataset's own per-tick ``activity`` tensor — the
    ground-truth key the PF-hybrid deliberately never reads — so using it here
    gives the baseline a signal our method forgoes, not the other way round.
    Contiguous runs of the same id form one task. Measured: ~27 tasks/day,
    median 30 min, mean 3.7 distinct predicates (vs 48-58 for a whole day),
    which is the regime NOPA's O-WAH episodes were designed for.

    ``split=True`` further cuts each activity at :func:`prep_cleanup_cut_ticks`
    (computed on the GT script, shifted onto this file's tick grid), so no
    object is relocated twice within one episode of the human's plan.
    """
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    act, times = d['activity'].tolist(), d['times'].tolist()
    cuts = set()
    if split:
        gt, shift = _gt_counterpart(pt_path)
        cuts = {c + shift for c in prep_cleanup_cut_ticks(gt)}
    segs = []
    for aid, i0, i1 in _activity_runs(act):
        starts = [i0] + sorted(c for c in cuts if i0 < c < i1)
        for s, e in zip(starts, starts[1:] + [i1]):
            segs.append((aid, float(times[s]), float(times[e - 1]), s, e))
    return segs


def build_task_samples(pt_path: str, predicates: list, pid_map: dict,
                       n_counts: int, split: bool = False) -> tuple:
    """Encode one day into GPN rows with ONE EPISODE PER GT TASK.

    Per task: the label is that task's goal count vector, ``s0`` is the world
    state immediately BEFORE the task's first tick, and the clock feature is
    time since task start. Tasks with no relocations are kept on purpose -- the
    all-zero label teaches the GPN that some tasks need no help (~15%).

    Events are attributed by TICK INDEX (``i0 <= tick < i1``), not by comparing
    times. Segments can be a single tick (``t0 == t1``), for which any open
    time-window is empty; and an event at the task's first tick belongs to that
    task, which is why ``s0`` is taken one tick earlier.
    """
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    times = d['times'].tolist()
    locs = d['edges'].argmax(dim=-1)                       # (T, N)
    tick_of = {float(t): i for i, t in enumerate(times)}
    events = sorted(extract_events_from_pt(pt_path)['events'],
                    key=lambda e: e['t_min'])
    rows, labels = [], []
    for _aid, t0, _t1, i0, i1 in segment_tasks(pt_path, split):
        s0_sat = s0_satisfied(locs[max(0, i0 - 1)], predicates)
        seg_ev = [e for e in events
                  if i0 <= tick_of.get(float(e['t_min']), -1) < i1]
        y = day_count_vector(seg_ev, pid_map, n_counts)
        counts = torch.zeros(len(pid_map), dtype=torch.long)
        ei = 0
        for ti in range(i0, i1):
            t = times[ti]
            while ei < len(seg_ev) and seg_ev[ei]['t_min'] <= t:
                pid = pid_map.get((int(seg_ev[ei]['obj_idx']),
                                   int(seg_ev[ei]['dest_idx'])))
                if pid is not None:
                    counts[pid] = min(int(counts[pid]) + 1, n_counts - 1)
                ei += 1
            rows.append(encode_gpn_input(counts, s0_sat, float(t) - t0, n_counts))
            labels.append(y)
    x = torch.stack(rows)
    return x, torch.stack(labels), x.shape[0]


def fit_task_clock_prior(pt_files: list, pid_map: dict,
                         sigma_floor_min: float = 20.0, split: bool = False) -> dict:
    """Per-predicate Gaussian over time SINCE TASK START (task-episode mode).

    Same shape as :func:`fit_clock_prior` (``{pid: (mu, sigma, n)}``) but the
    samples are within-task offsets, so the prior no longer smuggles the wall
    clock into a formulation whose episodes are meant to be clock-free.

    Attribution is by tick index, matching :func:`build_task_samples`: a time
    window would silently drop every event in a single-tick segment and every
    event on a segment's first tick, leaving vocabulary predicates with no
    prior at all.
    """
    samples = {}
    for path in pt_files:
        d = torch.load(path, map_location='cpu', weights_only=False)
        times = d['times'].tolist()
        tick_of = {float(t): i for i, t in enumerate(times)}
        segs = segment_tasks(path, split)
        for ev in extract_events_from_pt(path)['events']:
            pid = pid_map.get((int(ev['obj_idx']), int(ev['dest_idx'])))
            if pid is None:
                continue
            ti = tick_of.get(float(ev['t_min']), -1)
            for _aid, t0, _t1, i0, i1 in segs:
                if i0 <= ti < i1:
                    samples.setdefault(pid, []).append(float(ev['t_min']) - t0)
                    break
    prior = {}
    for pid, ts in samples.items():
        n = len(ts)
        mu = sum(ts) / n
        if n > 1:
            var = sum((t - mu) ** 2 for t in ts) / (n - 1)
            sigma = max(math.sqrt(var), sigma_floor_min)
        else:
            sigma = sigma_floor_min
        prior[pid] = (mu, sigma, n)
    return prior


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f'_norm_ppf: p must be in (0,1), got {p}')
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sample_truncnormal(mu: float, sigma: float, lower: float, rng) -> float:
    """Draw from ``N(mu, sigma)`` truncated to ``[lower, inf)`` by inverse CDF.

    The truncation at the current time is the state-conditioning MCTS would have
    supplied: a predicate not yet realized cannot have been scheduled in the
    past. Per-particle sampling *is* NOPA's bounded rationality — two particles
    holding the same goal get different predicted trajectories.
    """
    if sigma <= 0:
        raise ValueError(f'sample_truncnormal: sigma must be > 0, got {sigma}')
    alpha = (lower - mu) / sigma
    lo = _norm_cdf(alpha)
    if lo >= 1.0 - 1e-12:
        # Entirely in the tail: the schedule has passed. Return the boundary.
        return float(lower)
    u = lo + rng.random() * (1.0 - lo)
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return float(mu + sigma * _norm_ppf(u))


# ---------------------------------------------------------------------------
# GPN training rows
# ---------------------------------------------------------------------------

def build_day_samples(pt_path: str, predicates: list, pid_map: dict,
                      n_counts: int) -> tuple:
    """Encode one main-agent-alone day into ``(X, Y, n_ticks)`` GPN rows.

    One row per tick. The label is the day's **full** goal count vector and is
    therefore constant across the day's ticks — matching the paper, where the
    GPN predicts the episode's final goal from any prefix of it.
    """
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    times = d['times'].tolist()
    s0_sat = s0_satisfied(d['edges'][0].argmax(dim=-1), predicates)
    events = sorted(extract_events_from_pt(pt_path)['events'],
                    key=lambda e: e['t_min'])
    y = day_count_vector(events, pid_map, n_counts)

    rows, counts, ei = [], torch.zeros(len(pid_map), dtype=torch.long), 0
    for t in times:
        while ei < len(events) and events[ei]['t_min'] <= t:
            pid = pid_map.get((int(events[ei]['obj_idx']),
                               int(events[ei]['dest_idx'])))
            if pid is not None:
                counts[pid] = min(int(counts[pid]) + 1, n_counts - 1)
            ei += 1
        rows.append(encode_gpn_input(counts, s0_sat, float(t), n_counts))
    x = torch.stack(rows)
    return x, y.unsqueeze(0).expand(x.shape[0], -1).clone(), x.shape[0]
