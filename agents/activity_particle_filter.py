"""Particle filter for online activity recognition over task graphs.

Replaces the eval-time LLM tagger (H11: 60-68% committed accuracy, abstention
machine) with a calibrated filter over the structures we already trust
(curation purity 0.80-0.84). Design per 2026-06-12 discussion:

- Particle state: completed history [(activity, t_open, t_close)] + a small
  set of OPEN activities, each with a graph position marker and the set of
  satisfied nodes (re-armable; robot-satisfied marking reserved for the
  closed-loop integration).
- On each observed human move the particle weighs three interpretations:
  (a) the move continues one of its open activities — credit = best-path
      probability from the current marker to the matching node (partial
      credit by path length falls out of the product of branch probs);
  (b) the move opens a new activity — credit = entry-path probability from
      START (<= ENTRY_HOPS) x a clock prior derived from the curated
      training segments (transformer hook point for the closed-loop version);
  (c) background — derived from the training fraction of unassigned events.
  The particle's weight is multiplied by the TOTAL (marginal likelihood);
  the committed interpretation is sampled proportionally. The background
  floor is the 'epsilon' that prevents instant particle death.
- Re-observation of a satisfied node gets the node's expected multiplicity
  (count / n_segments) as credit, floored at background: multi-instance
  classes and within-activity oscillation are legitimate repeats.
- Recognition is exact (obj, dest): a human move credits an activity only
  when it matches the graph node exactly.
- Open activities: idle hazard in weights (v4+), stochastic close at END,
  undo penalty on human reactions (v4+).

Derived from the memory dir (no hand-tuned knobs): clock priors, background
rate, stale gap, re-observation multiplicity. The remaining constants are
structural and documented: N_PARTICLES, MAX_OPEN, ENTRY_HOPS, CONT_HOPS.
"""

import math
import random
from collections import defaultdict

from agents.task_graph import START, END

N_PARTICLES = 1000
MAX_OPEN = 3          # concurrent open activities per particle
CONT_HOPS = 4         # how far ahead a continuation may skip
STD_FLOOR_MIN = 20.0  # clock prior std floor (few training segments)
ADVANCE_P = 0.5       # P(a particle advances its marker over a robot
                      # placement) — stochastic, weight untouched: half the
                      # ensemble assumes the human will accept our chunk,
                      # half does not; the next human move sorts them out
BG_LABEL = '__background__'


class _Open:
    """One open activity hypothesis inside a particle (position + satisfied nodes)."""
    __slots__ = ('act', 'pos', 'opened_t', 'last_t', 'done', 'robot_done',
                 'anchor_t', 'anchor_pos')

    def __init__(self, act, pos, t):
        self.act = act
        self.pos = pos
        self.opened_t = t
        self.last_t = t
        self.done = frozenset([pos]) if pos != START else frozenset()
        self.robot_done = frozenset()   # nodes satisfied by the ROBOT
        # Pace anchor: (clock time, node) of the last HUMAN advance. Robot
        # advances never update it — early robot actions don't move the human's
        # timeline. Node times are dead-reckoned as
        # anchor_t + (mean_offset(node) - mean_offset(anchor_pos)).
        self.anchor_t = t
        self.anchor_pos = pos

    def clone(self):
        """Shallow copy for resampling."""
        o = _Open.__new__(_Open)
        (o.act, o.pos, o.opened_t, o.last_t, o.done, o.robot_done,
         o.anchor_t, o.anchor_pos) = (
            self.act, self.pos, self.opened_t, self.last_t, self.done,
            self.robot_done, self.anchor_t, self.anchor_pos)
        return o


class _Particle:
    """PF state: open activities, closed history, last assignment label."""
    __slots__ = ('open', 'history', 'last_assign')

    def __init__(self):
        self.open = []        # list[_Open]
        self.history = []     # (act, opened_t, closed_t)
        self.last_assign = BG_LABEL

    def clone(self):
        p = _Particle.__new__(_Particle)
        p.open = [o.clone() for o in self.open]
        p.history = list(self.history)
        p.last_assign = self.last_assign
        return p


def derive_priors(tagged_days: list) -> dict:
    """Clock priors, background rate, stale gap — all from the training tags."""
    starts = defaultdict(list)
    gaps = []
    n_events = 0
    n_assigned = 0
    pairs = set()
    for day in tagged_days:
        ev = day['events']
        n_events += len(ev)
        for e in ev:
            pairs.add((e['obj_idx'], e['dest_idx']))
        for seg in day['segments']:
            idxs = seg['event_idxs']
            n_assigned += len(idxs)
            starts[seg['activity_name']].append(ev[idxs[0]]['t_min'])
            ts = sorted(ev[i]['t_min'] for i in idxs)
            gaps.extend(b - a for a, b in zip(ts, ts[1:]))
    clock = {}
    for act, ss in starts.items():
        mu = sum(ss) / len(ss)
        var = sum((s - mu) ** 2 for s in ss) / max(len(ss) - 1, 1)
        clock[act] = (mu, max(math.sqrt(var), STD_FLOOR_MIN))
    unassigned_frac = max(1e-3, 1.0 - n_assigned / max(n_events, 1))
    gaps.sort()
    stale_gap = gaps[int(0.95 * (len(gaps) - 1))] if gaps else 120.0
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 30.0
    return {
        'clock': clock,
        'p_bg': unassigned_frac / max(len(pairs), 1),
        'stale_gap_min': stale_gap,
        'cont_tau_min': max(stale_gap, mean_gap, 1.0),
    }


class GraphParticleFilter:
    """Online activity recognition over ``TaskGraphSet`` structures (v6 core).

    TL;DR: on each human move, reweight particles by continue/open/background
    likelihood; robot placements update markers without full observe().
    """

    def __init__(self, graph_set, tagged_days, seed=0,
                 n_particles=N_PARTICLES, p_bg_override=None, robot_opens=False,
                 single_belief=False):
        self.graphs = graph_set.graphs   # activity_name -> TaskGraph
        self.priors = derive_priors(tagged_days)
        if p_bg_override is not None:     # fixed background rate (ablation)
            self.priors['p_bg'] = float(p_bg_override)
        # When True, a robot placement runs the full observe()-style particle
        # state update (can open/continue activities) but never touches weights.
        self.robot_opens = bool(robot_opens)
        # single_belief (ablation): collapse the filter to ONE deterministic
        # hypothesis — 1 particle, argmax data-association (no sampling), and
        # threshold-0.5 open/close decisions. Removes the distribution over
        # associations entirely (no ensemble, no recovery from a wrong guess).
        self._single = bool(single_belief)
        self.rng = random.Random(seed)
        self.n = 1 if self._single else n_particles
        self.reset()

    def _draw(self, p: float) -> bool:
        """Bernoulli(p), or its deterministic MAP (p > 0.5) in single_belief."""
        return (p > 0.5) if self._single else (self.rng.random() < p)

    def reset(self):
        """New day: uniform weights, empty robot-key set, clear path cache."""
        self.particles = [_Particle() for _ in range(self.n)]
        self.weights = [1.0 / self.n] * self.n
        self._path_cache = {}
        self._robot_keys_today = set()   # (obj, dest) the robot placed today
        self._t_last_obs = None

    # ------------------------------------------------------------ credits

    def _path_credit(self, act, src, key):
        """Best-path probability from src to the node matching key (exact),
        within CONT_HOPS."""
        ck = (act, src, key)
        if ck in self._path_cache:
            return self._path_cache[ck]
        g = self.graphs[act]
        targets = {}   # node id -> emission factor
        exact = g._key_to_id.get(key)
        if exact is not None:
            targets[exact] = 1.0
        if not targets:
            self._path_cache[ck] = (0.0, None)
            return (0.0, None)
        best, best_nid = 0.0, None
        frontier = {src: 1.0}
        for _ in range(CONT_HOPS):
            nxt = defaultdict(float)
            for s, p in frontier.items():
                for dst, tp in g.transition_probs(s).items():
                    if dst < 0:   # END
                        continue
                    q = p * tp
                    if dst in targets and q * targets[dst] > best:
                        best, best_nid = q * targets[dst], dst
                    if q > nxt[dst]:
                        nxt[dst] = q
            if not nxt:
                break
            frontier = dict(sorted(nxt.items(), key=lambda kv: -kv[1])[:16])
        self._path_cache[ck] = (best, best_nid)
        return (best, best_nid)

    def _reobs_credit(self, act, nid):
        """Expected extra occurrences of an already-satisfied node (multi-
        instance classes, oscillation), floored at background."""
        g = self.graphs[act]
        mult = g.nodes[nid]['count'] / max(g.n_segments, 1)
        return max(mult - 1.0, 0.0) / max(mult, 1.0)

    def _clock(self, act, t):
        """Gaussian clock prior P(start time | activity) at minute *t*."""
        if act not in self.priors['clock']:
            raise KeyError(
                f"No clock prior for activity {act!r} — every graph activity is "
                "seeded from a training segment, so this should never happen.")
        mu, sd = self.priors['clock'][act]
        z = (t - mu) / sd
        return math.exp(-0.5 * z * z)

    # ------------------------------------------------------------ update

    def _update_particle_state(self, pa, key, t, p_bg):
        """Build cont/open/bg options for one particle, sample one, apply the
        state change, and return the total option mass (used for weighting by
        the caller). Shared by observe() (human) and observe_robot() when
        robot_opens is set."""
        options = []   # (credit, kind, payload)
        for j, oa in enumerate(pa.open):
            g = self.graphs[oa.act]
            exact = g._key_to_id.get(key)
            if exact is not None and exact in oa.done:
                c = max(self._reobs_credit(oa.act, exact), p_bg)
                options.append((c, 'cont', (j, exact)))
            else:
                c, nid = self._path_credit(oa.act, oa.pos, key)
                if c > 0:
                    options.append((c, 'cont', (j, nid)))
        if len(pa.open) < MAX_OPEN:
            open_acts = {oa.act for oa in pa.open}
            for act in self.graphs:
                if act in open_acts:
                    continue
                g = self.graphs[act]
                c, nid = self._path_credit(act, START, key)
                for rk in self._robot_keys_today:
                    rnid = g._key_to_id.get(rk)
                    if rnid is None:
                        continue
                    c2, nid2 = self._path_credit(act, rnid, key)
                    if c2 > c:
                        c, nid = c2, nid2
                if c > 0:
                    options.append(
                        (c * self._clock(act, t), 'open', (act, nid)))
        options.append((p_bg, 'bg', None))

        total = sum(c for c, _, _ in options)
        if self._single:
            # deterministic MAP: commit to the single most-likely association.
            idx = max(range(len(options)), key=lambda i: options[i][0])
        else:
            r = self.rng.random() * total
            acc, idx = 0.0, len(options) - 1
            for i, (c, _, _) in enumerate(options):
                acc += c
                if r <= acc:
                    idx = i
                    break
        c, kind, payload = options[idx]
        if kind == 'cont':
            j, nid = payload
            oa = pa.open[j]
            oa.pos = nid
            oa.last_t = t
            oa.done = oa.done | {nid}
            oa.anchor_t = t
            oa.anchor_pos = nid
            pa.last_assign = oa.act
            self._maybe_close(pa, j, t)
        elif kind == 'open':
            act, nid = payload
            pa.open.append(_Open(act, nid, t))
            pa.last_assign = act
            self._maybe_close(pa, len(pa.open) - 1, t)
        else:
            pa.last_assign = BG_LABEL
        return total

    def observe(self, obj_idx, dest_idx, t):
        """Incorporate one human relocation; resample if ESS drops."""
        key = (int(obj_idx), int(dest_idx))
        p_bg = self.priors['p_bg']
        stale = self.priors['stale_gap_min']
        tau = self.priors['cont_tau_min']
        dt = (t - self._t_last_obs) if self._t_last_obs is not None else 0.0
        self._t_last_obs = t
        new_w = []
        for i, pa in enumerate(self.particles):
            w_pre = self.weights[i]
            if dt > 0 and pa.open:
                w_pre *= math.exp(-dt * len(pa.open) / tau)
                p_close = 1.0 - math.exp(-dt / stale)
                for j in range(len(pa.open) - 1, -1, -1):
                    oa = pa.open[j]
                    if oa.last_t < t and self._draw(p_close):
                        pa.history.append((oa.act, oa.opened_t, oa.last_t))
                        pa.open.pop(j)

            total = self._update_particle_state(pa, key, t, p_bg)
            new_w.append(w_pre * total)

        s = sum(new_w)
        if s <= 0:
            raise RuntimeError(
                "PF particle weights collapsed to zero — should never happen "
                "(the p_bg background floor keeps every weight > 0); indicates a bug.")
        self.weights = [w / s for w in new_w]
        ess = 1.0 / sum(w * w for w in self.weights)
        if ess < self.n / 2:
            self._resample()

    def _resample(self):
        """Systematic resample when effective sample size falls below N/2."""
        n = self.n
        positions = [(self.rng.random() + k) / n for k in range(n)]
        cum, j, picks = 0.0, 0, []
        cumw = []
        for w in self.weights:
            cum += w
            cumw.append(cum)
        for pos in positions:
            while j < n - 1 and cumw[j] < pos:
                j += 1
            picks.append(j)
        self.particles = [self.particles[k].clone() for k in picks]
        self.weights = [1.0 / n] * n

    # ------------------------------------------------------------ readout

    def _maybe_close(self, pa, j, t):
        """Stochastic close at the graph's learned end."""
        oa = pa.open[j]
        p_end = self.graphs[oa.act].transition_probs(oa.pos).get(END, 0.0)
        if p_end > 0 and self._draw(p_end):
            pa.history.append((oa.act, oa.opened_t, t))
            pa.open.pop(j)

    def observe_robot(self, obj_idx, dest_idx, t):
        """Robot placement: mark node robot-done; stochastic marker advance.

        When robot_opens is set, the placement instead drives the full
        observe()-style particle state update (can open/continue activities),
        but weights are never modified (no reweight, no resample)."""
        key = (int(obj_idx), int(dest_idx))
        self._robot_keys_today.add(key)
        if self.robot_opens:
            p_bg = self.priors['p_bg']
            for pa in self.particles:
                # preserve robot-done bookkeeping (used by undo/reaction)
                for oa in pa.open:
                    nid = self.graphs[oa.act]._key_to_id.get(key)
                    if nid is not None:
                        oa.robot_done = oa.robot_done | {nid}
                self._update_particle_state(pa, key, t, p_bg)  # weights untouched
            return
        for pa in self.particles:
            for oa in pa.open:
                nid = self.graphs[oa.act]._key_to_id.get(key)
                if nid is not None:
                    oa.done = oa.done | {nid}
                    oa.robot_done = oa.robot_done | {nid}
                    if self._draw(ADVANCE_P):
                        credit, _ = self._path_credit(oa.act, oa.pos, key)
                        if credit > 0:
                            oa.pos = nid
                            oa.last_t = t

    def observe_reaction(self, obj_idx, robot_dest_idx):
        """Human undo of a robot placement: re-arm node, penalize weights."""
        key = (int(obj_idx), int(robot_dest_idx))
        p_bg = self.priors['p_bg']
        changed = False
        for i, pa in enumerate(self.particles):
            for oa in pa.open:
                g = self.graphs[oa.act]
                nid = g._key_to_id.get(key)
                if nid is not None and nid in oa.robot_done:
                    oa.done = oa.done - {nid}
                    oa.robot_done = oa.robot_done - {nid}
                    c = g.p_node(nid)
                    self.weights[i] *= p_bg / (p_bg + c)
                    changed = True
        if changed:
            s = sum(self.weights)
            if s <= 0:
                raise RuntimeError(
                    "PF particle weights collapsed to zero in observe_reaction — "
                    "should never happen (p_bg floor keeps weights > 0); indicates a bug.")
            self.weights = [w / s for w in self.weights]
            ess = 1.0 / sum(w * w for w in self.weights)
            if ess < self.n / 2:
                self._resample()

    def posterior_last_assign(self):
        """P(activity | the most recent observed move)."""
        post = defaultdict(float)
        for pa, w in zip(self.particles, self.weights):
            post[pa.last_assign] += w
        return dict(sorted(post.items(), key=lambda kv: -kv[1]))

    def posterior_open(self):
        """P(activity is open right now) — actuation readout."""
        post = defaultdict(float)
        for pa, w in zip(self.particles, self.weights):
            for oa in pa.open:
                post[oa.act] += w
        return dict(sorted(post.items(), key=lambda kv: -kv[1]))

    def actuation_view(self, *, pace=True):
        """Per open activity: (P(open), position dist, base_t, done_mass).

        pace=True: base_t from human pace anchor (v5/v6 actuation path).
        """
        p_open = defaultdict(float)
        pos = defaultdict(lambda: defaultdict(float))
        base = defaultdict(float)
        done = defaultdict(lambda: defaultdict(float))
        for pa, w in zip(self.particles, self.weights):
            for oa in pa.open:
                p_open[oa.act] += w
                pos[oa.act][oa.pos] += w
                if pace:
                    a_off = (self.graphs[oa.act].mean_offset(oa.anchor_pos)
                             if oa.anchor_pos >= 0 else 0.0)
                    base[oa.act] += w * (oa.anchor_t - a_off)
                else:
                    base[oa.act] += w * oa.opened_t
                for nid in oa.done:
                    done[oa.act][nid] += w
        out = {}
        for act, p in p_open.items():
            d = pos[act]
            tot = sum(d.values())
            dm = {nid: m / p for nid, m in done[act].items()}
            out[act] = (p, {n: v / tot for n, v in d.items()},
                        base[act] / p, dm)
        return out

    def map_segments(self):
        """The MAP particle's day as tagger-style segments (transformer input)."""
        i = max(range(len(self.weights)), key=lambda k: self.weights[k])
        pa = self.particles[i]
        segs = [{'activity_name': a, 'start_min': t0, 'end_min': t1}
                for a, t0, t1 in pa.history]
        segs.extend({'activity_name': oa.act, 'start_min': oa.opened_t,
                     'end_min': oa.last_t} for oa in pa.open)
        return sorted(segs, key=lambda s: s['start_min'])
