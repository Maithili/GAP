"""Branching per-activity task graphs built from LLM-tagged activity segments.

Pure python (no torch). Nodes are actions identified by (obj_idx, dest_idx) in
slate index space; the source location is NOT part of node identity because at
eval time the robot may already have moved the object, so the source would not
match. Edges carry observed-succession counts; branches arise when an activity
has multiple observed successors of the same action (e.g. cereal vs oatmeal
breakfasts).

Sentinel node ids: START = -1, END = -2.
"""

import json
import os
from collections import defaultdict

START = -1
END = -2

# Probability smoothing over the observed successors of a node (plus END).
# No mass is given to never-seen transitions: the open-world action space is
# huge and uniform leakage would dilute the action-emission threshold.
SMOOTHING_ALPHA = 0.5


class TaskGraph:
    """Branching action graph for a single activity."""

    def __init__(self, activity_name: str):
        """Create an empty graph for one activity label."""
        self.activity_name = activity_name
        self.n_segments = 0
        # node id (int >= 0) -> node dict
        self.nodes = {}
        # (obj_idx, dest_idx) -> node id
        self._key_to_id = {}
        # src id -> dst id -> count   (ids include START/END)
        self.edge_counts = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------ build

    def _get_or_add_node(self, obj_idx: int, dest_idx: int,
                         obj_class: str, dest_class: str) -> int:
        """Return node id for ``(obj_idx, dest_idx)``, creating it if needed."""
        key = (int(obj_idx), int(dest_idx))
        if key not in self._key_to_id:
            node_id = len(self.nodes)
            self._key_to_id[key] = node_id
            self.nodes[node_id] = {
                'obj_idx': key[0],
                'dest_idx': key[1],
                'obj_class': obj_class,
                'dest_class': dest_class,
                'count': 0,
                'count_robot': 0,    # Robot share of node mass; read by scripts/plot_task_graphs.py.
                'offset_sum_min': 0.0,
                # Reserved for the beyond-imitation extension (not used yet):
                'robot_doable': None,
                'intrusive': None,
            }
        return self._key_to_id[key]

    def add_segment(self, events: list, weight: float = 1.0,
                    drop_reverted: bool = True) -> None:
        """Add one tagged segment of this activity.

        events: ordered list of dicts with keys
            obj_idx, dest_idx, obj_class, dest_class, t_min and optionally
            'actor' ('robot' contributions accumulate count_robot — the
            v2-PF joint-equilibrium statistic), 'reverted' (human undid the
            robot move), or 'superseded' (robot moved the same object again
            before the human reacted).
        weight: v2-PF continual learning — exponentially decayed by day age
        (gamma^age), so counts are recency-weighted floats.
        drop_reverted: clean re-curation — skip reverted/superseded robot
        events so graphs model human + accepted-robot mass only.
        Consecutive identical actions are collapsed (they are observation
        artifacts); non-adjacent repeats are kept (real structure, e.g.
        fridge-out then fridge-back).
        """
        if not events:
            return
        self.n_segments += weight
        t0 = float(events[0]['t_min'])
        prev_id = START
        prev_key = None
        for ev in events:
            if drop_reverted and (ev.get('reverted') or ev.get('superseded')):
                continue   # bad robot move: human undo or robot self-churn
            key = (int(ev['obj_idx']), int(ev['dest_idx']))
            if key == prev_key:
                continue
            node_id = self._get_or_add_node(
                ev['obj_idx'], ev['dest_idx'],
                ev.get('obj_class', str(ev['obj_idx'])),
                ev.get('dest_class', str(ev['dest_idx'])))
            node = self.nodes[node_id]
            node['count'] += weight
            if ev.get('actor') == 'robot':
                node['count_robot'] = node.get('count_robot', 0) + weight
            node['offset_sum_min'] += (float(ev['t_min']) - t0) * weight
            self.edge_counts[prev_id][node_id] += weight
            prev_id = node_id
            prev_key = key
        self.edge_counts[prev_id][END] += weight

    # ------------------------------------------------------------- inference

    def node_key(self, node_id: int):
        """Return ``(obj_idx, dest_idx)`` for a graph node id."""
        n = self.nodes[node_id]
        return (n['obj_idx'], n['dest_idx'])

    def transition_probs(self, src_id: int) -> dict:
        """Smoothed P(dst | src) over observed successors only."""
        succ = self.edge_counts.get(src_id)
        if not succ:
            return {}
        total = sum(succ.values())
        b = len(succ)
        return {dst: (c + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * b)
                for dst, c in succ.items()}

    def mean_offset(self, node_id: int) -> float:
        """Mean minutes from segment start at which this action occurs."""
        n = self.nodes[node_id]
        return n['offset_sum_min'] / max(n['count'], 1e-9)

    def _descendants(self, node_id: int) -> set:
        """All nodes reachable from node_id along observed edges (END/START
        sentinels excluded). Cached."""
        if not hasattr(self, '_desc_cache'):
            self._desc_cache = {}
        if node_id in self._desc_cache:
            return self._desc_cache[node_id]
        seen = set()
        stack = [d for d in self.edge_counts.get(node_id, {}) if d >= 0]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for d in self.edge_counts.get(x, {}):
                if d >= 0 and d not in seen:
                    stack.append(d)
        self._desc_cache[node_id] = seen
        return seen

    def prereq_nodes(self, node_id: int) -> list:
        """v5 precedence: same-object STRICT graph-ancestors of node_id —
        nodes B (same object) that can reach node_id along observed edges
        while node_id cannot reach B back. For a (pan, cupboard) stow these
        are the (pan, stove) use on the SAME taken path that must happen
        first; gating the stow on their satisfaction kills skip-ahead.
        Strict (B↛C excluded if C↛B too) so oscillation cycles don't
        mutually deadlock; branch-restricted by construction (a node on a
        parallel branch is not an ancestor). Cached per node."""
        if not hasattr(self, '_prereq_cache'):
            self._prereq_cache = {}
        if node_id in self._prereq_cache:
            return self._prereq_cache[node_id]
        obj = self.nodes[node_id]['obj_idx']
        off_c = self.mean_offset(node_id)
        # B is a prerequisite of C iff: same object, B reaches C along edges
        # (on the taken path), AND B's dominant time is earlier than C's.
        # The mean-offset condition de-noises strict reachability: busy
        # activities fuse into one cycle via rare (count-1) mis-ordered
        # segments, but the TRUE order is ~100% dominant in the data, and
        # mean_offset captures it — so reach-AND-earlier recovers the real
        # precedence where pure strict-ancestor goes silent.
        out = []
        for nid in self.nodes:
            if nid == node_id or self.nodes[nid]['obj_idx'] != obj:
                continue
            if (node_id in self._descendants(nid)            # B reaches C
                    and self.mean_offset(nid) < off_c - 1e-6):  # B earlier
                # P(B on the path) — a rare cross-branch node (count 1)
                # should barely gate a common stow; a node on every path
                # (count ≈ n_segments) gates fully.
                p_on_path = min(1.0, self.nodes[nid]['count']
                                / max(self.n_segments, 1e-9))
                out.append((nid, p_on_path))
        self._prereq_cache[node_id] = out
        return out

    def p_node(self, node_id: int) -> float:
        """Marginal frequency of the action within this activity."""
        if self.n_segments == 0:
            return 0.0
        return min(1.0, self.nodes[node_id]['count'] / self.n_segments)

    def match(self, observed: list, max_suffix: int = 6) -> dict:
        """Locate the current position via longest-suffix match.

        observed: ordered (obj_idx, dest_idx) tuples of the in-progress
        segment. Returns {node_id: weight} of candidate current positions
        (normalized), empty dict if nothing matches. Empty observed list
        matches START.
        """
        observed = [tuple(map(int, o)) for o in observed]
        if not observed:
            return {START: 1.0}
        # Pre-index: action key -> node ids (node identity is unique per key,
        # but keep the general form in case that changes).
        for k in range(min(len(observed), max_suffix), 0, -1):
            suffix = observed[-k:]
            cands = defaultdict(float)
            end_id = self._key_to_id.get(suffix[-1])
            if end_id is None:
                continue
            # Walk backwards from the suffix end checking predecessors.
            weight = self._suffix_weight(end_id, suffix[:-1])
            if weight > 0:
                cands[end_id] += weight
            if cands:
                total = sum(cands.values())
                return {n: w / total for n, w in cands.items()}
        return {}

    def _suffix_weight(self, node_id: int, remaining_prefix: list) -> float:
        """Count-weighted evidence that a path with labels remaining_prefix
        immediately precedes node_id."""
        if not remaining_prefix:
            # Any incoming edge (incl. from START) supports this position.
            incoming = sum(self.edge_counts[src].get(node_id, 0)
                           for src in list(self.edge_counts))
            return float(incoming) if incoming > 0 else float(
                self.nodes[node_id]['count'])
        prev_key = remaining_prefix[-1]
        prev_id = self._key_to_id.get(prev_key)
        if prev_id is None:
            return 0.0
        c = self.edge_counts.get(prev_id, {}).get(node_id, 0)
        if c == 0:
            return 0.0
        return c * max(self._suffix_weight(prev_id, remaining_prefix[:-1]), 1e-9)

    def predict_next(self, position: dict, lookahead: int = 3,
                     beam_width: int = 8, decay: float = 0.9) -> list:
        """Score upcoming actions from candidate positions.

        position: {node_id: weight} (e.g. from match(); {START: 1.0} for a
        segment that has not started). Returns [(node_id, score)] sorted desc,
        END excluded. Falls back to p_node ranking if position is empty.
        """
        if not position:
            return self.frequent_nodes()
        scores = defaultdict(float)
        beam = dict(position)
        for step in range(lookahead):
            next_beam = defaultdict(float)
            for src, w in beam.items():
                for dst, p in self.transition_probs(src).items():
                    if dst == END:
                        continue
                    scores[dst] += w * p * (decay ** step)
                    next_beam[dst] += w * p
            if not next_beam:
                break
            beam = dict(sorted(next_beam.items(), key=lambda kv: -kv[1])[:beam_width])
        return sorted(scores.items(), key=lambda kv: -kv[1])

    def frequent_nodes(self, exclude_keys: set = None) -> list:
        """Fallback ranking by marginal node frequency."""
        exclude_keys = exclude_keys or set()
        ranked = [(nid, self.p_node(nid)) for nid in self.nodes
                  if self.node_key(nid) not in exclude_keys]
        return sorted(ranked, key=lambda kv: -kv[1])

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            'activity_name': self.activity_name,
            'n_segments': self.n_segments,
            'nodes': [
                {**self.nodes[nid], 'id': nid,
                 'mean_offset_min': (self.nodes[nid]['offset_sum_min']
                                     / max(self.nodes[nid]['count'], 1e-9))}
                for nid in sorted(self.nodes)
            ],
            'edges': [
                {'src': src, 'dst': dst, 'count': c}
                for src in sorted(self.edge_counts)
                for dst, c in sorted(self.edge_counts[src].items())
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskGraph':
        g = cls(d['activity_name'])
        g.n_segments = d['n_segments']
        for n in d['nodes']:
            nid = n['id']
            g.nodes[nid] = {k: n[k] for k in
                            ('obj_idx', 'dest_idx', 'obj_class', 'dest_class',
                             'count', 'offset_sum_min', 'robot_doable',
                             'intrusive')}
            g.nodes[nid]['count_robot'] = n.get('count_robot', 0)
            g._key_to_id[(n['obj_idx'], n['dest_idx'])] = nid
        for e in d['edges']:
            g.edge_counts[e['src']][e['dst']] = e['count']
        return g


class TaskGraphSet:
    """All activities' graphs + build/save/load."""

    def __init__(self):
        """Empty container; populate via ``build_from_tagged_days`` or ``load``."""
        self.graphs = {}

    @classmethod
    def build_from_tagged_days(cls, tagged_days: list,
                               day_weights: list = None) -> 'TaskGraphSet':
        """tagged_days: list of {'events': [...], 'segments': [...]} dicts in
        the curator's tagged_days/NNN.json format. Rebuilt from scratch every
        time so vocabulary merges propagate for free.
        day_weights (v2-PF): per-day recency weights (gamma^age), aligned
        with tagged_days; None = uniform 1.0 (legacy behavior)."""
        gs = cls()
        for di, day in enumerate(tagged_days):
            w = day_weights[di] if day_weights is not None else 1.0
            events = day['events']
            for seg in day['segments']:
                name = seg['activity_name']
                if name not in gs.graphs:
                    gs.graphs[name] = TaskGraph(name)
                seg_events = [events[i] for i in seg['event_idxs']]
                gs.graphs[name].add_segment(seg_events, weight=w)
        return gs

    def save(self, path: str) -> None:
        """Serialize all activity graphs to JSON."""
        with open(path, 'w') as f:
            json.dump({name: g.to_dict() for name, g in self.graphs.items()},
                      f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TaskGraphSet':
        """Load a ``TaskGraphSet`` written by ``save``."""
        gs = cls()
        with open(path) as f:
            data = json.load(f)
        for name, gd in data.items():
            gs.graphs[name] = TaskGraph.from_dict(gd)
        return gs


# ---------------------------------------------------------------------------
# v2-PF continual learning: prequential gamma selection
# ---------------------------------------------------------------------------

GAMMA_GRID = [0.85, 0.90, 0.95, 0.98, 1.0]   # per-day decay candidates
GAMMA_MIN_HISTORY = 3      # days of history before a day can be scored
_PREQ_FLOOR = 1e-3         # log-lik floor for unseen transitions


def _day_loglik(gs: 'TaskGraphSet', day: dict) -> float:
    """One-step-ahead log-likelihood of a tagged day under a graph set:
    for each segment, walk its events scoring P(node | previous node) with a
    1-hop skip allowance (best intermediate) and a floor for unseen moves."""
    import math
    total = 0.0
    for seg in day['segments']:
        g = gs.graphs.get(seg['activity_name'])
        events = [day['events'][i] for i in seg['event_idxs']]
        if g is None:
            total += len(events) * math.log(_PREQ_FLOOR)
            continue
        prev = START
        prev_key = None
        for ev in events:
            key = (int(ev['obj_idx']), int(ev['dest_idx']))
            # Score under the same representation the graph encodes: add_segment
            # collapses consecutive-identical moves (no self-edge), so skip a
            # repeat here too instead of penalising it at the floor.
            if key == prev_key:
                continue
            prev_key = key
            nid = g._key_to_id.get(key)
            if nid is None:
                total += math.log(_PREQ_FLOOR)
                continue
            tp = g.transition_probs(prev)
            p = tp.get(nid, 0.0)
            if p == 0.0:   # 1-hop skip: best path through one intermediate
                for mid, pm in tp.items():
                    if mid < 0:
                        continue
                    p2 = g.transition_probs(mid).get(nid, 0.0) * pm
                    if p2 > p:
                        p = p2
            total += math.log(max(p, _PREQ_FLOOR))
            prev = nid
    return total


def select_gamma(tagged_days: list, grid: list = None) -> dict:
    """Prequential (one-step-ahead) selection of the per-day decay rate:
    for each candidate gamma, build graphs from days < k (recency-weighted)
    and score day k, summed over k. Training-side data only — the benchmark
    is never touched. Returns {'gamma', 'scores': {gamma: loglik}}."""
    grid = grid or GAMMA_GRID
    scores = {g: 0.0 for g in grid}
    n = len(tagged_days)
    for k in range(GAMMA_MIN_HISTORY, n):
        hist = tagged_days[:k]
        for gamma in grid:
            w = [gamma ** (k - 1 - d) for d in range(k)]
            gs = TaskGraphSet.build_from_tagged_days(hist, day_weights=w)
            scores[gamma] += _day_loglik(gs, tagged_days[k])
    # Tie-break toward the LARGEST gamma (1.0 = no discounting = the prior):
    # on a no-information tie the prequential criterion is indifferent, so we
    # default to not throwing away old mass rather than to max discount (0.85).
    best = max(scores, key=lambda g: (scores[g], g))
    return {'gamma': best, 'scores': scores, 'n_days': n}
