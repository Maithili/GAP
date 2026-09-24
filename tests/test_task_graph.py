"""Unit checks for agents/task_graph.py. Pure CPU, run directly:

    python tests/test_task_graph.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.task_graph import TaskGraph, TaskGraphSet, START, END


def _events(keys, t0=480.0):
    return [{'obj_idx': o, 'dest_idx': d,
             'obj_class': f'obj{o}', 'dest_class': f'dest{d}',
             't_min': t0 + 10 * i}
            for i, (o, d) in enumerate(keys)]


def build_breakfast():
    """3 cereal days + 1 oatmeal day, converging on a shared final action."""
    g = TaskGraph('breakfast')
    for _ in range(3):
        g.add_segment(_events([(1, 10), (2, 20), (3, 30)]))   # cereal branch
    g.add_segment(_events([(1, 10), (4, 40), (7, 70)]))       # oatmeal branch
    return g


def test_counts_and_branching():
    g = build_breakfast()
    assert g.n_segments == 4
    first = g._key_to_id[(1, 10)]
    probs = g.transition_probs(START)
    assert set(probs) == {first} and abs(probs[first] - 1.0) < 1e-9
    succ = g.transition_probs(first)
    cereal, oatmeal = g._key_to_id[(2, 20)], g._key_to_id[(4, 40)]
    assert set(succ) == {cereal, oatmeal}
    assert abs(succ[cereal] - (3 + 0.5) / (4 + 0.5 * 2)) < 1e-9
    assert abs(succ[oatmeal] - (1 + 0.5) / (4 + 0.5 * 2)) < 1e-9
    print('ok: counts_and_branching')


def test_duplicate_collapse_and_repeats():
    g = TaskGraph('x')
    # consecutive duplicate collapsed
    g.add_segment(_events([(1, 10), (1, 10), (2, 20)]))
    assert g.nodes[g._key_to_id[(1, 10)]]['count'] == 1
    # non-adjacent repeat kept (fridge-out ... fridge-back pattern)
    g2 = TaskGraph('y')
    g2.add_segment(_events([(5, 50), (6, 60), (5, 50)]))
    n5 = g2._key_to_id[(5, 50)]
    assert g2.nodes[n5]['count'] == 2
    assert g2.edge_counts[g2._key_to_id[(6, 60)]][n5] == 1
    print('ok: duplicate_collapse_and_repeats')


def test_match_and_predict():
    g = build_breakfast()
    # empty observed -> START; first prediction is the shared opening action
    pos = g.match([])
    assert pos == {START: 1.0}
    preds = g.predict_next(pos, lookahead=1)
    assert g.node_key(preds[0][0]) == (1, 10)

    # after the cereal-branch action, next is its branch-specific successor
    pos = g.match([(1, 10), (2, 20)])
    assert set(pos) == {g._key_to_id[(2, 20)]}
    preds = g.predict_next(pos, lookahead=2)
    assert g.node_key(preds[0][0]) == (3, 30)
    # oatmeal successor unreachable from the cereal position
    assert (4, 40) not in [g.node_key(n) for n, _ in preds]
    assert (7, 70) not in [g.node_key(n) for n, _ in preds]

    # ambiguous position (only opening action seen): both branches scored,
    # cereal branch higher
    pos = g.match([(1, 10)])
    preds = dict(g.predict_next(pos, lookahead=1))
    cereal, oatmeal = g._key_to_id[(2, 20)], g._key_to_id[(4, 40)]
    assert preds[cereal] > preds[oatmeal]

    # novel last action -> no match -> frequency fallback, headed by the
    # always-present opening action
    assert g.match([(99, 99)]) == {}
    fallback = g.frequent_nodes()
    assert g.node_key(fallback[0][0]) == (1, 10)
    assert abs(fallback[0][1] - 1.0) < 1e-9
    print('ok: match_and_predict')


def test_suffix_longest_first():
    """A shared action whose successor depends on what preceded it."""
    g = TaskGraph('z')
    for _ in range(2):
        g.add_segment(_events([(1, 1), (5, 5), (2, 2)]))
    for _ in range(2):
        g.add_segment(_events([(3, 3), (5, 5), (4, 4)]))
    # suffix (1,1),(5,5) pins the position; prediction must be (2,2)
    pos = g.match([(1, 1), (5, 5)])
    preds = g.predict_next(pos, lookahead=1)
    top_keys = [g.node_key(n) for n, _ in preds]
    assert top_keys[0] == (2, 2)
    # node identity is per-(obj,dest) so (5,5) is one node; with only (5,5)
    # observed both successors appear, tied
    preds2 = dict(g.predict_next(g.match([(5, 5)]), lookahead=1))
    k2, k4 = g._key_to_id[(2, 2)], g._key_to_id[(4, 4)]
    assert abs(preds2[k2] - preds2[k4]) < 1e-9
    print('ok: suffix_longest_first')


def test_mean_offset():
    g = TaskGraph('m')
    # events at +0, +10, +20 minutes from segment start (t0=480)
    g.add_segment(_events([(1, 10), (2, 20), (3, 30)]))
    # second segment shifted: same actions at +0, +10, +20 from t0=600
    g.add_segment(_events([(1, 10), (2, 20), (3, 30)], t0=600))
    assert abs(g.mean_offset(g._key_to_id[(1, 10)]) - 0.0) < 1e-9
    assert abs(g.mean_offset(g._key_to_id[(2, 20)]) - 10.0) < 1e-9
    assert abs(g.mean_offset(g._key_to_id[(3, 30)]) - 20.0) < 1e-9
    print('ok: mean_offset')


def test_serialization_roundtrip():
    g = build_breakfast()
    gs = TaskGraphSet()
    gs.graphs['breakfast'] = g
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'task_graphs.json')
        gs.save(path)
        gs2 = TaskGraphSet.load(path)
    g2 = gs2.graphs['breakfast']
    assert g2.n_segments == g.n_segments
    pos1, pos2 = g.match([(1, 10)]), g2.match([(1, 10)])
    assert pos1 == pos2
    assert g.predict_next(pos1) == g2.predict_next(pos2)
    # reserved extension fields survive
    assert all(n['robot_doable'] is None for n in g2.nodes.values())
    print('ok: serialization_roundtrip')


def test_build_from_tagged_days():
    events = _events([(1, 10), (2, 20), (8, 80), (9, 90)])
    day = {'events': events,
           'segments': [
               {'activity_name': 'breakfast', 'event_idxs': [0, 1]},
               {'activity_name': 'dishes', 'event_idxs': [2, 3]},
           ]}
    gs = TaskGraphSet.build_from_tagged_days([day, day])
    assert set(gs.graphs) == {'breakfast', 'dishes'}
    assert gs.graphs['breakfast'].n_segments == 2
    assert gs.graphs['dishes'].n_segments == 2
    assert (8, 80) in gs.graphs['dishes']._key_to_id
    assert (8, 80) not in gs.graphs['breakfast']._key_to_id
    print('ok: build_from_tagged_days')


def test_drop_reverted_and_superseded():
    """Clean re-curation skips bad robot events (human undo + robot churn)."""
    g = TaskGraph('act')
    ev_ok = _events([(1, 10), (2, 20)])
    ev_rev = _events([(3, 30)])
    ev_rev[0]['actor'] = 'robot'
    ev_rev[0]['reverted'] = True
    ev_sup = _events([(4, 40)])
    ev_sup[0]['actor'] = 'robot'
    ev_sup[0]['superseded'] = True
    g.add_segment(ev_ok + ev_rev + ev_sup)
    assert (1, 10) in g._key_to_id
    assert (2, 20) in g._key_to_id
    assert (3, 30) not in g._key_to_id
    assert (4, 40) not in g._key_to_id
    print('ok: drop_reverted_and_superseded')


if __name__ == '__main__':
    test_counts_and_branching()
    test_duplicate_collapse_and_repeats()
    test_match_and_predict()
    test_suffix_longest_first()
    test_mean_offset()
    test_serialization_roundtrip()
    test_build_from_tagged_days()
    test_drop_reverted_and_superseded()
    print('ALL task_graph checks passed')
