"""Unit checks for the NOPA baseline (agents/robot_policy_nopa.py).

Pure CPU, no training and no simulator: the policy is built with a stubbed
goal-proposal network so the Alg-2 / Alg-3 logic can be exercised directly.

    python -m pytest tests/test_nopa.py
"""

import glob
import math
import os
import random

import pytest
import torch

from agents.nopa_utils import (
    build_day_samples, build_predicate_vocab, encode_gpn_input, fit_clock_prior,
    s0_satisfied, sample_truncnormal, time_features)
from agents.robot_policy_nopa import REACTION_WINDOW_MIN, RobotPolicyNOPA

HH_DIR = os.path.join(os.path.dirname(__file__), '..', 'external', 'HOMER_PLUS',
                      'HouseholdA', 'processed_seqLM_coarse', 'train')


# ---------------------------------------------------------------------------
# Predicate vocabulary / encoding
# ---------------------------------------------------------------------------

def test_vocab_and_day_samples_round_trip():
    files = sorted(glob.glob(os.path.join(HH_DIR, '*.pt')))[:3]
    if not files:
        pytest.skip('HOMER+ HouseholdA data not available')
    predicates, pid_map = build_predicate_vocab(files)
    assert len(predicates) == len(pid_map) > 0
    assert predicates == sorted(predicates), 'vocab must be reproducible-sorted'
    assert all(pid_map[p] == i for i, p in enumerate(predicates))

    clock = fit_clock_prior(files, pid_map)
    assert set(clock) == set(range(len(predicates))), \
        'every vocabulary predicate must have an arrival-clock prior'
    assert all(sigma >= 20.0 for _, sigma, _ in clock.values())

    x, y, n_ticks = build_day_samples(files[0], predicates, pid_map, 4)
    assert x.shape == (n_ticks, len(predicates) * 5 + 4)
    assert y.shape == (n_ticks, len(predicates))
    # The label is the day's FULL goal and is constant across its ticks.
    assert torch.equal(y[0], y[-1])
    assert int((y[0] >= 1).sum()) > 0
    assert int(y.max()) <= 3


def test_encode_gpn_input_layout():
    counts = torch.tensor([0, 2, 1])
    s0_sat = torch.tensor([True, False, False])
    x = encode_gpn_input(counts, s0_sat, 600.0, 4)
    assert x.shape == (3 * 5 + 4,)
    assert x[:5].tolist() == [1, 0, 0, 0, 1]     # count 0, satisfied at s0
    assert x[5:10].tolist() == [0, 0, 1, 0, 0]   # count 2, not satisfied at s0
    assert torch.allclose(x[-4:], time_features(600.0))


def test_count_clipping_at_n_counts():
    x = encode_gpn_input(torch.tensor([9]), torch.tensor([False]), 600.0, 4)
    assert x[:4].tolist() == [0, 0, 0, 1]        # clipped into the 3+ bin


def test_truncnormal_respects_lower_bound():
    rng = random.Random(0)
    draws = [sample_truncnormal(600.0, 60.0, 700.0, rng) for _ in range(500)]
    assert min(draws) >= 700.0 - 1e-6
    # Truncation from below pulls the mean above the untruncated mu.
    assert sum(draws) / len(draws) > 700.0


# ---------------------------------------------------------------------------
# Policy logic  (stubbed GPN, no training, no simulator)
# ---------------------------------------------------------------------------

PREDICATES = [(0, 10), (1, 11), (2, 12)]


class _StubGPN:
    """Returns a fixed count distribution; stands in for the trained network."""

    def __init__(self, probs):
        self.probs = probs

    def __call__(self, x):
        return torch.log(self.probs).unsqueeze(0)


def _make_policy(**pol_overrides):
    cfg = {
        'benchmark': {'household': 'A'},
        'robot': {'nopa': {
            'common_data_path': os.path.join(
                os.path.dirname(__file__), '..', 'external', 'HOMER_PLUS',
                'Household{household}', 'processed_seqLM_coarse',
                'common_data.json'),
            'solo_days': 5,
            'include_finetune_days': False,
            'gpn': {'n_counts': 4, 'hidden': 8, 'h': 8, 'head_hidden': 4,
                    'dropout': 0.0, 'lr': 1e-3, 'batch_size': 8, 'val_days': 1,
                    'early_stop_patience': 5},
            'policy': {'K': 4, 'T_prop': 15, 'L_max': 100.0, 'w_r': 1.0,
                       'w_c': 1.0, 'w_m': 5.0, 'max_actions': 1,
                       'lm_units': 'ticks', 'min_value': 0.0,
                       'sigma_floor_min': 20.0, 'goal_source': 'gpn',
                       'inverse_planning': True, 'restore': True,
                       'restore_all_initial': False, 'plan_order_gate': True,
                       'argmax_goal': False,
                       'seed': 0},
        }},
    }
    cfg['robot']['nopa']['policy'].update(pol_overrides)
    p = RobotPolicyNOPA(cfg)
    p.predicates = PREDICATES
    p._pid = {q: i for i, q in enumerate(PREDICATES)}
    p.clock = {0: (700.0, 30.0, 5), 1: (800.0, 30.0, 5), 2: (900.0, 30.0, 5)}
    p.n_counts = 4
    # Every predicate is wanted exactly once.
    p.gpn = _StubGPN(torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 3))
    p._s0_locs = torch.tensor([100, 101, 102] + [0] * 10)
    p._s0_sat = torch.tensor([False, False, False])
    p._counts = torch.zeros(3, dtype=torch.long)
    return p


def test_is_undo_true_for_revert_but_false_for_redirect():
    """A revert of a robot placement is a reaction, not goal evidence.

    An LLM-human *redirect* sends the object somewhere new (dest != prev) and
    must survive as real evidence.
    """
    p = _make_policy()
    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]
    revert = {'obj_idx': 0, 'dest_idx': 100, 'prev_idx': 10}
    assert p._is_undo(revert, 605.0) is True
    assert p._placements[0]['status'] == 'reverted'

    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]
    redirect = {'obj_idx': 0, 'dest_idx': 55, 'prev_idx': 10}
    assert p._is_undo(redirect, 605.0) is False


def test_is_undo_false_outside_reaction_window():
    p = _make_policy()
    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]
    late = {'obj_idx': 0, 'dest_idx': 100, 'prev_idx': 10}
    assert p._is_undo(late, 600.0 + REACTION_WINDOW_MIN + 10.0) is False


def test_resample_and_rejection():
    p = _make_policy()
    p._resample(600.0, 'day_start')
    assert len(p._particles) == 4
    # Every particle wants all three predicates, ordered by the clock prior.
    for k in p._particles:
        assert set(k['tau']) == {0, 1, 2}
        taus = [t for _, t in k['plan']]
        assert taus == sorted(taus)
        assert min(taus) >= 600.0

    # Observing predicate 0 (which is in every plan prefix) keeps all particles
    # and removes it from their plans.
    p._particles = [k for k in p._particles if 0 in k['prefix']]
    assert len(p._particles) == 4
    p._consume(0)
    assert all(0 not in k['tau'] for k in p._particles)

    # A predicate no particle plans rejects all of them.
    p._particles = [k for k in p._particles if 99 in k['prefix']]
    assert p._particles == []


def test_value_prefers_confident_distant_goal_edges():
    p = _make_policy()
    p._resample(600.0, 'day_start')
    cur = p._s0_locs.clone()
    cands = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert (0, 10) in cands and cands[(0, 10)]['p_e'] == 1.0
    near = p._value(0, 10, cands[(0, 10)], cur)
    far = p._value(2, 12, cands[(2, 12)], cur)
    assert far > near > 0, 'a confident, further-out subgoal must score higher'

    # An uncertain edge must score below the same edge held with certainty.
    weak = dict(cands[(2, 12)]); weak['p_e'] = 0.1
    assert p._value(2, 12, weak, cur) < far


def test_restore_edge_only_for_robot_placements_no_longer_wanted():
    p = _make_policy()
    p._resample(600.0, 'day_start')
    cur = p._s0_locs.clone()
    cur[0] = 10          # robot placed object 0 at its destination
    cur[1] = 77          # the HUMAN moved object 1 somewhere off s0
    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]

    # Object 0 is still wanted by every particle -> no restore edge yet.
    keys = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert (0, 100) not in keys
    # The human-moved object is never a restore candidate under the default.
    assert (1, 101) not in keys

    # Once the goal no longer wants object 0, restoring it becomes a candidate
    # and scores positively (V = -w_c*1 + w_m*1).
    p._counts[0] = 1
    for k in p._particles:
        k['tau'].pop(0, None)
        k['plan'] = [s for s in k['plan'] if s[0] != 0]
        k['prefix'] = {q for q, _ in k['plan'][:15]}
    cands = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert (0, 100) in cands and cands[(0, 100)]['kind'] == 'restore'
    v = p._value(0, 100, cands[(0, 100)], cur)
    assert math.isclose(v, -1.0 + 5.0)


def test_restore_all_initial_is_the_literal_reading():
    p = _make_policy(restore_all_initial=True)
    p._resample(600.0, 'day_start')
    cur = p._s0_locs.clone()
    cur[1] = 77          # moved by the human only
    cands = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert (1, 101) in cands, 'literal Alg-3 admits every initial-state edge'


def test_ours_return_ablation_drops_restore_and_zeroes_wm():
    p = _make_policy(restore=False)
    p._resample(600.0, 'day_start')
    cur = p._s0_locs.clone()
    cur[0] = 10
    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]
    p._counts[0] = 1
    for k in p._particles:
        k['tau'].pop(0, None)
        k['plan'] = [s for s in k['plan'] if s[0] != 0]
        k['prefix'] = {q for q, _ in k['plan'][:15]}
    cands = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert not any(c['kind'] == 'restore' for c in cands.values())
    # w_m is pinned to 0, so moving an object off s0 carries no penalty.
    assert math.isclose(p._value(1, 11, cands[(1, 11)], cur),
                        cands[(1, 11)]['p_e'] * max(0.0, cands[(1, 11)]['L_M'] - 1) - 1.0)


def test_argmax_goal_gives_one_particle():
    p = _make_policy(argmax_goal=True)
    p._resample(600.0, 'day_start')
    assert len(p._particles) == 1


def test_uniform_goal_source_ignores_the_network():
    p = _make_policy(goal_source='uniform')
    p.gpn = None                       # must never be consulted
    probs = p._goal_probs(600.0)
    assert probs.shape == (3, 4)
    assert torch.allclose(probs, torch.full((3, 4), 0.25))


def test_lm_units_steps_uses_plan_rank():
    p = _make_policy(lm_units='steps')
    p._resample(600.0, 'day_start')
    cands = dict(p._candidate_edges(p._s0_locs.clone(), 600.0, torch.ones(13, dtype=torch.bool)))
    # Three predicates, so plan ranks are 1, 2, 3.
    assert sorted(c['L_M'] for c in cands.values()) == [1.0, 2.0, 3.0]


def test_bad_lm_units_raises():
    p = _make_policy(lm_units='furlongs')
    p._resample(600.0, 'day_start')
    with pytest.raises(ValueError, match='lm_units'):
        p._candidate_edges(p._s0_locs.clone(), 600.0, torch.ones(13, dtype=torch.bool))


def test_get_actions_without_a_trained_network_raises():
    p = _make_policy()
    p.gpn = None
    with pytest.raises(RuntimeError, match='goal proposal network'):
        p.get_actions(None, world_state={'nodes': [], 'edges': [],
                                         'activity': 'idle', 'time': 600})


def test_plan_order_gate_keeps_only_the_earliest_predicate_per_object():
    """A household object visits several destinations per day; scoring them all
    at once makes the helper ping-pong it between them."""
    preds = [(0, 10), (0, 20), (1, 11)]      # object 0 has TWO destinations
    p = _make_policy()
    p.predicates = preds
    p._pid = {q: i for i, q in enumerate(preds)}
    p.clock = {0: (700.0, 5.0, 5), 1: (1100.0, 5.0, 5), 2: (800.0, 5.0, 5)}
    p._s0_sat = torch.tensor([False, False, False])
    p._counts = torch.zeros(3, dtype=torch.long)
    p.gpn = _StubGPN(torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 3))
    p._resample(600.0, 'day_start')

    movable = torch.ones(13, dtype=torch.bool)
    gated = dict(p._candidate_edges(p._s0_locs.clone(), 600.0, movable))
    assert set(gated) == {(0, 10), (1, 11)}, \
        'only object 0\'s earliest destination survives the gate'

    p.pol['plan_order_gate'] = False
    literal = dict(p._candidate_edges(p._s0_locs.clone(), 600.0, movable))
    assert set(literal) == {(0, 10), (0, 20), (1, 11)}


def test_plan_order_gate_does_not_drop_restore_edges():
    p = _make_policy()
    p._resample(600.0, 'day_start')
    cur = p._s0_locs.clone()
    cur[0] = 10
    p._placements = [{'obj': 0, 'dest': 10, 'prev': 100, 't_emit': 600.0,
                      'status': 'active'}]
    p._counts[0] = 1
    for k in p._particles:
        k['tau'].pop(0, None)
        k['plan'] = [s for s in k['plan'] if s[0] != 0]
        k['prefix'] = {q for q, _ in k['plan'][:15]}
    cands = dict(p._candidate_edges(cur, 600.0, torch.ones(13, dtype=torch.bool)))
    assert (0, 100) in cands and cands[(0, 100)]['kind'] == 'restore'
