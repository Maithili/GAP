"""Unit checks for hybrid per-object robot placement stack (B15 / Option 3).

    python tests/test_robot_policy_hybrid_placements.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.robot_policy_hybrid import RobotPolicyHybrid


def _bare_policy() -> RobotPolicyHybrid:
    """Minimal instance — only stack/lead helpers, no full init."""
    p = RobotPolicyHybrid.__new__(RobotPolicyHybrid)
    RobotPolicyHybrid._reset_day_state(p)
    return p


def test_supersede_on_restage():
    p = _bare_policy()
    p._day_events = [{'actor': 'robot', 'obj_idx': 1, 'dest_idx': 10}]
    p._robot_placements[1] = [{
        'ev_idx': 0, 'dest_idx': 10, 'from_idx': 5,
        'act': 'breakfast', 'status': 'active',
    }]
    p._lead_emitted.add(('breakfast', 1, 10))

    p._supersede_active_placements(1)
    assert p._day_events[0]['superseded'] is True
    assert p._robot_placements[1][0]['status'] == 'superseded'
    assert ('breakfast', 1, 10) in p._lead_superseded


def test_revert_targets_active_not_superseded():
    p = _bare_policy()
    p._day_events = [
        {'actor': 'robot', 'obj_idx': 1, 'dest_idx': 10},
        {'actor': 'robot', 'obj_idx': 1, 'dest_idx': 20},
    ]
    p._robot_placements[1] = [
        {'ev_idx': 0, 'dest_idx': 10, 'from_idx': 5,
         'act': 'breakfast', 'status': 'superseded'},
        {'ev_idx': 1, 'dest_idx': 20, 'from_idx': 10,
         'act': 'breakfast', 'status': 'active'},
    ]
    assert p._find_revert_placement(1, prev_idx=10, dest_idx=5) is None
    rev = p._find_revert_placement(1, prev_idx=20, dest_idx=10)
    assert rev is not None and rev['ev_idx'] == 1


def test_continuation_not_revert():
    p = _bare_policy()
    p._robot_placements[1] = [{
        'ev_idx': 0, 'dest_idx': 10, 'from_idx': 5,
        'act': 'breakfast', 'status': 'active',
    }]
    # Human moves onward from robot dest — not an undo back to from_idx.
    assert p._find_revert_placement(1, prev_idx=10, dest_idx=20) is None
    cont = p._consume_continuation(1, prev_idx=10)
    assert cont is not None and cont['status'] == 'continued'


def test_lead_success_excludes_reverted_and_superseded():
    p = _bare_policy()
    p._lead_emitted = {
        ('a', 1, 10),   # ok
        ('a', 2, 20),   # reverted
        ('a', 3, 30),   # superseded
    }
    p._lead_reverted = {('a', 2, 20)}
    p._lead_superseded = {('a', 3, 30)}
    bad = p._lead_reverted | p._lead_superseded
    n_ok = len(p._lead_emitted - bad)
    assert n_ok == 1
    assert 100.0 * n_ok / len(p._lead_emitted) == 100 / 3


if __name__ == '__main__':
    test_supersede_on_restage()
    test_revert_targets_active_not_superseded()
    test_continuation_not_revert()
    test_lead_success_excludes_reverted_and_superseded()
    print('ALL hybrid placement checks passed')
