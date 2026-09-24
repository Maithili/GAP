"""Pytest configuration for the tests/ directory.

Puts the project root and the vendored dependencies on ``sys.path`` so the unit
tests import the package modules (``agents.*``, ``model.*``, ...) without each
test having to insert paths itself.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _p in (
    _ROOT,
    os.path.join(_ROOT, 'external', 'rail_tasksim'),
    os.path.join(_ROOT, 'external', 'rail_tasksim', 'simulation'),  # evolving_graph lives here
    os.path.join(_ROOT, 'external', 'SLaTe-PRO'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
