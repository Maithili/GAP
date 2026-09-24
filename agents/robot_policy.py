import os as _os
import sys
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  '..', 'external', 'rail_tasksim'))
from evolving_graph.scripts import parse_script_line


class RobotMoveAction:
    """A high-level MOVE action: teleport *object* to *destination*.

    The world translates this into the atomic sequence
    FIND(object) -> GRAB(object) -> WALK(destination) -> PUTBACK(object, destination),
    all executed instantly in a single timestep.
    """

    def __init__(self, object_name: str, object_id: int,
                 dest_name: str, dest_id: int):
        self.object_name = object_name
        self.object_id = object_id
        self.dest_name = dest_name
        self.dest_id = dest_id

    def __str__(self):
        return (f'[MOVE] <{self.object_name}> ({self.object_id}) '
                f'-> <{self.dest_name}> ({self.dest_id})')

    def script_action(self):
        """Return a parsed ``[TELEPORT]`` script line for ``World.execute_action``."""
        action_string = f'[TELEPORT] <{self.object_name}> ({self.object_id}) <{self.dest_name}> ({self.dest_id})'
        script_line = parse_script_line(action_string, 0, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
        return script_line

    def __repr__(self):
        return self.__str__()


class RobotPolicyBase:
    """Abstract base for robot policies.

    ``get_actions`` must return a list of :class:`RobotMoveAction` or an empty list.
    Optional kwargs (e.g. world_state, time_step_minutes) may be passed by the
    simulation; policies that need them can use them, others ignore them.
    """

    def __init__(self, config: dict):
        self.config = config

    def get_actions(self, node_map, *,
                   world_state=None) -> list[RobotMoveAction]:
        """Return robot actions for this tick (subclasses implement)."""
        raise NotImplementedError

    def reset(self):
        """Clear per-day/per-episode state (no-op in base)."""
        pass


class RobotPolicyNoop(RobotPolicyBase):
    """No-op robot policy: the robot never acts.

    Use this as a baseline (no robot interference) to isolate human behaviour
    and to compare co-adaptation metrics against a no-robot control condition.
    Also useful for reproducing SLaTe-PRO paper evaluations where the robot
    is passive and only the human's routine is observed.
    """

    def get_actions(self, node_map, *,
                   world_state=None) -> list[RobotMoveAction]:
        return []