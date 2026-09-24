import os as _os
import random as _random
import sys
import json
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  '..', 'external', 'rail_tasksim'))
from adict import adict
from evolving_graph.scripts import Script, parse_script_line

from scripts.convert_dataset import convert_dataset

def read_program(file_name, node_map):
    """Load (or convert+load) the processed JSON companion to a script ``.txt`` file."""
    converted_file_name = file_name.replace('.txt', '.json').replace('scripts_', 'converted_scripts_')
    if not _os.path.exists(converted_file_name):
        convert_dataset(file_name)
    try:
        json.load(open(converted_file_name, 'r'))
    except Exception as e:
        print(f"Error loading converted file {converted_file_name}: {e}...reconverting")
        convert_dataset(file_name)
    return json.load(open(converted_file_name, 'r'))


class HumanAgent:
    """Script-following human for closed-loop benchmark accommodation metrics.

    Each TELEPORT line is tagged ``skipped`` / ``corrected`` / ``done_as_is`` /
    etc. by comparing the live graph (post-robot) against pending robot actions.
    Non-TELEPORT lines are recorded as ``no_movement``.

    TL;DR: robot acts first each tick; human then runs script with accommodation.
    """

    def __init__(self, config: dict, node_map, script_file: str, accommodate_for_robot: bool = True):
        self.accommodate_for_robot = accommodate_for_robot
        self.config = config
        self.verbose = config['simulation']['verbose']

        self.script_file = script_file
        self.script = adict(read_program(self.script_file, node_map))
        self.next_action_idx = 0
        self.node_map = node_map
        self.activity_schedule = self.script.get('schedule', [])

        # Per-episode record of (timestep, action_str, outcome) tuples.
        # outcome ∈ {done_as_is, corrected, skipped}
        self.action_outcomes: list[dict] = []
        # Filled per-iteration by get_actions() so _execute_current_action can
        # tag each action_outcomes entry with the matching tensor-eval category.
        self._per_obj_tensor_category: dict = {}
        self._vh_id_to_slate_idx: dict = {}
        self.action_results = {
            'skipped': 0,      # robot matched planned (obj, dest); human did nothing
            'corrected': 0,    # robot moved same object to wrong dest; human fixed
            'done_as_is': 0,   # human executed the script move (robot did not help)
            'no_movement': 0,  # non-TELEPORT script line (not scored in human F1)
            'no_need_to_move': 0,  # object already at dest, no robot match
            'skipped_repeated': 0,    # duplicate script TELEPORT after first robot match
            'corrected_repeated': 0,
            'accepted': 0,    # LLM-human kept robot's different placement
            'corrected_despite_correct': 0,  # LLM-human moved away though robot was right
            'done_early': 0,  # script line satisfied by earlier redirect (excluded from F1)
            'redirected': 0,  # LLM-human moved robot misplacement to planned dest
        }

        # --- Vigilance configuration ---
        # Vigilance: human observes robot actions; if observed action does not match
        # any planned human action within window, human immediately corrects it.
        human_cfg = config['human']
        vig_cfg = human_cfg.get('curiosity_vigilance', {})
        self.vigilance_enabled = vig_cfg.get('enabled', False)
        self.vigilance_probability = vig_cfg.get('probability', 1.0)
        self.vigilance_window_minutes = vig_cfg.get('correction_window_hours', 6.0) * 60

        # Per-day vigilance counters
        self.vigilance_observations = 0
        self.vigilance_corrections = 0

        self.current_activity = 'idle'

        self.script_line_treatments = []

    # ------------------------------------------------------------------
    # Activity schedule lookup
    # ------------------------------------------------------------------

    def _activity_from_schedule(self, time_minutes: float) -> str:
        """Look up the activity at *time_minutes* from the parsed schedule.

        Returns the activity name if *time_minutes* falls within any
        [start, end) window, otherwise ``'idle'``.
        """
        for entry in self.activity_schedule:
            if entry['start'] <= time_minutes < entry['end']:
                return entry['activity']
        return 'idle'

    # ------------------------------------------------------------------
    # Core query — called once per simulation timestep
    # ------------------------------------------------------------------

    def get_actions(self, current_time: int, world_state: dict,
                    *, remember_actions: bool = True,
                    robot_actions: list[dict] = None,
                    per_obj_tensor_category: dict | None = None,
                    vh_id_to_slate_idx: dict | None = None):
        """Return script actions to execute this tick (may be empty).

        Matches pending ``robot_actions`` against TELEPORT lines to tag
        skipped/corrected/etc. Non-TELEPORT lines yield ``no_movement``.
        """
        # Stash for _execute_current_action to read when tagging action_outcomes.
        self._per_obj_tensor_category = per_obj_tensor_category or {}
        self._vh_id_to_slate_idx = vh_id_to_slate_idx or {}
        if robot_actions is None:
            robot_actions = []

        # Is there an action scheduled at or before this timestep?
        has_action = (self.next_action_idx < len(self.script.timestamps)
                      and current_time >= self.script.timestamps[self.next_action_idx])
        if not has_action:
            # Idle: label the timestep from the activity schedule and do nothing.
            self.current_activity = self._activity_from_schedule(current_time)
            return []

        # Execute every action scheduled at/before this timestep (with
        # accommodation). pending_teleports tracks teleports emitted so far this
        # timestep so the already-at-dest check uses the pending location instead
        # of the stale world_state snapshot.
        actions = []
        pending_teleports: dict[int, int] = {}
        while current_time >= self.script.timestamps[self.next_action_idx]:
            group = self._execute_current_action(current_time, world_state, remember_actions, robot_actions, pending_teleports)
            if group is not None:
                actions.extend(group)
        return actions

    # ------------------------------------------------------------------
    # Action execution (step 3 of the state machine)
    # ------------------------------------------------------------------

    def _execute_current_action(self, current_time: int,
                                world_state: dict | None, remember_actions: bool = True, 
                                robot_actions: list[dict] = None,
                                pending_teleports: dict[int, int] | None = None):
        """Run accommodation logic on the current action and advance the index.

        Records **one outcome per object movement** so that metrics reflect
        per-object granularity rather than per-action-group.
        """
        if pending_teleports is None:
            pending_teleports = {}
        self.current_activity = self._activity_from_schedule(current_time)
        actions = self.script.lines[self.next_action_idx]
        if not isinstance(actions, list):
            actions = [actions]

        script_actions = []
        robot_action_locs = [(a['obj_idx'],a['loc_idx']) for a in robot_actions]
        robot_action_objs = [a['obj_idx'] for a in robot_actions]
        for action_str in actions:
            action_script = parse_script_line(action_str, 0)
            outcome = 'done_as_is'
            if 'TELEPORT' not in action_script.action._name_:
                outcome = 'no_movement'
                self.script_line_treatments.append((action_script, 'no_movement'))
            if action_script.action._name_ == 'TELEPORT':
                # Only TELEPORT (object relocation) actions are accommodated/scored;
                # HOMER+ routines in this benchmark are relocation-only, so other
                # action types are passed through without an outcome.
                obj_id  = action_script.object().instance
                dest_id = action_script.subject().instance

                if obj_id in pending_teleports:
                    # A prior teleport this timestep already moved this object;
                    # world_state is stale — use the pending destination instead.
                    already_at_dest = (pending_teleports[obj_id] == dest_id)
                else:
                    already_at_dest = world_state is not None and any(
                        e['from_id'] == obj_id and e['to_id'] == dest_id
                        and e['relation_type'] in ('INSIDE', 'ON')
                        for e in world_state.get('edges', [])
                    )
                if already_at_dest:
                    # Object is already at the script's destination — human has
                    # nothing to do. Default outcome is 'no_need_to_move'; if a
                    # matching robot prediction exists we mark it 'skipped'.
                    outcome = 'no_need_to_move'
                    treatment = 'no_need_to_move'
                    if len(robot_actions) > 0 and (obj_id, dest_id) in robot_action_locs:
                        robot_outcome_idx = robot_action_locs.index((obj_id, dest_id))
                        print(f"Robot action predicted an idempotent move........{obj_id} to {dest_id} {robot_actions[robot_outcome_idx]['human_action_outcome']}")
                        if robot_actions[robot_outcome_idx]["human_action_outcome"] is None:
                            robot_actions[robot_outcome_idx]["human_action_outcome"] = 'skipped'
                            robot_actions[robot_outcome_idx]["eval_time"] = current_time
                            outcome = 'skipped'
                            treatment = 'skipped'
                        else:
                            outcome = 'skipped_repeated'
                            treatment = 'skipped_repeated'
                    self.script_line_treatments.append((action_script, treatment))
                else:
                    script_actions.append(action_script)
                    pending_teleports[obj_id] = dest_id
                    outcome = 'done_as_is'
                    if len(robot_actions) > 0:
                        ## This condition only really happens if the robot actions are suppressed
                        if (action_script.object().instance, action_script.subject().instance) in robot_action_locs:
                            robot_outcome_idx = robot_action_locs.index((action_script.object().instance, action_script.subject().instance))
                            if robot_actions[robot_outcome_idx]["human_action_outcome"] is None:
                                robot_actions[robot_outcome_idx]["human_action_outcome"] = 'skipped'
                                robot_actions[robot_outcome_idx]["eval_time"] = current_time
                                self.script_line_treatments.append((action_script, 'skipped'))
                                outcome = 'skipped'
                            else:
                                self.script_line_treatments.append((action_script, 'skipped_repeated'))
                                outcome = 'skipped_repeated'
                            if self.accommodate_for_robot:
                                script_actions.pop()
                        elif action_script.object().instance in robot_action_objs:
                            robot_outcome_idx = robot_action_objs.index(action_script.object().instance)
                            if robot_actions[robot_outcome_idx]["human_action_outcome"] is None:
                                robot_actions[robot_outcome_idx]["human_action_outcome"] = 'corrected'
                                robot_actions[robot_outcome_idx]["eval_time"] = current_time
                                self.script_line_treatments.append((action_script, 'corrected'))
                                outcome = 'corrected'
                            else:
                                self.script_line_treatments.append((action_script, 'corrected_repeated'))
                                outcome = 'corrected_repeated'
                if remember_actions:
                    obj_name = action_script.object().name
                    location_name = action_script.subject().name
                    is_teleport = (action_script.action._name_ == 'TELEPORT')
                    teleport_obj_id = (
                        int(action_script.object().instance) if is_teleport else None
                    )
                    slate_idx = (
                        self._vh_id_to_slate_idx.get(teleport_obj_id)
                        if teleport_obj_id is not None else None
                    )
                    if slate_idx is not None:
                        cats = self._per_obj_tensor_category.get(int(slate_idx)) or {'1step': 'no_event', 'anyFuture': 'no_event'}
                        cat_1s = cats['1step']
                        cat_af = cats['anyFuture']
                    else:
                        cat_1s = cat_af = 'unknown'
                    self.action_outcomes.append({
                        'timestep': current_time,
                        'action': str(action_script),
                        'object': obj_name,
                        'object_id': teleport_obj_id,
                        'slate_obj_idx': int(slate_idx) if slate_idx is not None else None,
                        'location': location_name,
                        'activity': self.current_activity,
                        'outcome': outcome,
                        'tensor_eval_category_1step': cat_1s,
                        'tensor_eval_category_anyFuture': cat_af,
                    })
                    self.action_results[outcome] += 1

        self.next_action_idx += 1

        return script_actions
    

    # ------------------------------------------------------------------
    # Accommodation helpers
    # ------------------------------------------------------------------

    def get_human_activity(self) -> str:
        """Activity label for the current or most recent script timestep."""
        return self.current_activity

    # ------------------------------------------------------------------
    # Vigilance helpers
    # ------------------------------------------------------------------

    def roll_vigilance(self) -> bool:
        """Sample whether legacy vigilance observes this robot action (Bernoulli)."""
        if not self.vigilance_enabled:
            return False
        return _random.random() < self.vigilance_probability

    def has_matching_future_action(self, current_time: int, obj_id: int, dest_id: int) -> bool:
        """True if a future TELEPORT matches (obj, dest); False if a conflicting move exists first."""
        window_end = current_time + self.vigilance_window_minutes
        for idx in range(self.next_action_idx, len(self.script.timestamps)):
            t = self.script.timestamps[idx]
            if t > window_end:
                break
            for action_str in self.script.lines[idx]:
                action = parse_script_line(action_str, 0)
                if (action.action._name_ == 'TELEPORT'):
                    if (action.object().instance == obj_id):
                        if (action.subject().instance == dest_id):
                            return True
                        else:
                            return False
        return False

    def review_robot_action(self, robot_action, world_state: dict, current_time: int,
                             day_history: list[dict] | None = None) -> tuple[str, dict | None]:
        """Decide whether to undo or allow a just-executed robot action.

        Returns ``(decision, record)`` where ``decision`` is one of ``'undo'``
        or ``'allow'``. ``record`` is an optional LLM-decision dict that
        ``closed_loop_eval`` will append to ``per_day[d]['llm_decisions']``
        when present.

        Base implementation: emulate the legacy vigilance roll + window check.
        Subclasses override to drive the decision differently (e.g. LLM).
        """
        if not self.roll_vigilance():
            return 'allow', None
        # Vigilance observed; check if script has a matching action soon.
        obj_id = getattr(robot_action, 'object_id', None)
        dest_id = getattr(robot_action, 'dest_id', None)
        self.vigilance_observations += 1
        if obj_id is None or dest_id is None:
            return 'allow', None
        if self.has_matching_future_action(current_time, obj_id, dest_id):
            return 'allow', None
        return 'undo', None

