import copy
import os as _os
import sys
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  '..', 'external', 'rail_tasksim'))
from adict import adict
import numpy as np
import json

from routines.GraphReader import GraphReader, init_graph_file, scene_num
from routines.ProgramExecutor import class_from_id
from dataset_utils import execute_script_utils as utils
from routines.object_locations import object_locations
from model.vocab import NODES_LIST
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph, State
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils


def filter_nodes(graph_dict: dict) -> dict:
    """Keep only nodes/edges present in ``NODES_LIST`` (HOMER+ vocabulary)."""
    nodes_list = NODES_LIST
    node_ids_to_keep = [n['id'] for n in nodes_list]
    graph_dict['nodes'] = [n for n in graph_dict['nodes'] if n['id'] in node_ids_to_keep]
    graph_dict['edges'] = [e for e in graph_dict['edges'] if e['from_id'] in node_ids_to_keep and e['to_id'] in node_ids_to_keep]
    return graph_dict


def validate_world_state(state: dict) -> None:
    """Assert that *state* has the required keys and correct types.
    Required schema:
      ``nodes``    – list of node dicts
      ``edges``    – list of edge dicts
      ``activity`` – str  (human activity label)
      ``time``     – int | float  (simulation timestep)
    """
    for key in ('nodes', 'edges', 'activity', 'time'):
        if key not in state:
            raise KeyError(f"world_state is missing required key '{key}'")
    if not isinstance(state['nodes'], list):
        raise TypeError(f"world_state['nodes'] must be a list, got {type(state['nodes']).__name__}")
    if not isinstance(state['edges'], list):
        raise TypeError(f"world_state['edges'] must be a list, got {type(state['edges']).__name__}")
    if not isinstance(state['activity'], str):
        raise TypeError(f"world_state['activity'] must be a str, got {type(state['activity']).__name__}")
    if not isinstance(state['time'], (int, float)):
        raise TypeError(f"world_state['time'] must be numeric, got {type(state['time']).__name__}")


def print_graph_difference(g1, g2):
    """Print node state and INSIDE/ON/HOLDS edge diffs between two graph dicts."""
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]
    ignore_for_edges = ['wall']

    for n in nodes_removed:
        print ('Removed node : ',n)
    for n in nodes_added:
        print ('Added node   : ',n)
    for n1 in g1['nodes']:
        n2 = [n2 for n2 in g2['nodes'] if n2['id']==n1['id']]
        if len(n2) > 0:
            assert len(n2) == 1
            n2 = n2[0]
            n1states = set(n1['states'])
            n2states = set(n2['states'])
            if n1states != n2states:
                print ('     State {}({}) : {} -> {}'.format( n1['class_name'], n1['id'], n1states.difference(n2states), n2states.difference(n1states)))

    remaining_objects = []
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print ('     - ',c1,e['relation_type'],c2)
            remaining_objects.append(e['from_id'])
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print ('     + ',c1,e['relation_type'],c2)
            if e['from_id'] in remaining_objects:
                remaining_objects.remove(e['from_id'])

def sparsify_graph_dict(graph_dict: dict) -> dict:
    """Remove redundant parent edges so each object has a single INSIDE/ON parent."""
    parents = {n['id']: [] for n in graph_dict['nodes']}
    edges_to_remove = []
    for edge in graph_dict['edges']:
        if edge['relation_type'] in ['INSIDE','ON']:
            parents[edge['from_id']].append(edge['to_id'])
    for node in graph_dict['nodes']:
        if len(parents[node['id']]) > 1:
            parents_to_remove = []
            for parent in parents[node['id']]:
                parents_of_parent = parents[parent]
                parents_to_remove += [id for id in parents[node['id']] if id in parents_of_parent]
            parents_to_remove = list(set(parents_to_remove))
            assert len(parents[node['id']]) - len(parents_to_remove) == 1, f"Node {node['id']} has {len(parents[node['id']])} parents, but {len(parents_to_remove)} parents to remove"
            edges_to_remove += [(node['id'], parent) for parent in parents_to_remove]

    graph_dict['edges'] = [e for e in graph_dict['edges'] if (e['from_id'], e['to_id']) not in edges_to_remove and e['relation_type'] in ['INSIDE','ON']]

    return graph_dict
                
class World:
    """VirtualHome graph executor for the closed-loop benchmark.

    Wraps ``EnvironmentGraph`` + ``ScriptExecutor``. Robot teleports use
    ``execute_robot_move``; human/robot script lines use ``execute_action``.
    """

    def __init__(self, config: dict, init_graph_dict: dict = None):
        """Build the scene from *init_graph_dict* or ``config['simulation']['init_graph_file']``."""
        if init_graph_dict is None:
            init_graph_file = config['simulation']['init_graph_file']
            init_graph_dict = json.load(open(init_graph_file))
            init_graph_dict = filter_nodes(init_graph_dict)
        # Backfill any NODES_LIST nodes missing from init_graph_dict so that
        # every node referenced by scripts or robot policies exists in the
        # EnvironmentGraph.  Edges for the added nodes come from the base
        # init graph (config's init_graph_file).
        existing_ids = {n['id'] for n in init_graph_dict['nodes']}
        missing_ids = {n['id'] for n in NODES_LIST} - existing_ids
        if missing_ids:
            base_graph_path = config['simulation']['init_graph_file']
            base_graph = json.load(open(base_graph_path))
            base_graph = filter_nodes(base_graph)
            base_node_by_id = {n['id']: n for n in base_graph['nodes']}
            for mid in missing_ids:
                node = base_node_by_id.get(mid)
                if node is None:
                    node = next((n for n in NODES_LIST if n['id'] == mid), None)
                if node is not None:
                    init_graph_dict['nodes'].append(node)
            all_ids = {n['id'] for n in init_graph_dict['nodes']}
            existing_edges = {(e['from_id'], e['relation_type'], e['to_id'])
                              for e in init_graph_dict['edges']}
            for edge in base_graph['edges']:
                key = (edge['from_id'], edge['relation_type'], edge['to_id'])
                if key not in existing_edges and \
                   edge['from_id'] in all_ids and edge['to_id'] in all_ids:
                    init_graph_dict['edges'].append(edge)

        ## If character is not in the graph, add it.
        character_node = {
            "properties": [],
            "id": 240,
            "states": [],
            "category": "Characters",
            "class_name": "character",
            "static": True
        }
        character_edge = {
            "from_id": 240,
            "to_id": 162,
            "relation_type": "INSIDE"
        }
        if 'character' not in [n['class_name'] for n in init_graph_dict['nodes']]:
            init_graph_dict['nodes'].append(character_node)
            init_graph_dict['edges'].append(character_edge)
        
        init_graph_dict = sparsify_graph_dict(init_graph_dict)

        self.verbose = config['simulation']['verbose']
        self.init_graph = GraphReader(init_graph_dict)
        self.current_time = 0
        self.name_equivalence = utils.load_name_equivalence()
        self.env = EnvironmentGraph(self.init_graph.graph_dict.copy())
        self.executor = ScriptExecutor(self.env, self.name_equivalence)
        self.node_map = self.init_graph.node_map
        self.human_activity = 'idle'

    def add_script_object(self, script_object):
        """Register a parsed script object with the executor (needed before GRAB/PUT)."""
        if script_object is None: return
        self.executor.state._script_objects[(script_object.name, script_object.instance)] = script_object.instance

    def execute_action(self, action):
        """Run one parsed script action. Returns ``(state, graph_list)`` or ``(None, [])`` on recognized physics refusal; raises on unrecognized failure."""
        if action is None:
            return None, []
        self.add_script_object(action.object())
        self.add_script_object(action.subject())
        # try:
        success, state, graph_list = self.executor.execute(Script([action]), w_graph_list=True)
        # except Exception as e:
        #     if self.verbose:
        #         print(f'Action execution raised: {e}')
        #         print(f'Action: {action}')
        #     return None, []
        if not success:
            error_str = self.executor.info.get_error_string()
            # Known, code-recognized executor refusals (physics constraints):
            # the action legitimately cannot be performed, so report failure
            # for the caller to handle. Any UNrecognized error is an unexpected
            # bug → crash (no silent fallback), regardless of verbose. The
            # verbose flag controls only the diagnostic print, never control flow.
            recognized = (
                'Character is holding' in error_str
                or 'would create cycle' in error_str
                or 'inside other closed thing' in error_str
            )
            if self.verbose:
                if 'Character is holding' in error_str:
                    object = error_str[error_str.index('<')+1:error_str.index('>')]
                    if object in object_locations:
                        print(f'{object} is holding {object_locations[object]}')
                elif 'inside other closed thing' in error_str:
                    object = error_str[error_str.index('<')+1:error_str.index('>')]
                    if object in object_locations:
                        print(f'{object} is inside {object_locations[object]}')
            if recognized:
                return None, []
            raise Exception(f'Action failed: {action} Error: {error_str}')
        if self.verbose:
            if len(graph_list) > 1:
                print(action)
                print_graph_difference(graph_list[0], graph_list[1])
        # Update world env from the new state (executor keeps same graph reference; state holds deltas)
        self.env = EnvironmentGraph(state.to_dict())
        return state, graph_list

    def set_human_activity(self, activity: str):
        """Set the activity label exposed in ``get_state()['activity']``."""
        self.human_activity = activity

    def execute_robot_move(self, move_action):
        """Execute a high-level MOVE action as an instant teleportation.

        Translates the move into FIND -> GRAB -> WALK -> PUTBACK and runs
        them all within a single persistent EnvironmentState so the sequence
        succeeds atomically.  Only one ``current_time`` increment occurs.

        Args:
            move_action: a RobotMoveAction with object_name/id and dest_name/id.

        Returns:
            (success: bool, graph_list: list) — same style as execute_action.
        """
        from evolving_graph.environment import EnvironmentState
        from evolving_graph.execution import ScriptExecutor as _SE, ExecutionInfo

        obj = move_action.object_name
        oid = move_action.object_id
        dst = move_action.dest_name
        did = move_action.dest_id

        # Build the four sub-action strings and parse them
        sub_action_strs = [
            f'[FIND] <{obj}> ({oid})',
            f'[GRAB] <{obj}> ({oid})',
            f'[WALK] <{dst}> ({did})',
            f'[PUTBACK] <{obj}> ({oid}) <{dst}> ({did})',
        ]
        sub_actions = []
        for idx, s in enumerate(sub_action_strs, start=1):
            sub_actions.append(
                parse_script_line(s, idx,
                                  custom_patt_params=r'\<(.+?)\>\s*\((.+?)\)')
            )

        # Execute all sub-actions within one persistent state
        state = EnvironmentState(self.env, self.name_equivalence,
                                 instance_selection=True)
        pre_graph = state.to_dict()

        for i, action in enumerate(sub_actions):
            info = ExecutionInfo()
            remaining = Script(sub_actions[i:])
            next_state = next(
                _SE.call_action_method(remaining, state, info), None
            )
            if next_state is None:
                # if self.verbose:
                print(f'  Robot MOVE sub-action failed: {action}')
                print(f'  Error: {info.get_error_string()}')
                return False, []
            state = next_state

        # All sub-actions succeeded — commit the final state
        post_graph = state.to_dict()
        self.env = EnvironmentGraph(post_graph)
        self.executor = ScriptExecutor(self.env, self.name_equivalence)

        if self.verbose:
            print_graph_difference(pre_graph, post_graph)
        # Always prints (even when not verbose) — side-effectful debug line
        print(f'Robot MOVE action completed: {move_action}; graph changes induced: {print_graph_difference(pre_graph, post_graph)}')

        return True, [pre_graph, post_graph]

    def get_state(self):
        """Return the live graph dict plus ``activity`` and ``time`` (validated)."""
        state = self.env.to_dict()
        state['activity'] = self.human_activity
        state['time'] = self.current_time
        validate_world_state(state)
        return state

    def get_node_ids(self):
        """Return all node ids in the current environment graph."""
        return self.env.get_node_ids()

    def reset(self, init_graph_dict=None):
        """Reset the world to its initial state.

        If *init_graph_dict* is provided, it replaces the canonical init graph
        (useful for per-file resets where each routine starts from its own
        recorded initial state).  Otherwise the original init graph is reused.
        """
        if init_graph_dict is not None:
            init_graph_dict = filter_nodes(init_graph_dict)
            init_graph_dict = sparsify_graph_dict(init_graph_dict)
            self.init_graph = GraphReader(init_graph_dict)
        self.env = EnvironmentGraph(copy.deepcopy(self.init_graph.graph_dict))
        self.executor = ScriptExecutor(self.env, self.name_equivalence)
        self.current_time = 0
