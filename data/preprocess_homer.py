"""Preprocessing script that dry-runs HOMER+ scripts to extract expected object states.

Loads a HOMER+ script and the corresponding initial scene graph,
executes each action sequentially via ScriptExecutor (no robot actions),
and records expected initial/final locations and states for each action.

Usage:
    python data/preprocess_homer.py \
        --script_file external/HOMER_PLUS/HouseholdB/scripts_train/000.txt \
        --verbose

Output is saved as JSON in data/processed/<Household>/<script_number>.json
"""

import sys
import os
import json
import argparse

# Resolve paths relative to repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'external', 'rail_tasksim'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'external', 'rail_tasksim', 'simulation'))

from routines.GraphReader import GraphReader
from evolving_graph.scripts import Script, ScriptLine, Action
from evolving_graph.environment import EnvironmentGraph, EnvironmentState
from evolving_graph.execution import ScriptExecutor, ExecutionInfo
import evolving_graph.utils as graph_utils

from environment.human_agent import read_program


def get_object_location(graph_dict, object_id):
    """Get all location-relevant edges for an object from a graph dict.

    Tracks INSIDE/ON edges from the object and HOLDS_RH/HOLDS_LH edges to the
    object (indicating the character is holding it).

    Returns:
        list of dicts with keys: relation, target_id, target_name
    """
    if object_id is None:
        return []
    # Build a quick id -> class_name lookup
    id_to_name = {n['id']: n['class_name'] for n in graph_dict['nodes']}
    locations = []
    for edge in graph_dict['edges']:
        if edge['from_id'] == object_id and edge['relation_type'] in ('INSIDE', 'ON'):
            locations.append({
                'relation': edge['relation_type'],
                'target_id': edge['to_id'],
                'target_name': id_to_name.get(edge['to_id']),
            })
        elif edge['to_id'] == object_id and edge['relation_type'] in ('HOLDS_RH', 'HOLDS_LH'):
            locations.append({
                'relation': edge['relation_type'],
                'target_id': edge['from_id'],
                'target_name': id_to_name.get(edge['from_id']),
            })
    return locations


def get_object_states(graph_dict, object_id):
    """Get an object's states from a graph dict."""
    if object_id is None:
        return []
    for node in graph_dict['nodes']:
        if node['id'] == object_id:
            return list(node['states'])
    return []


def preprocess_script(script_file, init_graph_file, output_file=None, verbose=False):
    """Dry-run a HOMER+ script and record expected object locations and states.

    Executes every action within a single persistent EnvironmentState so that
    FIND -> GRAB sequences work correctly (FIND registers the object, GRAB
    expects it to be registered).

    For each action in the script, records:
      - action_type, object_name, object_id, subject_name, subject_id
      - expected_initial_location / expected_final_location  (INSIDE/ON/HOLDS edges)
      - expected_initial_states / expected_final_states
      - timestamp, activity_label, original action string

    Args:
        script_file:     Path to the HOMER+ script (.txt).
        init_graph_file: Path to the initial scene graph (.json).
        output_file:     Where to save the processed JSON.  If None, a default
                         path under data/processed/ is computed.
        verbose:         Print progress information.

    Returns:
        The processed dict (also saved to disk when *output_file* is set).
    """
    # Load initial scene graph
    graph_reader = GraphReader(init_graph_file)
    node_map = graph_reader.node_map

    # Parse the script using the existing reader
    program = read_program(script_file, node_map)

    # Flatten all script lines while remembering their (step_idx, line_idx)
    all_lines = []
    line_step_map = []  # (step_idx, line_idx_within_step) for each flat line
    for step_idx in range(len(program['timestamps'])):
        for line_idx, line in enumerate(program['lines'][step_idx]):
            all_lines.append(line)
            line_step_map.append((step_idx, line_idx))

    # Create execution environment — use a single persistent state so that
    # FIND -> GRAB chains work (FIND registers objects in _script_objects).
    name_equivalence = graph_utils.load_name_equivalence()
    env = EnvironmentGraph(graph_reader.graph_dict.copy())
    state = EnvironmentState(env, name_equivalence, instance_selection=True)
    info = ExecutionInfo()

    # Prepare output structure
    processed_steps = []
    for step_idx in range(len(program['timestamps'])):
        processed_steps.append({
            'timestamp': program['timestamps'][step_idx],
            'actions': [],
        })

    # Execute each action within the persistent state
    num_ok = 0
    num_fail = 0
    for flat_idx, script_line in enumerate(all_lines):
        step_idx, line_idx = line_step_map[flat_idx]
        activity_labels = program['activity_labels'][step_idx]

        # --- Pre-action snapshot ---
        pre_graph = state.to_dict()

        target_obj = script_line.object()
        target_obj_id = target_obj.instance if target_obj else None
        target_obj_name = target_obj.name if target_obj else None

        subject_obj = script_line.subject()
        subject_obj_id = subject_obj.instance if subject_obj else None
        subject_obj_name = subject_obj.name if subject_obj else None

        expected_initial_location = get_object_location(pre_graph, target_obj_id)
        expected_initial_states = get_object_states(pre_graph, target_obj_id)

        # --- Execute the single action within the persistent state ---
        # Build a Script with remaining lines (some executors peek ahead)
        remaining_script = Script(all_lines[flat_idx:])
        info_single = ExecutionInfo()
        next_state = next(
            ScriptExecutor.call_action_method(remaining_script, state, info_single),
            None,
        )

        if next_state is not None:
            state = next_state
            post_graph = state.to_dict()
            success = True
            num_ok += 1
        else:
            post_graph = pre_graph  # graph unchanged on failure
            success = False
            num_fail += 1

        expected_final_location = get_object_location(post_graph, target_obj_id)
        expected_final_states = get_object_states(post_graph, target_obj_id)

        action_record = {
            'action_type': script_line.action.name,
            'object_name': target_obj_name,
            'object_id': target_obj_id,
            'subject_name': subject_obj_name,
            'subject_id': subject_obj_id,
            'expected_initial_location': expected_initial_location,
            'expected_final_location': expected_final_location,
            'expected_initial_states': expected_initial_states,
            'expected_final_states': expected_final_states,
            'action_str': str(script_line),
            'execution_success': success,
            'activity_label': (activity_labels[line_idx]
                               if line_idx < len(activity_labels) else None),
        }
        processed_steps[step_idx]['actions'].append(action_record)

        if verbose:
            loc_changed = expected_initial_location != expected_final_location
            st_changed = expected_initial_states != expected_final_states
            status = 'OK' if success else 'FAIL'
            print(f'  [{status}] Step {step_idx:>3d}.{line_idx}  {script_line}')
            if loc_changed:
                _il = expected_initial_location
                _fl = expected_final_location
                print(f'         loc: {_il} -> {_fl}')
            if st_changed:
                print(f'         st:  {expected_initial_states} -> {expected_final_states}')

    result = {
        'script_file': os.path.abspath(script_file),
        'init_graph_file': os.path.abspath(init_graph_file),
        'num_steps': len(processed_steps),
        'num_actions_ok': num_ok,
        'num_actions_fail': num_fail,
        'steps': processed_steps,
    }

    # Save to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f'\nSaved processed script ({len(processed_steps)} steps, '
                  f'{num_ok} ok / {num_fail} fail) -> {output_file}')

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_path(script_file):
    """Derive a default output path from the script file path."""
    script_basename = os.path.basename(script_file).replace('.txt', '.json')
    # Try to extract the household name from the path
    parts = os.path.normpath(script_file).split(os.sep)
    household = 'unknown'
    for p in parts:
        if p.startswith('Household'):
            household = p
            break
    return os.path.join(_REPO_ROOT, 'data', 'processed', household, script_basename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess HOMER+ scripts for co-adaptation simulation')
    parser.add_argument('--script_file', type=str, required=True,
                        help='Path to the HOMER+ script (.txt)')
    parser.add_argument('--init_graph_file', type=str,
                        default='data/init_graph_homer_plus.json',
                        help='Path to the initial scene graph (.json)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output path for processed JSON (auto-derived if omitted)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = _default_output_path(args.script_file)

    preprocess_script(
        args.script_file,
        args.init_graph_file,
        args.output_file,
        args.verbose,
    )
