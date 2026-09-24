import re
import json
import os
import random as _random
import sys
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'SLaTe-PRO'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim', 'simulation'))

from adict import adict
from evolving_graph.scripts import Script, parse_script_line, Action

ACTIONS_TO_REMOVE = ['WALK', 'FIND', 'GRAB', 'SIT', 'STANDUP', 'LIE', 'READ', 'EAT', 'DRINK', 'WAKEUP', 'SLEEP', 'PUTON', 'PUTOFF']
ACTIONS_TO_CONVERT_TO_TELEPORT = ['PUTBACK', 'PUTIN', 'DROP', 'POUR']
ACTIONS_TO_CONSERVE = ['OPEN', 'CLOSE', 'SWITCHON', 'SWITCHOFF', 'PLUGIN', 'PLUGOUT', 'SCRUB', 'WIPE', 'RINSE', 'WASH'] ## state-changing actions that don't change the object's location


def convert_dataset(file_name):
    """Read script file and extract script lines, timestamps, activity labels, object movements, and state changes.
    
    Format:
    - Script lines: [ACTION] <object> (id) [index]
    - Timestamps: float value on line immediately after script line
    - Object movements: timestamp: object_name moved (ignore moved_under_sensor)
    - State changes: (s)IDobject_name['old_states']->['new_states']
    """
    
    def parse_states(states_str):
        """Parse state string like '', 'CLOSED', or 'CLOSED', 'CLEAN' into a list."""
        if not states_str:
            return []
        states = []
        # Use regex to find all quoted strings
        for match in re.findall(r"'([^']*)'|([^,']+)", states_str):
            state = match[0] if match[0] else match[1]
            if state.strip():
                states.append(state.strip())
        return states if states else []

    def identify_line_type(line):
        header_regex_check = lambda l: re.match(r'^\w+\s*\(\d+:\d+\s*-\s*\d+:\d+\)$', l)
        ## separate this because sometimes the timestamp is in the same line as these
        if line.strip() == '':
            return None
        elif line.startswith('- '):
            return 'object_movement_removed'
        elif line.startswith('+ '):
            return 'object_movement_added'
        elif line.startswith('(s)'):
            return 'state_change'
        elif line.startswith('###'):
            return 'activity_label'
        elif line.startswith('##'):
            return 'micro_activity_label'
        elif '[' in line:
            return 'script_line'
        elif header_regex_check(line) or header_regex_check(line.replace('- 1day ','')):
            return None
        elif ':' in line and 'moved' in line:
            return None
        else:
            try:
                float(line.strip())
                return 'timestamp'
            except ValueError:
                raise ValueError(
                    f"Cannot determine line type for: {line.strip()!r}")
    
    script = {
        'lines': [],
        'timestamps': [],
        'activity_labels': [],
        'micro_activity_labels': [],
        'state_changes': [],
        'schedule': [],
        }

    # First pass: parse the activity schedule from the header
    header_pattern = re.compile(
        r'^(\w[\w\s]*?)\s*\((\d+):(\d+)\s*-\s*(?:1day\s*-\s*)?(\d+):(\d+)\)')
    with open(file_name) as f:
        for line in f:
            m = header_pattern.match(line.strip())
            if m:
                activity = m.group(1).strip()
                start_min = int(m.group(2)) * 60 + int(m.group(3))
                end_min = int(m.group(4)) * 60 + int(m.group(5))
                if end_min <= start_min:
                    end_min += 24 * 60  # wraps past midnight
                script['schedule'].append({
                    'activity': activity,
                    'start': start_min,
                    'end': end_min,
                })
            elif line.strip().startswith('###'):
                break  # end of header

    with open(file_name) as f:
        current_line_index = -1

        # Read all lines into a list to process sequentially
        file_lines = list(f)

        current_activity = None
        script_line_cache = None
        last_walk_location = None
        object_current_location = {}
        pre_grab_location = {}
        for line in file_lines:
            line = line.strip().replace('<<Graph Changes>>', '').replace('<<Activity Change>>', '')
            line_type = identify_line_type(line)

            if line_type is None:
                continue
        
            if line_type == 'timestamp':
                timestamp = float(line)
                if current_line_index >= 0 and timestamp == script['timestamps'][current_line_index] and script_line_cache is not None:
                    script['lines'][current_line_index].append(str(script_line_cache))
                    script_line_cache = None
                else:
                    script['timestamps'].append(timestamp)
                    assert current_line_index == len(script['timestamps']) - 2, f"Timestamp {timestamp} does not match current line index {current_line_index}"
                    
                    current_line_index = len(script['timestamps']) - 1
                    if script_line_cache is not None:
                        script['lines'].append([str(script_line_cache)])
                    else:
                        script['lines'].append([])
                    script_line_cache = None
                    script['state_changes'].append([])
                    script['activity_labels'].append(current_activity)
                    script['micro_activity_labels'].append(None)
                    assert len(script['lines']) == current_line_index + 1, f"Lines length mismatch"
                    assert len(script['activity_labels']) == current_line_index + 1, f"Activity labels length mismatch"
                    # assert len(script['object_movements']) == current_line_index + 1, f"Object movements length mismatch"
                    assert len(script['state_changes']) == current_line_index + 1, f"State changes length mismatch"
            
            if line_type == 'script_line':
                assert script_line_cache is None, f"Cannot cache 2 script lines"
                script_line_cache = parse_script_line(line, current_line_index + 1, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                if script_line_cache.action._name_ in ACTIONS_TO_REMOVE:
                    if script_line_cache.action._name_ == 'WALK':
                        last_walk_location = script_line_cache.object()
                    elif script_line_cache.action._name_ == 'GRAB':
                        obj_id = script_line_cache.object().instance
                        pre_grab_location[obj_id] = object_current_location.get(obj_id)
                    script_line_cache = None
                elif script_line_cache.action._name_ in ACTIONS_TO_CONVERT_TO_TELEPORT:
                    if script_line_cache.action._name_ == 'DROP':
                        dest = last_walk_location
                    else:
                        dest = script_line_cache.subject()
                    obj_id = script_line_cache.object().instance
                    object_current_location[obj_id] = dest
                    script_line_cache = parse_script_line(f'[TELEPORT] <{script_line_cache.object().name}> ({script_line_cache.object().instance}) <{dest.name}> ({dest.instance})', current_line_index + 1)
                elif script_line_cache.action._name_ == 'PUTOBJBACK':
                    obj = script_line_cache.object()
                    dest = pre_grab_location.get(obj.instance)
                    if dest is not None:
                        object_current_location[obj.instance] = dest
                        script_line_cache = parse_script_line(
                            f'[TELEPORT] <{obj.name}> ({obj.instance}) <{dest.name}> ({dest.instance})',
                            current_line_index + 1)
                    else:
                        script_line_cache = None
                elif script_line_cache.action._name_ in ACTIONS_TO_CONSERVE:
                    pass
                else:
                    print(f"Unknown action: {script_line_cache.action._name_}")
                    exit()
                
            if line_type == 'activity_label':
                current_activity = line.replace('###', '').split('-')[0].strip()
            
            if line_type == 'micro_activity_label':
                script['micro_activity_labels'][current_line_index] = line.replace('##', '').split('until')[0].strip()
            
            if line_type == 'state_change':
                state_match = re.match(r'\(s\)(\d+)(\w+)\[(.*?)\]->\[(.*?)\]', line)
                if not state_match:
                    print(f'[WARN] Could not parse state-change line: {line}')
                if state_match:
                    object_name = state_match.group(2)
                    old_states_str = state_match.group(3).strip()
                    new_states_str = state_match.group(4).strip()
                    
                    old_states = parse_states(old_states_str)
                    new_states = parse_states(new_states_str)
                    script['state_changes'][current_line_index].append((object_name, old_states, new_states))

    fn = file_name.replace('.txt', '.json').replace('scripts_', 'converted_scripts_')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(script, open(fn, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.path):
        if '/scripts_train' in root or '/scripts_test' in root:
            for file in files:
                if file.endswith('.txt'):
                    convert_dataset(os.path.join(root, file))