"""Utilities for converting VirtualHome world state to SLaTe-PRO input format.

Converts the live environment graph (nodes + edges) into the tensor format
expected by SLaTe-PRO: node_features, edges (probability distributions),
activity_features, time_features, and dynamic_edges_mask.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from model.vocab import VH_NAME_TO_ID


def homers_script_path_to_processed_pt(script_path: str) -> str:
    """Map a human ``scripts_{train,test}/*.txt`` path to the matching GT ``.pt``.

    Same transform as ``closed_loop_eval`` uses for offline edges / routines.
    """
    return script_path.replace('/scripts_', '/processed_seqLM_coarse/').replace('.txt', '.pt')


def _slatepro_data_root() -> str:
    """Return the absolute path to external/SLaTe-PRO/data/HOMER+ from project root."""
    # This file is in agents/; project root is one level up
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'external', 'SLaTe-PRO', 'data', 'HOMER+')


def _infer_n_nodes_from_data(data_dir: str) -> int | None:
    """Load a sample .pt file to get the actual graph node count."""
    for split in ('test', 'train'):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        pt_files = sorted(f for f in os.listdir(split_dir) if f.endswith('.pt'))
        if pt_files:
            sample = torch.load(os.path.join(split_dir, pt_files[0]), map_location='cpu',
                                weights_only=True)
            # edges shape: (T, n_nodes, n_nodes)
            return int(sample['edges'].size(1))
    return None


def _load_vh_node_ids_from_routines(
    project_root: str,
    common_data_path: str,
    ignored_node_classes: set,
) -> list[int] | None:
    """Load the sorted VH node IDs used in the HOMER+ training data.

    Reads the first routine JSON from HOMER_PLUS/Household*/routines_train/ and
    returns the sorted list of VH node IDs (excluding ignored classes).  These
    are the exact 80 nodes the SLaTe-PRO model was trained on; a virtual 'home'
    node is appended during world-state construction (see filter_world_state_to_homer_nodes).

    Returns None if the routines directory cannot be found.
    """
    # Infer household name from common_data_path (e.g. householdA → HouseholdA)
    parts = os.path.normpath(common_data_path).split(os.sep)
    household_dir = None
    for p in parts:
        if p.lower().startswith('household'):
            household_dir = p
    if household_dir is None:
        return None
    household_cap = household_dir[0].upper() + household_dir[1:]

    routines_dir = os.path.join(
        project_root, 'external', 'HOMER_PLUS', household_cap, 'routines_train'
    )
    if not os.path.isdir(routines_dir):
        return None

    routine_files = sorted(f for f in os.listdir(routines_dir) if f.endswith('.json'))
    if not routine_files:
        return None

    with open(os.path.join(routines_dir, routine_files[0])) as f:
        routine = json.load(f)

    first_graph = routine['graphs'][0]
    node_ids = sorted(
        n['id'] for n in first_graph['nodes']
        if n['class_name'] not in ignored_node_classes
    )
    return node_ids


def load_common_data(common_data_path: str) -> dict:
    """Load common_data.json and augment with vh_node_ids and n_nodes.

    vh_node_ids comes from the HOMER_PLUS JSON routines (sorted VH IDs of the
    80 real nodes present in the household, excluding ignored classes).
    n_nodes = len(vh_node_ids) + 1 (for virtual 'home' node).
    """
    with open(common_data_path) as f:
        common_data = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ignored = set(common_data.get('ignored_node_classes',
                                   ['wall', 'light', 'ceiling', 'floor', 'curtain', 'maindoor']))
    # Use all non-home node IDs from common_data (matches the 92 nodes the
    # model was trained on).  The old routine-JSON approach only returned 80
    # IDs, missing the 12 "constant" nodes that HOMER_PLUS process_data.py
    # injects from CustomScene2_graph_fromABC.json.
    vh_node_ids = [nid for nid in common_data['node_ids'] if nid != -1]
    common_data['vh_node_ids'] = vh_node_ids
    common_data['n_nodes'] = len(vh_node_ids) + 1  # +1 for virtual 'home'

    return common_data


def filter_world_state_to_homer_nodes(world_state: dict, common_data: dict) -> dict:
    """Filter a full VirtualHome world state to only the HOMER+ tracked nodes.

    In the live simulation the VirtualHome scene has ~417 nodes, but SLaTe-PRO
    was trained on 81 specific HOMER+ nodes (80 real + 1 virtual 'home').
    When ``common_data`` contains ``vh_node_ids`` (loaded automatically by
    ``load_common_data``), this function:

    1. Retains only the 80 real HOMER+ nodes (by VH node ID).
    2. Appends a virtual 'home' node with an ID that sorts *last*, so the local
       index assigned in ``world_state_to_history_initialization`` matches the
       index used in the trained model (home = local index 80).
    3. Adds a home→home INSIDE self-loop (matching the HOMER+ preprocessing).
    4. Removes all edges that reference non-HOMER+ nodes.

    No-op when the world state already has ≤ n_nodes entries (e.g., eval script).
    """
    vh_node_ids = common_data.get('vh_node_ids')
    expected_n_nodes = common_data.get('n_nodes')

    if vh_node_ids is None or expected_n_nodes is None:
        return world_state

    current_nodes = world_state.get('nodes', [])
    if len(current_nodes) <= expected_n_nodes:
        return world_state  # Already correctly sized (eval-script mode)

    # Filter real nodes
    vh_id_set = set(vh_node_ids)
    filtered_nodes = [n for n in current_nodes if n['id'] in vh_id_set]

    # Virtual home node: use ID > all real IDs so it sorts last → local index 80
    home_vh_id = -1
    filtered_nodes.append({
        'id': home_vh_id,
        'class_name': 'home',
        'node_class_idx': 0,   # 'home' is class index 0 in node_classes
        'states': [],
        'name': 'home',
    })

    # Keep edges between HOMER+ nodes, plus HOLDS edges (character→object)
    # whose target is a tracked node.  The character node (ID 240) is NOT in
    # vh_id_set, so a pure both-ends filter would drop all HOLDS edges,
    # breaking held-object carry-forward in get_object_location_matrix.
    filtered_edges = [
        e for e in world_state.get('edges', [])
        if (e['from_id'] in vh_id_set and e['to_id'] in vh_id_set)
        or (e['relation_type'] in ('HOLDS_RH', 'HOLDS_LH') and e['to_id'] in vh_id_set)
    ]
    filtered_edges.append(
        {'from_id': home_vh_id, 'to_id': home_vh_id, 'relation_type': 'INSIDE'}
    )

    return {**world_state, 'nodes': filtered_nodes, 'edges': filtered_edges}

## ToDo: Hardocded to 16 for SLaTe-PRO c_len. Need to make this configurable.
def time_encoder(t, target_len=16):
    """Encode time t (minutes) into sinusoidal features. Pads or truncates to target_len (SLaTe-PRO c_len)."""
    T_freq = [10, 30, 60, 60*3, 60*6, 60*12, 60*24]
    omegas = torch.Tensor([2*np.pi/t_freq for t_freq in T_freq])
    out = torch.cat([torch.sin((om*t).unsqueeze(-1)) for om in omegas] + [torch.cos((om*t).unsqueeze(-1)) for om in omegas], dim=-1)
    if out.size(-1) < target_len:
        pad = torch.zeros(target_len - out.size(-1), device=out.device, dtype=out.dtype)
        out = torch.cat([out, pad], dim=-1)
    elif out.size(-1) > target_len:
        out = out[..., :target_len]
    return out

def _densify(edges: np.ndarray) -> np.ndarray:
    """Compute transitive closure of adjacency matrix.

    Ported from external/SLaTe-PRO/helpers/readers.py:20-28.
    """
    dense = edges.copy()
    for _ in range(edges.shape[-1]):
        new = np.matmul(dense, edges)
        new = new * (dense == 0)
        if (new == 0).all():
            break
        dense += new
    return dense


def _sparsify(edges: np.ndarray) -> np.ndarray:
    """Remove transitive edges, keeping only direct-parent (tree) edges.

    Ported from external/SLaTe-PRO/helpers/readers.py:30-34.
    """
    dense = _densify(edges.copy())
    remove = np.matmul(dense, dense)
    return dense * (remove == 0).astype(int)


def get_object_location_matrix(
    world_state: dict,
    vh_id_to_slate_idx: dict[int, int],
    n_nodes: int,
    common_data: dict | None = None,
    prev_edge_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert VirtualHome edges to SLaTe-PRO edge matrix (tree structure).

    Matches the paper's preprocessing in readers.py read_graphs():
    1. Build full adjacency from all INSIDE/ON edges
    2. Add explicit room→home edges
    3. Sparsify to a tree (each object → direct parent)
    4. Add home→home self-loop
    5. Normalize rows
    6. Carry forward held-object locations from previous state

    Parameters
    ----------
    common_data : If provided, used to identify room nodes (by category) and
                  the home node index.  When None, falls back to heuristic room
                  detection (less reliable).
    prev_edge_matrix : Normalized edge matrix from the previous timestep.
                       When provided, objects currently held by the character
                       (HOLDS_RH/HOLDS_LH) retain their previous location
                       instead of getting an all-zero row.
    """
    edges_np = np.zeros((n_nodes, n_nodes))
    ws_edges = world_state.get('edges', [])

    # Detect objects held by the character (VH: character --HOLDS--> object).
    # HOMER_PLUS never records held state — objects keep their last known
    # location until actually put down.
    held_local_idxs = set()
    for edge in ws_edges:
        if edge['relation_type'] in ('HOLDS_RH', 'HOLDS_LH'):
            held_id = edge['to_id']
            if held_id in vh_id_to_slate_idx:
                held_local_idxs.add(vh_id_to_slate_idx[held_id])

    # Build full adjacency from INSIDE/ON edges (skip self-loops — the paper
    # adds the home→home self-loop AFTER sparsification, line 322 in readers.py)
    for edge in ws_edges:
        if edge['relation_type'] not in ('INSIDE', 'ON'):
            continue
        from_id = edge['from_id']
        to_id = edge['to_id']
        if from_id == to_id:
            continue
        if from_id in vh_id_to_slate_idx and to_id in vh_id_to_slate_idx:
            i = vh_id_to_slate_idx[from_id]
            j = vh_id_to_slate_idx[to_id]
            edges_np[i, j] = 1

    # Identify room nodes and home node so we can add room→home edges
    # (rooms have no INSIDE parent in VH but the paper explicitly adds them)
    home_local_idx = None
    room_local_idxs = []

    if common_data is not None:
        node_categories = common_data.get('node_categories', [])
        node_classes = common_data['node_classes']
        # Build vh_id → category map via class index
        # First-occurrence class_to_idx (already fixed above, but we rebuild
        # locally since this function may be called independently)
        _c2i = {}
        for ci, cn in enumerate(node_classes):
            if cn not in _c2i:
                _c2i[cn] = ci

        node_id_to_class = {}
        for node in world_state.get('nodes', []):
            node_id_to_class[node['id']] = node.get('class_name', '')

        for vh_id, local_i in vh_id_to_slate_idx.items():
            cname = node_id_to_class.get(vh_id, '')
            if cname == 'home':
                home_local_idx = local_i
                continue
            cidx = _c2i.get(cname, 0)
            cat = node_categories[cidx] if cidx < len(node_categories) else ''
            if cat == 'Rooms':
                room_local_idxs.append(local_i)

    # Add room→home edges (matching paper's read_graphs lines 315-316)
    if home_local_idx is not None:
        for r_idx in room_local_idxs:
            edges_np[r_idx, home_local_idx] = 1

    # Sparsify to tree structure (matching paper's read_graphs line 321)
    edges_np = _sparsify(edges_np)

    # Home→home self-loop (matching paper's read_graphs line 322)
    if home_local_idx is not None:
        edges_np[home_local_idx, home_local_idx] = 1

    edge_matrix = torch.from_numpy(edges_np).float()
    row_sums = edge_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    edge_matrix = edge_matrix / row_sums

    # Carry forward held objects from previous state
    if prev_edge_matrix is not None and held_local_idxs:
        for i in held_local_idxs:
            edge_matrix[i, :] = prev_edge_matrix[i, :]

    # print movements
    if prev_edge_matrix is not None:
        num_obj_moved = (edge_matrix - prev_edge_matrix).max(dim=-1).values.sum().item()
        print(f"Number of objects moved: {num_obj_moved}")

    return edge_matrix

def world_state_to_history_initialization(world_state: dict, common_data: dict) -> dict:
    """Convert a single VirtualHome world state to SLaTe-PRO history initialization format.

    The edge matrix is (n_nodes, n_nodes) where n_nodes = number of nodes actually
    present in *world_state* (e.g. 81 for Household A/B, 67 for C).  Node features
    are (n_nodes, n_classes) one-hot over the full vocabulary so the model can
    distinguish node types regardless of how many nodes are in the graph.

    Returns a dict with keys: nodes, edges, activity_features, activity_ids,
    time_features, dynamic_edges_mask, node_id_to_class, vh_id_to_slate_idx,
    sequence_length.
    """
    n_classes    = len(common_data['node_classes'])   # vocabulary size (e.g. 93)
    n_activities = len(common_data['activities'])
    node_classes = common_data['node_classes']

    node_id_to_class = {id:classname for id, classname in zip(common_data['node_ids'], common_data['node_classes'])}
    n_nodes = len(node_id_to_class)   # actual graph size (81 for A/B, 67 for C)
    node_idxs = torch.arange(n_nodes)

    # Local index ordering: sorted by VH ID (matches training data .pt files).
    sorted_vh_ids = common_data['node_ids']
    vh_id_to_slate_idx = {vh_id: local_i for local_i, vh_id in enumerate(sorted_vh_ids)}

    # For nodes carrying 'node_class_idx' (e.g. from eval script's fake world_state),
    # use that raw index directly to handle duplicate class names exactly.
    vh_id_to_raw_class_idx = {}
    for node in world_state.get('nodes', []):
        if 'node_class_idx' in node:
            vh_id_to_raw_class_idx[node['id']] = int(node['node_class_idx'])

    # Node features: one-hot over full vocabulary — shape (1, 1, n_nodes, n_classes)
    node_features_one = F.one_hot(node_idxs, num_classes=n_classes).float()
    node_features_one = node_features_one.unsqueeze(0).unsqueeze(0)

    # Edges: (1, 1, n_nodes, n_nodes) using local indices
    edges = get_object_location_matrix(
        world_state, vh_id_to_slate_idx, n_nodes, common_data, prev_edge_matrix=None
    ).unsqueeze(0).unsqueeze(0)

    assert edges.shape[-2:] == (n_nodes, n_nodes), (
        f"Edge matrix shape {edges.shape} does not match n_nodes={n_nodes}"
    )

    # Activity
    activities = common_data['activities']
    try:
        act_idx = activities.index(world_state['activity'])
    except ValueError:
        # Paper parity: unknown activity maps to Idle/Other. If no such fallback
        # exists, crash rather than silently using index 0 (would mask vocab drift).
        idle_candidates = [a for a in ('idle', 'other', 'Idle', 'Other') if a in activities]
        if not idle_candidates:
            raise ValueError(
                f"activity '{world_state['activity']}' not in common_data activities "
                "and no Idle/Other fallback exists — vocab mismatch.")
        act_idx = activities.index(idle_candidates[0])
        print(f"Warning: activity '{world_state['activity']}' not in common_data activities; "
              f"mapping to '{activities[act_idx]}' (paper parity)")
    activity = F.one_hot(torch.tensor(act_idx), num_classes=n_activities).float().unsqueeze(0).unsqueeze(0)
    time = time_encoder(world_state['time']).unsqueeze(0).unsqueeze(0)

    # Dynamic mask: (n_nodes, n_nodes) — zero rows for static node categories
    static_node_categories = common_data['static_node_categories']
    node_categories = common_data['node_categories']
    dynamic_mask = torch.tensor([1 if cat not in static_node_categories + ['home'] else 0 for cat in node_categories], dtype=torch.float)
    dynamic_mask = dynamic_mask.unsqueeze(-1).repeat(1, n_nodes) * (1 - torch.eye(n_nodes))
    dynamic_mask = dynamic_mask.unsqueeze(0).unsqueeze(0)

    return {
        'nodes': node_idxs.unsqueeze(0).unsqueeze(0),
        'node_features': node_features_one,
        'edges': edges,
        'activity_features': activity,
        'activity_ids': activity.argmax(dim=-1),
        'time_features': time,
        'dynamic_edges_mask': dynamic_mask,
        'node_id_to_class': node_id_to_class,
        'vh_id_to_slate_idx': vh_id_to_slate_idx,
        'sequence_length': 1,
        'times': [world_state['time']],
    }


def update_history_time(world_state: dict, time_in: torch.Tensor, raw_times_in: list[int]) -> tuple[torch.Tensor, list[int]]:
    """Append encoded time features and raw minute for one timestep."""
    raw_time_new = world_state['time']
    time_new = time_encoder(raw_time_new).unsqueeze(0).unsqueeze(0)
    return torch.cat([time_in, time_new], dim=1), raw_times_in + [raw_time_new]

def update_history_edges(world_state: dict, edges_in: torch.Tensor, vh_id_to_slate_idx,
                         common_data: dict | None = None) -> torch.Tensor:
    """Append one row-normalized object-location matrix to the history tensor."""
    prev_edges = edges_in[0, -1]  # previous timestep's normalized edge matrix
    edges_new = get_object_location_matrix(
        world_state, vh_id_to_slate_idx, edges_in.shape[-1], common_data,
        prev_edge_matrix=prev_edges,
    ).unsqueeze(0).unsqueeze(0)
    return torch.cat([edges_in, edges_new], dim=1)

def update_history_activity(world_state: dict, activity_in: torch.Tensor, common_data_activities: list) -> torch.Tensor:
    """Append one-hot activity vector; unknown labels map to Idle/Other or raise."""
    try:
        act_idx = common_data_activities.index(world_state['activity'])
    except ValueError:
        # Paper parity: unknown activity maps to Idle/Other; crash if no such
        # fallback exists rather than silently using index 0 (masks vocab drift).
        idle_candidates = [a for a in ('idle', 'other', 'Idle', 'Other') if a in common_data_activities]
        if not idle_candidates:
            raise ValueError(
                f"activity '{world_state['activity']}' not in common_data activities "
                "and no Idle/Other fallback exists — vocab mismatch.")
        act_idx = common_data_activities.index(idle_candidates[0])
        print(f"Warning: activity '{world_state['activity']}' not in common_data activities; "
              f"mapping to '{common_data_activities[act_idx]}' (paper parity)")
    activity_new = F.one_hot(torch.tensor(act_idx), num_classes=activity_in.shape[-1]).float().unsqueeze(0).unsqueeze(0)
    return torch.cat([activity_in, activity_new], dim=1)

#  from agents.slatepro_utils import print_node_info ; graph_dict = state._graph.to_dict() ; print_node_info(graph_dict, id=1026)

def print_node_info(graph_dict, id=None, name=None, silent=False):
    """Debug helper: print one node's incident edges; return the node dict."""
    node = None
    if id is not None:
        if len([n for n in graph_dict['nodes'] if n['id'] == id]) == 0:
            print(f"Node with id {id} not found in graph")
            return None
        node = [n for n in graph_dict['nodes'] if n['id'] == id][0]
    elif name is not None:
        if len([n for n in graph_dict['nodes'] if n['class_name'] == name]) == 0:
            print(f"Node with name {name} not found in graph")
            return None
        node = [n for n in graph_dict['nodes'] if n['class_name'] == name][0]
    else:
        print("Either id or name must be provided")
        return None
    name = node['class_name']
    id = node['id']
    if not silent:
        print(f"Found node {name} with id {id}")
    if node is not None and not silent:
        print(f"Edges from {name}: ", [f"{e['relation_type']} {[e['to_class'] if 'to_class' in e else e['to_id']]}, " for e in graph_dict['edges'] if e['from_id'] == id])
        print(f"Edges to {name}: ", [f"{e['relation_type']} {[e['from_class'] if 'from_class' in e else e['from_id']]}, " for e in graph_dict['edges'] if e['to_id'] == id])
    return node
