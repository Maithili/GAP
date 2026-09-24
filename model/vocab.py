import json
from typing import Dict
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)


NODES_LIST = vocab['nodes']
NODE_CLASSES = [n['class_name'] for n in vocab['nodes']]
N_NODE_CLASSES: int = len(NODE_CLASSES)
NODES_DYNAMIC_STATUS = [not n['static'] for n in vocab['nodes']]
VALID_LOCATIONS = [0 if n['class_name'] in ['character', 'home'] else 1 for n in vocab['nodes']]

ACTIVITIES = vocab['activities']
N_ACTIVITIES: int = len(ACTIVITIES)
ACTIVITY_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIVITIES)}

VH_ID_TO_GRAPH_IDX = {n['id']: i for i, n in enumerate(vocab['nodes'])}
VH_ID_TO_NAME = {n['id']: n['class_name'] for n in vocab['nodes']}
VH_NAME_TO_ID = {n['class_name']: n['id'] for n in vocab['nodes']}
SORTED_VH_IDS = [n['id'] for n in vocab['nodes']]
