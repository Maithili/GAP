"""Plain-LLM baseline robot policy (no bells or whistles).

The LLM sees the raw observation history — the past N warmup days of object
relocations, then the incomplete current day — and directly predicts up to X
relocations (object name, destination name) for the robot to do now. No
curation, no task graphs, no transformer, no ranker. The ONLY filter applied
is dropping predictions whose object is already at the predicted destination.
Deliberately pure: if it re-predicts moves the human corrected, that is
informative baseline behavior.

Driver contract identical to the hybrid policy (tuple return, .history shim,
.train()/.load_artifacts()/.data_quantity/.reset()/._last_fwd).
"""

import glob
import json
import os

import torch

from agents.llm_human_prompt import LLMHumanClient
from agents.llm_memory_curator import extract_events_from_pt, render_events, _hhmm
from agents.robot_policy import RobotPolicyBase, RobotMoveAction
from agents.slatepro_utils import (
    load_common_data, filter_world_state_to_homer_nodes,
    world_state_to_history_initialization, update_history_edges,
    update_history_activity, update_history_time)

SYSTEM_PROMPT = (
    "You are a proactive household robot living with one person. You watch "
    "where they move objects each day. Learn their routine from the past "
    "days of observations, then — given today's observations so far — "
    "predict which objects they will need moved next and where, so the "
    "robot can pre-place them. Predict only moves that genuinely help the "
    "person's routine right now or very soon."
)


class RobotPolicyPlainLLM(RobotPolicyBase):

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = config['robot']['plain_llm']
        household = config['benchmark']['household']
        self.common_data = load_common_data(
            self.cfg['common_data_path'].format(household=household))
        self.client = LLMHumanClient(
            model=self.cfg['model'],
            cache_dir=self.cfg['cache_dir'],
            provider=self.cfg.get('provider', 'openai'),
        )
        self.history_text = None      # list[str], one per past day
        self.data_quantity = None
        self._last_fwd = None
        self.history = None
        self._log_path = None
        # first-occurrence class->slate idx maps (paper's class_to_idx
        # convention); objects restricted to movable rows
        names = self.common_data['node_classes']
        self._class_to_idx = {}
        for i, n in enumerate(names):
            self._class_to_idx.setdefault(n, i)
        self._reset_day_state()

    def _reset_day_state(self):
        self._day_events = []
        self._emitted_last_tick = set()

    # ------------------------------------------------------------------
    # Driver contract
    # ------------------------------------------------------------------

    def _update_history(self, world_state: dict) -> None:
        world_state = filter_world_state_to_homer_nodes(
            world_state, self.common_data)
        if self.history is None:
            self.history = world_state_to_history_initialization(
                world_state, self.common_data)
        else:
            self.history['edges'] = update_history_edges(
                world_state, self.history['edges'],
                self.history['vh_id_to_slate_idx'], self.common_data)
            self.history['activity_features'] = update_history_activity(
                world_state, self.history['activity_features'],
                self.common_data['activities'])
            self.history['time_features'], self.history['times'] = \
                update_history_time(world_state,
                                    self.history['time_features'],
                                    self.history['times'])
            self.history['sequence_length'] += 1

    def reset(self):
        """Clear per-day event log; keep loaded history text."""
        self.history = None
        self._last_fwd = None
        self._reset_day_state()

    def train(self, dataset_path: str, output_dir: str, days: int) -> None:
        """No learning: render the last N warmup days into prompt text."""
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')
        if os.path.isfile(weights_path):
            print(f"Plain-LLM history for day {days} already built, loading.")
            self.load_artifacts(weights_path)
            self.data_quantity = {'train': len(self.history_text), 'val': 0}
            return
        pt_files = sorted(glob.glob(os.path.join(dataset_path, 'train', '*.pt')))
        if not pt_files:
            raise RuntimeError(f"no warmup .pt files in {dataset_path}/train")
        n = int(self.cfg['history_days'])
        pt_files = pt_files[-n:]
        self.history_text = [
            render_events(extract_events_from_pt(p)['events'])
            for p in pt_files]
        torch.save({'history_text': self.history_text,
                    'n_days': len(self.history_text)}, weights_path)
        self.data_quantity = {'train': len(self.history_text), 'val': 0}
        self._log_path = os.path.join(output_dir, 'plainllm_log.jsonl')
        print(f"[plain_llm] built {len(self.history_text)}-day history "
              f"-> {weights_path}")

    def load_artifacts(self, weights_path: str) -> None:
        d = torch.load(weights_path, map_location='cpu', weights_only=False)
        self.history_text = d['history_text']
        self._log_path = os.path.join(os.path.dirname(weights_path),
                                      'plainllm_log.jsonl')
        print(f"[plain_llm] loaded {len(self.history_text)}-day history "
              f"from {weights_path}")

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _observe_human_moves(self, t_now: float) -> bool:
        if self.history['sequence_length'] < 2:
            self._emitted_last_tick = set()
            return False
        prev = self.history['edges'][0, -2].argmax(dim=-1)
        cur = self.history['edges'][0, -1].argmax(dim=-1)
        movable = self.history['dynamic_edges_mask'][0, 0].sum(-1) > 0
        names = self.common_data['node_classes']
        new = False
        for i in ((cur != prev) & movable).nonzero(as_tuple=False).squeeze(-1).tolist():
            dest = int(cur[i])
            if (i, dest) in self._emitted_last_tick:
                continue
            self._day_events.append({
                't_min': float(t_now), 'obj_idx': int(i), 'dest_idx': dest,
                'prev_idx': int(prev[i]), 'obj_class': names[i],
                'dest_class': names[dest], 'prev_class': names[int(prev[i])]})
            new = True
        self._emitted_last_tick = set()
        return new

    def _schema(self) -> dict:
        names = self.common_data['node_classes']
        movable = self.history['dynamic_edges_mask'][0, 0].sum(-1) > 0
        obj_names = sorted({names[i] for i in range(len(names)) if movable[i]})
        # 'home' is the graph root (VH id -1), not a placeable surface — the
        # executor crashes resolving it as a destination
        dest_names = sorted(set(names) - {'home'})
        return {
            "name": "predict_relocations",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["predictions"],
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["object", "destination"],
                            "properties": {
                                "object": {"type": "string", "enum": obj_names},
                                "destination": {"type": "string",
                                                "enum": dest_names},
                            },
                        },
                    },
                },
            },
        }

    def get_actions(self, node_map, *, world_state: dict = None):
        """Ask LLM for relocations every tick (past human moves + current time)."""
        if world_state is None:
            raise ValueError('world_state is required for the plain_llm policy')
        if self.history_text is None:
            raise RuntimeError('plain_llm asked to act before train()')
        self._update_history(world_state)
        self._last_fwd = None
        t_now = float(world_state['time'])
        time_str = _hhmm(t_now)

        self._observe_human_moves(t_now)

        x = int(self.cfg['max_predictions'])
        past = "\n\n".join(
            f"### Day -{len(self.history_text) - i}\n{txt}"
            for i, txt in enumerate(self.history_text))
        if self._day_events:
            today_body = render_events(self._day_events)
        else:
            today_body = (
                "The person has taken no relocation actions yet today "
                "(morning start — predict from past-day routine and time of day)."
            )
        user = (
            f"## Past {len(self.history_text)} days of observations "
            "(learn the person's routine from these)\n"
            f"{past}\n\n"
            f"## Today so far (incomplete; current time is {time_str})\n"
            f"{today_body}\n\n"
            f"## Predict up to {x} object relocations the robot should do "
            "RIGHT NOW (object and destination), based on what the person "
            "will need next in their routine.\n"
        )
        payload, _, _ = self.client.call(SYSTEM_PROMPT, user, self._schema())
        if self.client.cost_usd > float(self.cfg['max_usd']):
            raise RuntimeError(
                f"plain_llm exceeded max_usd={self.cfg['max_usd']}")

        cur_locs = self.history['edges'][0, -1].argmax(dim=-1)
        slate_to_vh = {v: k for k, v in
                       self.history['vh_id_to_slate_idx'].items()}
        node_id_to_class = self.history['node_id_to_class']
        actions, obj_idxs, loc_idxs, prev_loc_idxs, log = [], [], [], [], []
        for p in payload['predictions'][:x]:
            obj_idx = self._class_to_idx.get(p['object'])
            dest_idx = self._class_to_idx.get(p['destination'])
            rec = {'object': p['object'], 'destination': p['destination']}
            log.append(rec)
            if obj_idx is None or dest_idx is None:
                rec['outcome'] = 'unmapped'
                continue
            if p['destination'] == 'home':
                # graph root (VH id -1), unplaceable; old cached responses may
                # still contain it after the enum fix
                rec['outcome'] = 'unplaceable_home'
                print(f"[plain_llm] dropped unplaceable destination 'home' "
                      f"for {p['object']}")
                continue
            if int(cur_locs[obj_idx]) == dest_idx:
                rec['outcome'] = 'already_at_dest'   # the only filter
                continue
            obj_vh, dest_vh = slate_to_vh.get(obj_idx), slate_to_vh.get(dest_idx)
            if obj_vh is None or dest_vh is None:
                raise RuntimeError(
                    f'no VH id for slate idx {obj_idx}/{dest_idx}')
            rec['outcome'] = 'emitted'
            actions.append(RobotMoveAction(
                node_id_to_class.get(obj_vh, 'unknown'), obj_vh,
                node_id_to_class.get(dest_vh, 'unknown'), dest_vh))
            obj_idxs.append(obj_idx)
            loc_idxs.append(dest_idx)
            prev_loc_idxs.append(int(cur_locs[obj_idx]))
            self._emitted_last_tick.add((obj_idx, dest_idx))
        if self._log_path and log:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps({'t_min': t_now, 'predictions': log}) + '\n')
        return actions, (obj_idxs, loc_idxs, prev_loc_idxs)
