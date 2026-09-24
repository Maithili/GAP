"""P0 oracle ceiling policy — DELIBERATELY reads the GT day script.

This is a measurement instrument, not a method: it knows every (time, object,
destination) move the human will make today (from the day's source .pt) and
pre-places objects under the same physics as every other policy (10-min
ticks, same executor, same reconciliation). It cannot be perfect by
construction: a script line whose object was used by its previous line less
than two ticks earlier has no free tick to claim. Its f1 decomposes failure:

    1 − oracle_f1            = structurally unclaimable (no policy can get it)
    oracle_f1 − best_method  = the true remaining method deficit

Pure ceiling only — no policy-style guards (caps, act windows, etc.).

Eligibility for line i of an object (reach times t_{i-1} < t_i from the GT
stream): emit at EXACTLY one tick in advance,
    t_now == t_i − 10   and   t_i − 10 >= t_{i-1} + 10
Within each benchmark tick the **robot acts first, then the human**
(``closed_loop_eval.py``); acting AT t_i is too late (the human already moved
that tick). Acting one tick early (t_i − 10) places the object just before the
human needs it → ``skipped``. The second condition drops lines with no free
tick (object used <2 ticks before its next move) — the structurally
unclaimable ceiling. At most one move per object per tick (earliest reach wins).

GT source: the processed ``.pt`` paired with the human's ``script_path`` for
the day (set via ``reset(script_path=...)`` from ``closed_loop_eval``).
"""

import json
import os

import torch

from agents.llm_memory_curator import extract_events_from_pt
from agents.robot_policy import RobotPolicyBase, RobotMoveAction
from agents.slatepro_utils import (
    homers_script_path_to_processed_pt,
    load_common_data, filter_world_state_to_homer_nodes,
    world_state_to_history_initialization, update_history_edges,
    update_history_activity, update_history_time)

TICK_MIN = 10.0


class RobotPolicyOracle(RobotPolicyBase):

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = config['robot']['oracle']
        household = config['benchmark']['household']
        self.common_data = load_common_data(
            self.cfg['common_data_path'].format(household=household))
        self.data_quantity = None
        self._last_fwd = None
        self.history = None
        self._log_path = None
        self._day_counter = -1
        self._script_path = None
        self._reset_day_state()

    def _reset_day_state(self):
        self._script = None            # obj_idx -> [(t_reach, dest_idx), ...]
        self._emitted_lines = set()    # (obj_idx, line_index)

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

    def reset(self, *, script_path: str | None = None):
        """Clear per-day state; *script_path* selects the human's GT ``.pt``."""
        if script_path is not None:
            self._script_path = script_path
        self.history = None
        self._last_fwd = None
        self._day_counter += 1
        self._reset_day_state()

    def train(self, dataset_path: str, output_dir: str, days: int) -> None:
        """No learning — write the bookkeeping stub the driver expects."""
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')
        if not os.path.isfile(weights_path):
            torch.save({'oracle': True}, weights_path)
        self.data_quantity = {'train': 0, 'val': 0}
        self._log_path = os.path.join(output_dir, 'oracle_log.jsonl')
        print("[oracle] ready (pure ceiling, emit=1-tick-ahead)")

    def load_artifacts(self, weights_path: str) -> None:
        self._log_path = os.path.join(os.path.dirname(weights_path),
                                      'oracle_log.jsonl')
        print(f"[oracle] load_artifacts noop ({weights_path})")

    # ------------------------------------------------------------------
    # GT script access (the whole point of the oracle)
    # ------------------------------------------------------------------

    def _load_day_script(self) -> dict:
        """Load GT relocations from the ``.pt`` paired with today's human script."""
        if not self._script_path:
            raise RuntimeError(
                'oracle: no script_path set — call reset(script_path=...) '
                'before get_actions')
        pt = homers_script_path_to_processed_pt(self._script_path)
        if not os.path.isfile(pt):
            raise RuntimeError(
                f"oracle: no GT file {pt} for human script {self._script_path} "
                f"(day_counter={self._day_counter})")
        events = extract_events_from_pt(pt)['events']
        script = {}
        for e in events:
            script.setdefault(e['obj_idx'], []).append(
                (float(e['t_min']), int(e['dest_idx'])))
        for lines in script.values():
            lines.sort(key=lambda x: x[0])
        return script

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def get_actions(self, node_map, *, world_state: dict = None):
        """Emit every eligible 1-tick-ahead GT teleport (pure physics ceiling)."""
        if world_state is None:
            raise ValueError('world_state is required for the oracle policy')
        self._update_history(world_state)
        self._last_fwd = None
        t_now = float(world_state['time'])
        if self._script is None:
            self._script = self._load_day_script()

        cur_locs = self.history['edges'][0, -1].argmax(dim=-1)
        slate_to_vh = {v: k for k, v in
                       self.history['vh_id_to_slate_idx'].items()}
        node_id_to_class = self.history['node_id_to_class']

        eligible = []   # (t_reach, obj_idx, dest_idx, line_idx)
        for obj_idx, lines in self._script.items():
            for i, (t_reach, dest_idx) in enumerate(lines):
                if (obj_idx, i) in self._emitted_lines:
                    continue
                t_prev = lines[i - 1][0] if i > 0 else -1e9
                emit_t = t_reach - TICK_MIN          # act exactly one tick early
                if t_now != emit_t:
                    continue
                if emit_t < t_prev + TICK_MIN:       # no free tick before reach
                    continue                         # structurally unclaimable
                if int(cur_locs[obj_idx]) == dest_idx:
                    continue   # already there (or we did an earlier line)
                eligible.append((t_reach, obj_idx, dest_idx, i))
        eligible.sort()   # soonest reach first

        # At most one move per object per tick: keep the first (earliest reach).
        seen_obj, deduped = set(), []
        for e in eligible:
            if e[1] in seen_obj:
                continue
            seen_obj.add(e[1])
            deduped.append(e)
        eligible = deduped

        actions, obj_idxs, loc_idxs, prev_loc_idxs, log = [], [], [], [], []
        for t_reach, obj_idx, dest_idx, line_i in eligible:
            obj_vh, dest_vh = slate_to_vh.get(obj_idx), slate_to_vh.get(dest_idx)
            if obj_vh is None or dest_vh is None:
                raise RuntimeError(f'no VH id for slate idx {obj_idx}/{dest_idx}')
            actions.append(RobotMoveAction(
                node_id_to_class.get(obj_vh, 'unknown'), obj_vh,
                node_id_to_class.get(dest_vh, 'unknown'), dest_vh))
            obj_idxs.append(obj_idx)
            loc_idxs.append(dest_idx)
            prev_loc_idxs.append(int(cur_locs[obj_idx]))
            self._emitted_lines.add((obj_idx, line_i))
            log.append({'t_now': t_now, 't_reach': t_reach,
                        'obj': self.common_data['node_classes'][obj_idx],
                        'dest': self.common_data['node_classes'][dest_idx]})
        if self._log_path and log:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps({'day': self._day_counter,
                                    'script': self._script_path,
                                    'emitted': log}) + '\n')
        return actions, (obj_idxs, loc_idxs, prev_loc_idxs)
