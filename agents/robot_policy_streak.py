"""STREAK-based robot policy (CoRL 2022 ancestor of SLaTe-PRO).

Wraps ``GraphTranslatorModule`` for stateless single-step (prev→next) graph
translation. Maintains SLaTe-PRO-compatible ``history`` so ``closed_loop_eval``
can emit training datasets unchanged.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from adict import adict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

_log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
STREAK_ROOT = os.path.join(PROJECT_ROOT, 'external', 'STREAK')
sys.path.insert(0, STREAK_ROOT)
sys.path.insert(0, os.path.join(STREAK_ROOT, 'helpers'))

from GraphTranslatorModule import GraphTranslatorModule  # noqa: E402
from graph_translator_cl import GraphTranslatorCL  # noqa: E402
from encoders import time_external, TimeEncodingOptions  # noqa: E402

from agents.robot_policy import RobotPolicyBase, RobotMoveAction  # noqa: E402
from agents.slatepro_utils import (  # noqa: E402
    filter_world_state_to_homer_nodes,
    load_common_data,
    update_history_activity,
    update_history_edges,
    update_history_time,
    world_state_to_history_initialization,
)

from coadaptation_adapter import prepare_streak_dataset, build_containment_ban_mask  # noqa: E402


def _homer_common_data_path(household: str) -> str:
    hh = (household or 'A').upper()
    return os.path.join(
        PROJECT_ROOT, 'external', 'HOMER_PLUS', f'Household{hh}',
        'processed_seqLM_coarse', 'common_data.json',
    )


class RobotPolicyStreak(RobotPolicyBase):
    """Robot policy using STREAK to predict object relocations."""

    def __init__(self, config: dict):
        super().__init__(config)
        streak_cfg = config['robot']['streak']
        self.confidence_threshold = streak_cfg.get('confidence_threshold', 0.0)
        self.lookahead_steps = streak_cfg.get('lookahead_steps', 3)
        # Optional per-tick action cap: emit only the top-`max_actions` valid
        # relocations by confidence (None = uncapped, the paper default). Bounds
        # the Markovian re-proposal flood; keeps the model's highest-confidence
        # picks and drops the long tail.
        self.max_actions = streak_cfg.get('max_actions', None)
        self.checkpoint_path = streak_cfg.get('model_checkpoint')
        self.train_from_scratch = streak_cfg.get('train_from_scratch', True)
        self.time_encoder_name = streak_cfg.get('time_encoder', 'sine_informed')
        # Resolve the time encoder ONCE — the single source of truth used identically
        # for (a) training data via RoutinesDataset, (b) eval via _time_encode, and
        # (c) model c_len. STREAK's paper default is sine_informed (14-dim sinusoids);
        # time_external (4-dim raw [weeks,days,hrs,mins]) is a poor, discontinuous
        # representation. Keeping one callable everywhere prevents train/eval mismatch.
        _opt = ('external' if self.time_encoder_name in ('time_external', 'external')
                else self.time_encoder_name)
        self.time_encoder = TimeEncodingOptions()(_opt)
        # Continual-learning: each train() round (every train_every_n_days) is one task.
        self.cl_mode = streak_cfg.get('cl_mode', 'none')   # none | l2 | ewc
        self.ewc_lambda = streak_cfg.get('ewc_lambda')      # None -> variant default
        if self.cl_mode not in ('none', 'l2', 'ewc'):
            raise ValueError(f"robot.streak.cl_mode must be none|l2|ewc, got {self.cl_mode!r}")
        self._cl_consumed = 0   # count of source train .pt files already used by prior tasks
        self.time_step_minutes = config['simulation']['time_step_minutes']
        self.household_name = config.get('household', config.get('benchmark', {}).get('household'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.history = None
        self._last_fwd = None
        self.data_quantity = {'train': -1, 'val': -1}

        common_path = (
            os.path.join(self.checkpoint_path, 'common_data.json')
            if self.checkpoint_path and os.path.isdir(self.checkpoint_path)
            else _homer_common_data_path(self.household_name)
        )
        if not os.path.isfile(common_path):
            raise FileNotFoundError(
                f'STREAK/HOMER common_data.json not found at {common_path}. '
                'Set robot.streak.model_checkpoint or config household.'
            )
        self.common_data = load_common_data(common_path)
        self.n_nodes = self.common_data['n_nodes']
        self._build_model_configs()
        self._load_nonstatic_mask(common_path)
        # Containment-ban mask: forbid X->Y when Y normally rests on X (would invert
        # an observed containment and create an ancestor cycle). Surgical — bans only
        # those inversions, not a destination whitelist.
        homer_dir = os.path.dirname(_homer_common_data_path(self.household_name))
        self.containment_ban = build_containment_ban_mask(homer_dir, self.n_nodes)
        print(f'[STREAK] containment-ban mask: {int(self.containment_ban.sum())} '
              f'forbidden (X->Y where Y rests on X) of {self.n_nodes**2} pairs')

        if not self.train_from_scratch and self.checkpoint_path:
            self._load_model_from_checkpoint(self.checkpoint_path)
        else:
            print('[STREAK] train_from_scratch=True - model weights created on first train()')

        self._vh_id_to_idx: dict[int, int] = {}
        self._idx_to_class: dict[int, str] = {}

    def _new_model(self):
        """Instantiate base or CL model per cl_mode. CL variants subclass the base,
        so inference (forward) is identical; cl_mode only adds a training penalty."""
        if self.cl_mode == 'none':
            return GraphTranslatorModule(model_configs=self.model_configs)
        return GraphTranslatorCL(
            model_configs=self.model_configs,
            cl_mode=self.cl_mode,
            ewc_lambda=self.ewc_lambda,
            t=0,
        )

    def _build_model_configs(self) -> None:
        # c_len from the actual encoder output (robust: matches data.params c_len,
        # which is what the model is ultimately built/loaded with).
        c_len = int(self.time_encoder(0.0).numel())
        self.model_configs = adict(
            n_nodes=self.n_nodes,
            n_len=self.n_nodes,
            c_len=c_len,
            edge_importance='predicted',
            hidden_layer_size=20,
        )

    def _load_nonstatic_mask(self, common_data_path: str) -> None:
        streak_root = os.path.dirname(common_data_path)
        ns_path = os.path.join(streak_root, 'nonstatic_edges.pt')
        if os.path.isfile(ns_path):
            self.nonstatic_edges = torch.load(ns_path, map_location='cpu', weights_only=True).float()
        else:
            from coadaptation_adapter import build_nonstatic_edges
            self.nonstatic_edges = build_nonstatic_edges(self.common_data, self.n_nodes)
        if self.nonstatic_edges.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f'nonstatic_edges shape {tuple(self.nonstatic_edges.shape)} '
                f'!= ({self.n_nodes}, {self.n_nodes})'
            )

    def _load_model_from_checkpoint(self, checkpoint_dir: str) -> None:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if not ckpt_files:
            raise FileNotFoundError(f'No .ckpt files in {checkpoint_dir}')
        ckpt = max(ckpt_files, key=os.path.getctime)
        print(f'[STREAK] Loading checkpoint {ckpt}')
        if self.cl_mode == 'none':
            self.model = GraphTranslatorModule.load_from_checkpoint(
                ckpt, model_configs=self.model_configs)
        else:
            self.model = GraphTranslatorCL.load_from_checkpoint(
                ckpt, model_configs=self.model_configs,
                cl_mode=self.cl_mode, ewc_lambda=self.ewc_lambda)
        self._setup_model_for_inference()

    def reset(self) -> None:
        self.history = None
        self._last_fwd = None

    def _time_encode(self, t_minutes: float) -> torch.Tensor:
        # Same callable used for the training data (RoutinesDataset) — no mismatch.
        return self.time_encoder(float(t_minutes))

    def _binarize_edges(self, edges: torch.Tensor) -> torch.Tensor:
        locs = edges.argmax(dim=-1)
        out = torch.zeros_like(edges)
        out[torch.arange(edges.size(0), device=edges.device), locs] = 1.0
        row_sums = out.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise AssertionError(
                'Edge matrix axis mismatch: expected row-normalized object→location '
                f'(row sums min={row_sums.min():.4f} max={row_sums.max():.4f})'
            )
        return out

    def _build_input(self, world_state: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ws = filter_world_state_to_homer_nodes(world_state, self.common_data)
        hist = world_state_to_history_initialization(ws, self.common_data)
        self._vh_id_to_idx = hist['vh_id_to_slate_idx']
        self._idx_to_class = hist['node_id_to_class']

        edges = self._binarize_edges(hist['edges'][0, 0])
        n = edges.size(0)
        assert n == self.n_nodes, f'Graph size {n} != model n_nodes {self.n_nodes}'
        nodes = torch.eye(n, device=self.device)
        context = self._time_encode(float(world_state['time'])).to(self.device)

        return (
            edges.unsqueeze(0).to(self.device),
            nodes.unsqueeze(0),
            context.unsqueeze(0),
        )

    def _apply_dynamic_mask(self, edges_pred: torch.Tensor, current_edges: torch.Tensor) -> torch.Tensor:
        dyn = self.nonstatic_edges.to(edges_pred.device)
        current_edges = current_edges.to(edges_pred.device)
        masked = edges_pred.clone()
        masked[dyn == 0] = -float('inf')                       # static objects don't move
        ban = self.containment_ban.to(edges_pred.device)       # ban[X,Y]: Y rests on X
        masked[ban] = -float('inf')                            # forbid X->Y inversions (anti-cycle)
        probs = F.softmax(masked, dim=-1)
        probs[dyn == 0] = current_edges[dyn == 0]
        return probs

    def _run_forward(self, world_state: dict) -> dict:
        edges, nodes, context = self._build_input(world_state)
        current_edges = edges[0].clone().cpu()
        dyn_flat = self.nonstatic_edges.sum(dim=-1) > 0

        best_dest = current_edges.argmax(dim=-1).clone()
        best_conf = torch.zeros(self.n_nodes)
        changes_pred = torch.zeros(self.n_nodes, dtype=torch.bool)

        with torch.no_grad():
            self.model.eval()
            input_edges = edges
            input_context = context
            for step in range(self.lookahead_steps):
                if step > 0:
                    t_future = float(world_state['time']) + self.time_step_minutes * step
                    input_context = self._time_encode(t_future).unsqueeze(0).to(self.device)

                edges_pred, _, _, _ = self.model(input_edges, nodes, input_context)
                probs = self._apply_dynamic_mask(edges_pred[0], input_edges[0]).cpu()
                pred_locs = probs.argmax(dim=-1)
                conf = probs.max(dim=-1).values
                reference = current_edges.argmax(dim=-1)

                if step == 0:
                    best_dest = pred_locs.clone()
                    changes_pred = (best_dest != reference) & dyn_flat
                    new_changes = changes_pred.clone()
                else:
                    new_changes = (pred_locs != reference) & dyn_flat & ~changes_pred
                    best_dest[new_changes] = pred_locs[new_changes]
                    changes_pred = changes_pred | new_changes

                continued = pred_locs == best_dest
                better = ((conf > best_conf) & continued) | new_changes
                best_conf[better] = conf[better]

                input_edges = probs.unsqueeze(0).to(self.device)

        valid = changes_pred & (best_conf >= self.confidence_threshold) & dyn_flat
        return {
            'pred_locations': best_dest,
            'current_locations': current_edges.argmax(dim=-1),
            'relocation_confidences': best_conf,
            'valid_relocations': valid,
            'dyn_mask_flat': dyn_flat,
        }

    def _update_history(self, world_state: dict) -> None:
        world_state = filter_world_state_to_homer_nodes(world_state, self.common_data)
        if self.history is None:
            self.history = world_state_to_history_initialization(world_state, self.common_data)
        else:
            self.history['edges'] = update_history_edges(
                world_state, self.history['edges'], self.history['vh_id_to_slate_idx'],
                self.common_data)
            self.history['activity_features'] = update_history_activity(
                world_state, self.history['activity_features'], self.common_data['activities'])
            self.history['time_features'], self.history['times'] = update_history_time(
                world_state, self.history['time_features'], self.history['times'])
            self.history['sequence_length'] += 1

    def get_actions(self, node_map, *, world_state: dict):
        self._update_history(world_state)
        if self.model is None:
            return [], ([], [], [])

        fwd = self._run_forward(world_state)
        self._last_fwd = fwd

        valid = fwd['valid_relocations']
        if not valid.any():
            return [], ([], [], [])

        slate_to_vh = {v: k for k, v in self._vh_id_to_idx.items()}
        actions = []
        obj_idxs, loc_idxs, prev_idxs = [], [], []

        indices = torch.where(valid)[0]
        order = fwd['relocation_confidences'][indices].argsort(descending=True)
        if self.max_actions is not None:
            order = order[:self.max_actions]
        for rank in order:
            idx = indices[rank]
            action = self._idx_to_action(
                idx.item(),
                fwd['pred_locations'][idx].item(),
                slate_to_vh,
            )
            if action is not None:
                actions.append(action)
                obj_idxs.append(idx.item())
                loc_idxs.append(fwd['pred_locations'][idx].item())
                prev_idxs.append(fwd['current_locations'][idx].item())

        return actions, (obj_idxs, loc_idxs, prev_idxs)

    def _idx_to_action(self, obj_idx: int, loc_idx: int, slate_to_vh: dict) -> RobotMoveAction | None:
        obj_vh_id = slate_to_vh.get(obj_idx)
        loc_vh_id = slate_to_vh.get(loc_idx)
        if obj_vh_id is None or loc_vh_id is None:
            return None
        obj_class = self._idx_to_class.get(obj_vh_id, 'unknown')
        loc_class = self._idx_to_class.get(loc_vh_id, 'unknown')
        if obj_class == 'home' or loc_class == 'home':
            return None
        return RobotMoveAction(obj_class, obj_vh_id, loc_class, loc_vh_id)

    def _setup_model_for_inference(self) -> None:
        self.model.eval()
        self.model.to(self.device)

    def load_artifacts(self, weights_path: str) -> None:
        if self.model is None:
            self.model = self._new_model()
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self._setup_model_for_inference()

    def _window_basenames(self, dataset_path: str):
        """For CL: the source train .pt files NOT yet used by a prior task (the new
        window). Returns (new_basenames, total_count). Empty new -> caller reuses all."""
        src_train = os.path.join(dataset_path, 'train')
        allf = sorted(os.path.splitext(f)[0] for f in os.listdir(src_train)
                      if f.endswith('.pt'))
        return allf[self._cl_consumed:], len(allf)

    def train(self, dataset_path: str, output_dir: str, days: str) -> None:
        from reader import RoutinesDataset

        # Optional explicit seed (STREAK is otherwise unseeded and multimodal:
        # weight init + Lightning data shuffling land in different basins run to
        # run). Set STREAK_SEED to make a draw reproducible and distinguishable.
        # Seed here, before _new_model() (weight init) and trainer.fit (shuffle).
        _seed = os.environ.get('STREAK_SEED', '')
        if _seed != '':
            from pytorch_lightning import seed_everything
            seed_everything(int(_seed), workers=True)
            print(f'[STREAK] seed_everything({int(_seed)})')

        ckpt_dir = os.path.join(output_dir, f'pl_checkpoints_day{days}')
        os.makedirs(ckpt_dir, exist_ok=True)
        done_marker = os.path.join(ckpt_dir, 'training_done')
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')
        cl_meta_path = os.path.join(output_dir, f'cl_meta_day{days}.json')

        if os.path.isfile(done_marker) and os.path.isfile(weights_path):
            print(f'[STREAK] Training for day {days} already complete, loading weights.')
            self.load_artifacts(weights_path)
            self.data_quantity = {'train': -1, 'val': -1}
            # Restore CL bookkeeping so the NEXT task windows correctly on resume.
            if self.cl_mode != 'none' and os.path.isfile(cl_meta_path):
                with open(cl_meta_path) as f:
                    self._cl_consumed = int(json.load(f).get('consumed', self._cl_consumed))
            return

        homer_common = _homer_common_data_path(self.household_name)
        # CL: train each task on only the NEW window (new days since last task);
        # base STREAK (cl_mode=none): cumulative, faithful train-from-scratch each round.
        only_basenames, total_files = (None, None)
        if self.cl_mode != 'none':
            only_basenames, total_files = self._window_basenames(dataset_path)
            if not only_basenames:
                print(f'[STREAK-CL] no new routines since last task (consumed='
                      f'{self._cl_consumed}); skipping consolidation for day {days}.')
                return
            task_out = os.path.join(dataset_path, f'streak_adapted_task_day{days}')
            streak_data_path = prepare_streak_dataset(
                dataset_path, homer_common, output_path=task_out,
                only_basenames=only_basenames, force=False)
            print(f'[STREAK-CL] task t={int(self.model.cl_t.item()) if self.model else 0} '
                  f'window={only_basenames} (cl_mode={self.cl_mode}, lambda={self.ewc_lambda})')
        else:
            streak_data_path = prepare_streak_dataset(dataset_path, homer_common, force=False)

        data = RoutinesDataset(data_path=streak_data_path,
                               time_encoder=self.time_encoder, batch_size=1)
        train_loader = data.get_train_loader()
        test_loader = data.get_test_loader()

        self.data_quantity = {
            'train': len(train_loader) if train_loader else -1,
            'val': len(test_loader) if test_loader else -1,
        }

        if self.model is None:
            cfg = adict(self.model_configs)
            cfg.update(data.params)
            self.model_configs = cfg
            self.model = self._new_model()

        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        resume_ckpt = last_ckpt if os.path.isfile(last_ckpt) else None
        base_max_epochs = int(os.environ.get('MAX_TRAIN_EPOCHS', '500'))
        max_epochs = base_max_epochs
        if resume_ckpt:
            resumed_epoch = int(torch.load(
                resume_ckpt, map_location='cpu', weights_only=False).get('epoch', 0))
            max_epochs = max(base_max_epochs, resumed_epoch + 200)
            print(f'[STREAK] Resuming from {resume_ckpt} (epoch={resumed_epoch})')

        ckpt_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=10)
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            max_epochs=max_epochs,
            log_every_n_steps=1,
            callbacks=[ckpt_callback],
        )
        print(f'[STREAK] Training on {self.data_quantity["train"]} examples '
              f'({streak_data_path})')
        trainer.fit(self.model, train_loader, ckpt_path=resume_ckpt)

        # CL task boundary: snapshot trained weights as reference params (and, for
        # EWC, the diagonal Fisher over THIS task's data), then advance the task
        # counter. Must run before saving so weights_dayN.pt carries the updated
        # buffers (resume restores the consolidated state for the next task).
        if self.cl_mode != 'none':
            self.model.consolidate(train_loader)
            self._cl_consumed = total_files
            with open(cl_meta_path, 'w') as f:
                json.dump({'consumed': self._cl_consumed,
                           'cl_t': int(self.model.cl_t.item())}, f)

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)
        with open(done_marker, 'w') as f:
            f.write('done\n')
        self._setup_model_for_inference()
