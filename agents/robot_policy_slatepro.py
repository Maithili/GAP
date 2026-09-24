"""SLaTe-PRO-based robot policy.

Wraps the SLaTe-PRO model to predict object relocations and generate
robot actions (teleport moves) based on predicted future object locations.
"""

import json
import time
import logging
import os
import sys
import torch
import datetime

_log = logging.getLogger(__name__)

# Add SLaTe-PRO to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SLATEPRO_ROOT = os.path.join(PROJECT_ROOT, 'external', 'SLaTe-PRO')
sys.path.insert(0, SLATEPRO_ROOT)
sys.path.insert(0, os.path.join(SLATEPRO_ROOT, 'helpers'))
sys.path.insert(0, os.path.join(SLATEPRO_ROOT, 'models'))

from adict import adict
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

try:
    from MultiModalUserTracking import MultiModalUserTrackingModule
except ImportError:
    MultiModalUserTrackingModule = None
    print('Warning: Could not import MultiModalUserTrackingModule. SLaTe-PRO policy will not work.')

from agents.robot_policy import RobotPolicyBase, RobotMoveAction
from agents.slatepro_utils import (
    filter_world_state_to_homer_nodes,
    load_common_data,
    world_state_to_history_initialization,
    update_history_edges,
    update_history_activity,
    update_history_time,
    time_encoder,
)

from pytorch_lightning.loggers import WandbLogger
import importlib
_loader_sequential = importlib.import_module("external.SLaTe-PRO.helpers.loader_sequential")
RoutinesDataset = _loader_sequential.RoutinesDataset

def dictcompare(d1, d2):
    """Recursively compare two dicts; print first mismatch and return False."""
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    if k1 != k2:
        print(f'Keys do not match: {k1} != {k2}')
        print(f'Keys in d1 but not in d2: {k1 - k2}')
        print(f'Keys in d2 but not in d1: {k2 - k1}')
        return False
    for k in k1:
        if d1[k] != d2[k]:
            print(f'Values for key {k} do not match: {d1[k]} != {d2[k]}')
            return False
    return True

def _raise_ckpt_not_found(requested: str):
    """Raise with instructions when no SLaTe-PRO checkpoint directory exists."""
    raise FileNotFoundError(
        f'SLaTe-PRO checkpoint directory not found: {requested}\n'
        'Train a model first, e.g.:\n'
        '  python scripts/pretrain_slatepro.py --household B --output_dir checkpoints/slate_pro/\n'
        'Then use the run directory that contains a .ckpt file, e.g.:\n'
        '  --robot.slate_pro.model_checkpoint checkpoints/slate_pro/householdB/default_100'
    )


class RobotPolicySlatepro(RobotPolicyBase):
    """Robot policy using SLaTe-PRO model to predict object relocations.

    At each timestep, feeds the current world state through SLaTe-PRO,
    extracts predicted object relocations, and generates teleport actions
    for objects that are predicted to move.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        robot_cfg = config['robot']
        slate_cfg = robot_cfg['slate_pro']
        paths_cfg = config.get('paths', {})

        self.confidence_threshold = slate_cfg['confidence_threshold']
        self.max_history_len = slate_cfg.get('max_history_len', 144)
        self.lookahead_steps = slate_cfg.get('lookahead_steps', 1)
        self.checkpoint_path = slate_cfg['model_checkpoint']
        self.train_from_scratch = slate_cfg['train_from_scratch']
        self.time_step_minutes = config['simulation']['time_step_minutes']
        self.household_name = config.get('household')
        
        if MultiModalUserTrackingModule is None:
            raise ImportError('SLaTe-PRO modules not available. Cannot create RobotPolicySlatepro.')
        
        # Load model
        self.model = None
        self.common_data = None
        self.node_map_reverse = {}  # slate_idx -> (vh_node_id, class_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not self.train_from_scratch:
            # Checkpoint dir supplies architecture/config only (_load_model_only_cfg);
            # sim-trained weights come from train() or closed_loop_eval resume.
            checkpoint_root = self.checkpoint_path
            self._load_model_only_cfg(checkpoint_root, config['household'])
        else:
            raise NotImplementedError('Training from scratch not yet implemented in RobotPolicySlatepro')

        self.history = None
        self._last_fwd = None  # cached _run_forward output for offline-GT eval

        self.logger = WandbLogger(settings=wandb.Settings(start_method="fork"))
        self.logger.experiment.config.update(config)
        # self.logger = wandb.init(project="coadaptation-simulation", name="slatepro-incremental")

    def reset(self):
        """Clear per-episode state so each run starts fresh."""
        self.history = None

    def _probe_model_inputs_vs_offline(self, graph_seq_nodes, graph_seq_edges,
                                        graph_dyn_edges, activity_seq, time_context,
                                        offline_data):
        """Compare the exact tensors about to be fed to the model against
        offline_data at the matching positions. Asserts on any mismatch."""
        import math as _math
        offline_times = offline_data['times']
        curr_t = float(self.history['times'][-1])
        match = (offline_times == curr_t).nonzero(as_tuple=False)
        if match.numel() == 0:
            return
        idx = int(match[0, 0].item())
        T_graph    = graph_seq_edges.size(1)
        T_activity = activity_seq.size(1)
        T_time     = time_context.size(1)
        n_classes  = graph_seq_nodes.size(-1)
        n_act      = activity_seq.size(-1)
        c_len      = time_context.size(-1)
        # Use real num_states from history, not shape inference (iter 0 and
        # iter 1 both give T_graph=2 T_activity=1 T_time=1, making shape-only
        # detection ambiguous).
        num_states = self.history['sequence_length']
        padded = num_states < 2

        def _eq(name, actual, expected, atol=1e-3):
            actual = actual.cpu().float()
            expected = expected.cpu().float()
            if actual.shape != expected.shape:
                print(
                    f"[WARN][model-input probe] shape mismatch in {name}: "
                    f"actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
                return
            if not torch.allclose(actual, expected, atol=atol):
                d = (actual - expected).abs()
                print(
                    f"[WARN][model-input probe] value mismatch in {name} at curr_t={curr_t} "
                    f"idx={idx} padded={padded}: max_abs={d.max().item():.4g} "
                    f"n_diff={(d > atol).sum().item()}/{d.numel()}")

        # Map to offline positions. Closed-loop history positions [0..num_states-1]
        # correspond to offline positions [idx-num_states+1..idx].
        hist_start = idx - num_states + 1

        # Expected graph_seq_edges
        if padded:
            # _prepare repeats single history frame (history[0] = offline[idx])
            expected_edges = offline_data['edges'][idx:idx+1].unsqueeze(0).repeat(1, 2, 1, 1).float()
        else:
            expected_edges = offline_data['edges'][hist_start:idx+1].unsqueeze(0).float()
        _eq('graph_seq_edges', graph_seq_edges, expected_edges)

        # Expected graph_seq_nodes = one-hot(offline.nodes) broadcast to T_graph
        nodes_1hot = torch.nn.functional.one_hot(
            offline_data['nodes'].long(), num_classes=n_classes
        ).float()  # (N, n_classes)
        expected_nodes = nodes_1hot.unsqueeze(0).unsqueeze(0).repeat(1, T_graph, 1, 1)
        _eq('graph_seq_nodes', graph_seq_nodes, expected_nodes)

        # Expected graph_dyn_edges = active_edges broadcast to T_graph
        expected_dyn = offline_data['active_edges'].unsqueeze(0).unsqueeze(0).repeat(
            1, T_graph, 1, 1).float()
        _eq('graph_dyn_edges', graph_dyn_edges, expected_dyn)

        # Expected activity_seq: _prepare slices history[:, :num_states-1, :]
        # → offline positions [hist_start .. idx-1] = [hist_start .. hist_start+num_states-2].
        # For padded (num_states=1): history has 1 frame, _prepare doesn't slice
        # → offline position [idx:idx+1].
        if padded:
            act_start, act_end = idx, idx + 1
        else:
            act_start, act_end = hist_start, hist_start + num_states - 1
        expected_act = torch.nn.functional.one_hot(
            offline_data['activity'][act_start:act_end].long(),
            num_classes=n_act).float().unsqueeze(0)
        _eq('activity_seq', activity_seq, expected_act)

        # Expected time_context: _prepare slices history[:, 1:num_states, :]
        # → offline positions [hist_start+1 .. idx] = [hist_start+1 .. hist_start+num_states-1].
        # For padded (num_states=1): _prepare uses full 1-frame history
        # → offline position [idx:idx+1].
        _T_FREQ = torch.tensor([10, 30, 60, 60*3, 60*6, 60*12, 60*24], dtype=torch.float32)
        if padded:
            t_range = offline_times[idx:idx+1]
        else:
            t_range = offline_times[hist_start + 1 : hist_start + num_states]
        omegas = 2 * _math.pi / _T_FREQ
        ts = t_range.float().unsqueeze(-1) * omegas
        tf = torch.cat([torch.sin(ts), torch.cos(ts)], dim=-1)
        if tf.size(-1) < c_len:
            pad = torch.zeros(tf.size(0), c_len - tf.size(-1))
            tf = torch.cat([tf, pad], dim=-1)
        expected_time = tf.unsqueeze(0)
        _eq('time_context', time_context, expected_time)
        self._last_fwd = None

    def _find_checkpoint_only_cfg(self, checkpoint_dir: str) -> tuple[str, str]:
        """Resolve *checkpoint_dir* to a (.ckpt path, run directory) pair for config discovery.

        The returned ``.ckpt`` path is used only to locate ``common_data.json`` and
        ``model_configs*.json`` in the run tree — tensor weights are **not** read here.
        Sim-trained weights are loaded later via ``train()`` / ``closed_loop_eval`` resume.

        Handles:
        * Direct path to a ``.ckpt`` file
        * A run directory containing ``.ckpt`` files
        * A root directory → descend into household subdir → run subdir
        """
        requested = os.path.abspath(checkpoint_dir)
        # Direct path to a .ckpt file
        if os.path.isfile(requested) and requested.endswith('.ckpt'):
            return requested, os.path.dirname(requested)

        run_dir = requested
        # Try lowercase variant for case-sensitive filesystems
        if not os.path.isdir(run_dir):
            base, name = os.path.split(run_dir.rstrip(os.sep))
            if name and name != name.lower():
                run_dir = os.path.join(base, name.lower())
        if not os.path.isdir(run_dir):
            _raise_ckpt_not_found(requested)

        # Descend into household-specific subdir if needed
        if self.household_name is not None:
            subdirs = [d for d in os.listdir(run_dir)
                       if os.path.isdir(os.path.join(run_dir, d))]
            household_lower = self.household_name.lower()
            for sub in sorted(subdirs):
                if household_lower in sub.lower():
                    run_dir = os.path.join(run_dir, sub)
                    break

        # If run_dir has no .ckpt, look one level deeper (e.g. householdB/default_100)
        ckpt_files = [f for f in os.listdir(run_dir) if f.endswith('.ckpt')]
        if not ckpt_files:
            assert 'default_100' in os.listdir(run_dir), f"No default_100 directory found in {run_dir}"
            run_dir = os.path.join(run_dir, 'default_100')
            if os.path.isdir(run_dir):
                ckpt_files = [f for f in os.listdir(run_dir) if f.endswith('.ckpt')]

        # Prefer 'best' checkpoint, else latest by name, else first
        best = [f for f in ckpt_files if 'best' in f.lower()]
        if best:
            chosen = best[0]
        else:
            ckpt_files.sort(reverse=True)
            chosen = ckpt_files[0]

        return os.path.join(run_dir, chosen), run_dir

    def _load_slatepro_config_only_cfg(self, run_dir: str) -> 'adict':
        """Load SLaTe-PRO YAML/JSON config only (no tensor weights)."""
        import yaml
        default_config_path = os.path.join(SLATEPRO_ROOT, 'config', 'default.yaml')
        with open(default_config_path) as f:
            model_configs = adict(yaml.safe_load(f))
        config_file = os.path.join(run_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file) as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k != 'DATA_INFO' and not callable(v):
                    model_configs[k] = v
        return model_configs

    def _load_model_only_cfg(self, checkpoint_dir: str, household: str):
        """Build ``MultiModalUserTrackingModule`` from checkpoint-dir config only.

        Reads ``common_data.json`` and ``model_configs*.json`` beside the resolved
        ``.ckpt`` file. Does **not** load HOMER+ or Lightning tensor weights —
        the module starts with random init. Sim-trained weights are loaded by
        ``train()`` (``weights_day*.pt``) or ``closed_loop_eval`` resume paths.
        """
        checkpoint_path, run_dir = self._find_checkpoint_only_cfg(checkpoint_dir)
        print(f'[SLaTe-PRO] Initializing model architecture from checkpoint config: '
              f'{checkpoint_path} (weights not loaded)')

        # Load common_data and set all model dimensions / data-derived keys (required at __init__)
        common_data_path = '/'.join(checkpoint_path.split('/')[:-2]) + '/common_data.json'
        print(f"Loading common data from: {common_data_path}")
        self.common_data = load_common_data(common_data_path)
        n_node_classes = len(self.common_data['node_classes'])   # vocabulary size (e.g. 93)
        n_nodes        = self.common_data['num_nodes']             # graph size (e.g. 81)
        n_activities   = len(self.common_data['activities'])
        
        # Load model
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cfg_file = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.json') and 'model_configs' in f], reverse=True)[0]
        cfg_path = os.path.join(checkpoint_dir, cfg_file)
        ckpt_config = json.load(open(cfg_path))
        assert ckpt_config['n_nodes'] == n_nodes, f"Expected {n_nodes} nodes, got {ckpt_config['n_nodes']}"
        assert ckpt_config['n_len'] == n_node_classes, f"Expected {n_node_classes} node classes, got {ckpt_config['n_len']}"
        assert ckpt_config['n_activities'] == n_activities, f"Expected {n_activities} activities, got {ckpt_config['n_activities']}"
        # load_common_data() augments common_data with runtime-derived keys
        # (vh_node_ids, n_nodes) for live-sim node filtering; these post-date the
        # frozen checkpoint DATA_INFO and are derived from node_ids (already
        # compared). Exclude them from the data-config identity check rather than
        # failing on keys the checkpoint could not have stored. Still raises on
        # genuine vocab/node/activity drift.
        _runtime_augmented = ('vh_node_ids', 'n_nodes')
        cd_core = {k: v for k, v in self.common_data.items()
                   if k not in _runtime_augmented}
        if not dictcompare(cd_core, ckpt_config['DATA_INFO']):
            raise ValueError('Data configs do not match')
        
        timestr = time.strftime("%m%d_%H%M%S")
        ckpt_config["logs_dir"] = f"./logs/SLaTe-PRO_{timestr}"
        self.model_configs = adict(ckpt_config)
        self.model = MultiModalUserTrackingModule(
            model_configs=self.model_configs,
            original_model=ckpt_config.get('original_model', False),
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Set object consistency (required by model) — shape (n_nodes,)
        obj_consistency = torch.zeros(n_nodes, dtype=torch.long)
        self.model.set_object_consistency(obj_consistency)
        self.model.cfg.query_types = []  # Disable query logic for inference

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _update_history(self, world_state: dict) -> None:
        """Append *world_state* to the accumulated trajectory history.

        Filters the world state to the HOMER+ node set first (no-op when the
        world state is already the right size, e.g. in eval-script mode).
        """
        
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

    def get_history(self):
        graph_seq_nodes, graph_seq_edges, graph_dyn_edges, activity_seq, time_context = self._prepare_history_tensors(self.history, self.history['sequence_length'])
        return {
            'graph_seq_nodes': graph_seq_nodes,
            'graph_seq_edges': graph_seq_edges,
            'graph_dyn_edges': graph_dyn_edges,
            'activity_seq': activity_seq,
            'time_context': time_context,
            'times':torch.tensor(self.history['times']).to(self.device)
        }

    def _prepare_history_tensors(self, batch: dict, num_states: int):
        """Slice and pad the accumulated history into model-ready tensors.

        Handles three cases:
        * ``num_states < 2``             – pad to 2 (GraphEncoder minimum)
        * ``num_states > max_history_len`` – trim to the most recent window
        * otherwise                      – use as-is

        Returns
        -------
        tuple of (graph_seq_nodes, graph_seq_edges, graph_dyn_edges,
                  activity_seq, time_context), all on ``self.device``.
        """
        if num_states < 2:
            graph_seq_nodes = batch['node_features'].repeat(1, 2, 1, 1).to(self.device)
            graph_seq_edges = batch['edges'].repeat(1, 2, 1, 1).to(self.device)
            graph_dyn_edges = batch['dynamic_edges_mask'].repeat(1, 2, 1, 1).to(self.device)
            activity_seq    = batch['activity_features'].to(self.device)
            time_context    = batch['time_features'].to(self.device)
        elif num_states > self.max_history_len:
            graph_seq_nodes = batch['node_features'].repeat(1, self.max_history_len, 1, 1).to(self.device)
            graph_seq_edges = batch['edges'][:, -self.max_history_len:, :, :].to(self.device)
            graph_dyn_edges = batch['dynamic_edges_mask'].repeat(1, self.max_history_len, 1, 1).to(self.device)
            activity_seq    = batch['activity_features'][:, -self.max_history_len:-1, :].to(self.device)
            # time_context: training/paper eval uses batch_time[:,1:,:] (shift by +1).
            # For max-history case, shift the window by 1 to match that convention.
            time_context    = batch['time_features'][:, -self.max_history_len+1:, :].to(self.device)
        else:
            graph_seq_nodes = batch['node_features'].repeat(1, num_states, 1, 1).to(self.device)
            graph_seq_edges = batch['edges'].to(self.device)
            graph_dyn_edges = batch['dynamic_edges_mask'].repeat(1, num_states, 1, 1).to(self.device)
            activity_seq    = batch['activity_features'][:, :num_states - 1, :].to(self.device)
            # Training and paper eval use time_context = batch_time[:,1:,:] (shifted forward by 1
            # so that latent[t] for the t→t+1 edge transition is paired with time[t+1]).
            # We mirror that by slicing time_features[1..num_states-1] instead of [0..num_states-2].
            time_context    = batch['time_features'][:, 1:num_states, :].to(self.device)
        return graph_seq_nodes, graph_seq_edges, graph_dyn_edges, activity_seq, time_context

    def _run_forward(self) -> dict:
        """Run the SLaTe-PRO model on the current history.

        When ``lookahead_steps > 1``, performs autoregressive multi-step
        prediction matching the paper's ``evaluate_prediction`` loop:
        feed predicted edges back as input, track the best (highest
        confidence) predicted destination per object across all steps.

        Returns a dict with raw prediction tensors (all on CPU, no_grad):

        * ``pred_edges``           (n_nodes, n_nodes) – predicted next-state softmax
        * ``current_edges``        (n_nodes, n_nodes) – current-state distribution
        * ``dyn_mask_2d``          (n_nodes, n_nodes) – dynamic-edge mask
        * ``pred_activity``        (n_activities,)    – next-activity softmax
        * ``current_locations``    (n_nodes,)         – argmax of current_edges
        * ``pred_locations``       (n_nodes,)         – argmax of pred_edges (best across steps)
        * ``dyn_mask_flat``        (n_nodes,)         – True for movable objects
        * ``relocation_confidences`` (n_nodes,)       – predicted-move confidence (best across steps)
        * ``valid_relocations``    (n_nodes,)         – above threshold & dynamic
        """
        batch = self.history
        num_states = batch['edges'].size(1)
        graph_seq_nodes, graph_seq_edges, graph_dyn_edges, activity_seq, time_context = \
            self._prepare_history_tensors(batch, num_states)

        last_idx = graph_seq_edges.size(1) - 1
        num_steps = self.lookahead_steps

        # TEMP: probe inputs right at the model callsite against offline_data.
        # `_debug_offline_data` and `_debug_dump_path` can be set externally.
        _probe = getattr(self, '_debug_offline_data', None)
        _dump_path = getattr(self, '_debug_dump_path', None)
        if _probe is not None:
            self._probe_model_inputs_vs_offline(
                graph_seq_nodes, graph_seq_edges, graph_dyn_edges,
                activity_seq, time_context, _probe)

        with torch.no_grad():
            self.model.eval()
            if self.model.original_model:
                raise NotImplementedError('Original model not implemented in RobotPolicySlatepro')

            graph_latents, _, _ = self.model.object_activity_coembedding_module.autoencode_graph(
                graph_seq_nodes, graph_seq_edges, graph_dyn_edges, time_context=time_context)
            activity_latents, _, _ = self.model.object_activity_coembedding_module.autoencode_activity(
                activity_seq, time_context=time_context)

            # TEMP: one-shot dump of live model callsite state.
            if _dump_path and not getattr(self, '_dumped_run_forward', False):
                import os as _os
                _os.makedirs(_os.path.dirname(_dump_path) or '.', exist_ok=True)
                torch.save({
                    'inputs': {
                        'graph_seq_nodes': graph_seq_nodes.cpu(),
                        'graph_seq_edges': graph_seq_edges.cpu(),
                        'graph_dyn_edges': graph_dyn_edges.cpu(),
                        'activity_seq':    activity_seq.cpu(),
                        'time_context':    time_context.cpu(),
                    },
                    'latents': {
                        'graph_latents':    graph_latents.latent.cpu(),
                        'activity_latents': activity_latents.latent.cpu(),
                    },
                    'history_times':  list(self.history['times']),
                    'num_states':     num_states,
                }, _dump_path)
                print(f"[LIVE DUMP] saved to {_dump_path}")
                self._dumped_run_forward = True

            all_latents = graph_latents + activity_latents   # (1, seq_len, emb)

            input_edges_fwd = graph_seq_edges[:, last_idx : last_idx + 1, :, :]
            input_nodes     = graph_seq_nodes[:, last_idx : last_idx + 1, :, :]
            dyn_mask        = graph_dyn_edges[:, last_idx : last_idx + 1, :, :]

            current_edges_2d = graph_seq_edges[0, last_idx].cpu()
            dyn_mask_2d      = graph_dyn_edges[0, last_idx].cpu()
            current_locations = current_edges_2d.argmax(dim=-1)
            dyn_mask_flat     = dyn_mask_2d.sum(dim=-1) > 0
            reference_edges   = current_locations  # (n_nodes,) — argmax of current state

            # Accumulators for best prediction across steps (like paper lines 669-734)
            n_nodes = current_edges_2d.size(0)
            best_dest        = None       # (n_nodes,) best predicted destination
            best_confidence  = torch.zeros(n_nodes)  # best confidence per object
            changes_pred     = torch.zeros(n_nodes, dtype=torch.bool)  # any step predicted movement
            pred_activity_1d = None

            latents_forward = all_latents  # will be updated by predict() each step
            seq_len = time_context.size(1)
            c_len = time_context.size(-1)
            # Paper evaluate_prediction shifts time_context by `step` each lookahead
            # iteration. Closed-loop history ends at "now", so append synthetic future
            # encodings (+10 min per step) for steps 1..num_steps-1.
            if num_steps > 1:
                t_now = self.history['times'][-1]
                future = [
                    time_encoder(
                        t_now + self.time_step_minutes * (k + 1),
                        target_len=c_len,
                    ).to(self.device)
                    for k in range(num_steps - 1)
                ]
                extended_tc = torch.cat(
                    [time_context, torch.stack(future, dim=0).unsqueeze(0)], dim=1)
            else:
                extended_tc = time_context

            for step in range(num_steps):
                # Paper: predict(..., time_context[:, step:pred_seq_len+step])
                tc_pred = extended_tc[:, step:step + seq_len, :]
                latents_forward, _ = self.model.predict(latents_forward, tc_pred)

                # Paper: decode with time_context[:, step+1:pred_seq_len+step+1][-1]
                latents   = latents_forward[:, -1:, :]
                tc        = extended_tc[:, step + seq_len - 1:step + seq_len, :]
                latent_in = latents + tc if self.model.cfg.addtnl_time_context else latents

                pred_edges_raw, _, _ = self.model.object_activity_coembedding_module.decode_graph(
                    latents=latent_in, input_nodes=input_nodes,
                    input_edges=input_edges_fwd, dynamic_edges_mask=dyn_mask)

                if pred_activity_1d is None:
                    pred_activity_raw, _, _ = self.model.object_activity_coembedding_module.decode_activity(
                        latents=latent_in)
                    pred_activity_1d = pred_activity_raw.squeeze().cpu()

                pred_edges_2d_step = pred_edges_raw.squeeze(0).squeeze(0).cpu()
                pred_locs_step     = pred_edges_2d_step.argmax(dim=-1)

                if step == 0:
                    # First step: initialise dest_pred and changes_pred (paper line 720-726)
                    best_dest    = pred_locs_step.clone()
                    changes_pred = (best_dest != reference_edges) & dyn_mask_flat
                    new_changes  = changes_pred
                else:
                    # Subsequent steps: track newly changed objects (paper lines 711-714)
                    new_changes = (pred_locs_step != reference_edges) & dyn_mask_flat & ~changes_pred
                    # Update dest for newly changed objects
                    best_dest[new_changes] = pred_locs_step[new_changes]
                    changes_pred = changes_pred | new_changes

                # Track best confidence per object (paper lines 730-734)
                confidence_in_relocation = (pred_edges_2d_step * torch.nn.functional.one_hot(
                    best_dest, num_classes=pred_edges_2d_step.size(-1)).float()).sum(dim=-1)
                # "continued" = this step still predicts the same dest as best_dest
                continued = (pred_locs_step == best_dest)
                better = ((confidence_in_relocation > best_confidence) & continued) | new_changes
                best_confidence[better] = confidence_in_relocation[better]

                # Feed predicted edges back as input for next step (paper line 807)
                input_edges_fwd = pred_edges_raw.clone().detach()

            # Final output: use best across all steps
            pred_locations         = best_dest
            relocation_confidences = best_confidence
            relocations            = changes_pred
            valid_relocations      = relocations & (relocation_confidences >= self.confidence_threshold)

            # Use the last step's pred_edges for the return dict
            pred_edges_2d = pred_edges_2d_step

        return {
            'pred_edges':             pred_edges_2d,
            'current_edges':          current_edges_2d,
            'dyn_mask_2d':            dyn_mask_2d,
            'pred_activity':          pred_activity_1d,
            'current_locations':      current_locations,
            'pred_locations':         pred_locations,
            'dyn_mask_flat':          dyn_mask_flat,
            'relocation_confidences': relocation_confidences,
            'relocations':            relocations,       # raw changes_pred (before threshold)
            'valid_relocations':      valid_relocations,
        }

    # ------------------------------------------------------------------
    # Public action / prediction APIs
    # ------------------------------------------------------------------

    def get_prediction_distribution(self, time_step: int, node_map, *,
                                    world_state=None) -> dict | None:
        """Return raw model predictions for evaluation (no action selected).

        Updates the internal history exactly like ``get_actions`` does, then
        returns the full ``_run_forward()`` dict instead of converting to a
        ``RobotMoveAction``.  Use this in evaluation scripts to compute
        precision / recall without side-effects on the world.

        Returns ``None`` if the model is not loaded.
        """
        if world_state is None:
            raise ValueError('world_state is required for SLaTe-PRO policy')
        if self.model is None:
            return None
        self._update_history(world_state)
        return self._run_forward()

    def get_actions(self, node_map, *, world_state: dict) -> 'RobotMoveAction | list[RobotMoveAction] | None':
        """Update history, run forward pass, select and return the best action."""
        self._update_history(world_state)

        fwd = self._run_forward()
        self._last_fwd = fwd

        valid_relocations      = fwd['valid_relocations']
        relocation_confidences = fwd['relocation_confidences']
        pred_locations         = fwd['pred_locations']
        current_locations      = fwd['current_locations']

        if not valid_relocations.any():
            return [], ([], [], [])

        # Log top relocations at DEBUG level every 10 timesteps
        if _log.isEnabledFor(logging.DEBUG):
            node_id_to_class = self.history.get('node_id_to_class', {})
            slate_to_vh_dbg = {v: k for k, v in self.history['vh_id_to_slate_idx'].items()}
            for obj_idx in torch.where(valid_relocations)[0][:10]:
                obj_class      = node_id_to_class.get(slate_to_vh_dbg.get(obj_idx.item()), '?')
                cur_loc_class  = node_id_to_class.get(slate_to_vh_dbg.get(current_locations[obj_idx].item()), '?')
                pred_loc_class = node_id_to_class.get(slate_to_vh_dbg.get(pred_locations[obj_idx].item()), '?')
                _log.debug('[t=%d] %s: current=%s -> predicted=%s (conf=%.3f)',
                           world_state['time'], obj_class, cur_loc_class, pred_loc_class,
                           relocation_confidences[obj_idx])

        relocation_indices = torch.where(valid_relocations)[0]
        confidences        = relocation_confidences[relocation_indices]

        # Invert slate_idx → VH node ID using the history mapping so we can
        # look up nodes by exact ID rather than fuzzy class-name matching.
        slate_to_vh = {v: k for k, v in self.history['vh_id_to_slate_idx'].items()}

        actions = []
        # Sort by confidence (highest first) for deterministic ordering
        sorted_order = confidences.argsort(descending=True)
        obj_idxs = []
        loc_idxs = []
        prev_loc_idxs = []

        for rank in sorted_order:
            idx = relocation_indices[rank]
            action = self._slate_action_to_vh_action(
                idx.item(), pred_locations[idx].item(),
                current_locations[idx].item(),
                world_state, node_map, slate_to_vh=slate_to_vh)
            if action is not None:
                actions.append(action)
                obj_idxs.append(idx.item())
                loc_idxs.append(pred_locations[idx].item())
                prev_loc_idxs.append(current_locations[idx].item())

        # Emit every above-threshold action (multiple-action behavior is the
        # only mode used; hardcoded rather than configurable).
        return actions, (obj_idxs, loc_idxs, prev_loc_idxs)


    def _slate_action_to_vh_action(
        self, slate_obj_idx, slate_loc_idx, slate_current_idx,
        world_state, node_map, slate_to_vh: dict | None = None,
    ) -> RobotMoveAction:
        """Convert SLaTe-PRO action to VirtualHome action.

        slate_obj_idx, slate_loc_idx, and slate_current_idx are LOCAL node
        indices (0..n_nodes-1).  Uses the inverted slate_idx → VH node ID
        mapping (``slate_to_vh``) to convert to VH IDs and class names.
        """
        if slate_to_vh is None:
            slate_to_vh = {v: k for k, v in self.history['vh_id_to_slate_idx'].items()}

        node_id_to_class = self.history.get('node_id_to_class', {})

        obj_vh_id = slate_to_vh.get(slate_obj_idx)
        loc_vh_id = slate_to_vh.get(slate_loc_idx)
        if obj_vh_id is None or loc_vh_id is None:
            return None

        obj_class = node_id_to_class.get(obj_vh_id, 'unknown')
        loc_class = node_id_to_class.get(loc_vh_id, 'unknown')

        return RobotMoveAction(obj_class, obj_vh_id, loc_class, loc_vh_id)

    def _setup_model_for_inference(self):
        """Post-training setup: eval mode, device, object consistency, disable queries."""
        self.model.eval()
        self.model.to(self.device)
        n_nodes = self.common_data['num_nodes']
        obj_consistency = torch.zeros(n_nodes, dtype=torch.long)
        self.model.set_object_consistency(obj_consistency)
        self.model.cfg.query_types = []

    def train(self, dataset_path: str, output_dir: str, days: int) -> None:
        """Train the model on the given dataset, with checkpoint resumability."""
        from pytorch_lightning.callbacks import ModelCheckpoint

        # Per-round checkpoint directory and done marker
        ckpt_dir = os.path.join(output_dir, f'pl_checkpoints_day{days}')
        os.makedirs(ckpt_dir, exist_ok=True)
        done_marker = os.path.join(ckpt_dir, 'training_done')
        weights_path = os.path.join(output_dir, f'weights_day{days}.pt')

        # Skip training if this round already completed (resumed job)
        if os.path.isfile(done_marker) and os.path.isfile(weights_path):
            print(f"Training for day {days} already complete, loading weights.")
            self.model = MultiModalUserTrackingModule(model_configs=self.model_configs, original_model=False)
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self._setup_model_for_inference()
            self.data_quantity = {'train': -1, 'val': -1}
            return

        data = RoutinesDataset(data_path=dataset_path)
        train_loader = data.get_train_loader()
        val_loader = data.get_val_loader()
        self.data_quantity = {
            'train': len(train_loader) if train_loader is not None else -1,
            'val': len(val_loader) if val_loader is not None else -1,
        }

        # Check for PL checkpoint to resume from
        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        resume_ckpt = last_ckpt if os.path.isfile(last_ckpt) else None
        # Cap fresh training at MAX_TRAIN_EPOCHS (default 450); for resumed
        # continual rounds, allow +200 more past the resumed epoch (early_stopping
        # still gates the real stop). Lower it (e.g. MAX_TRAIN_EPOCHS=100) for
        # quick smoke runs that just need to reach the finetune/LLM phase fast.
        base_max_epochs = int(os.environ.get('MAX_TRAIN_EPOCHS', '450'))
        max_epochs = base_max_epochs
        if resume_ckpt:
            resumed_epoch = int(torch.load(resume_ckpt, map_location='cpu', weights_only=False).get('epoch', 0))
            max_epochs = max(base_max_epochs, resumed_epoch + 200)
            print(f"Resuming training from {resume_ckpt} (epoch={resumed_epoch}, max_epochs={max_epochs})")

        self.model = MultiModalUserTrackingModule(model_configs=self.model_configs, original_model=False)
        ckpt_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, every_n_epochs=10)
        early_stop_callback = EarlyStopping(monitor="Val_ES_accuracy", patience=40, verbose=False, mode="max")

        trainer = Trainer(accelerator='gpu', devices=torch.cuda.device_count(), logger=self.logger, log_every_n_steps=1, max_epochs=max_epochs,
                          callbacks=[ckpt_callback, early_stop_callback], check_val_every_n_epoch=5)
        print(f"Training on {len(train_loader)} training examples and {len(val_loader)} validation examples")
        trainer.fit(self.model, train_loader, val_loader, ckpt_path=resume_ckpt)

        # Save final weights
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), weights_path)
        # torch.save(self.model.state_dict(), os.path.join(output_dir, 'weights.pt'))

        # Mark training complete so resumed jobs skip this round
        with open(done_marker, 'w') as f:
            f.write('done\n')

        metrics = trainer.callback_metrics
        print("Training complete. Validation metrics:")
        for key, value in sorted(metrics.items()):
            if key.startswith('Val'):
                print(f"  {key}: {value:.4f}")

        self._setup_model_for_inference()