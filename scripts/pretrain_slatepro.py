#!/usr/bin/env python
"""Pre-train a SLaTe-PRO model on processed HOMER+ data.

This script wraps the training logic from ``external/SLaTe-PRO/run.py`` into
a convenient CLI for the co-adaptation simulation project.  It trains a model
for a given household (A, B, or C) and saves checkpoints for later use by
``RobotPolicySlatepro``.

Usage
-----
    python scripts/pretrain_slatepro.py --household B --output_dir checkpoints/slate_pro/

Optional:
    --epochs          Override number of training epochs (default: from config)
    --batch_size      Override batch size (default: from config)
    --config          Path to SLaTe-PRO YAML config (default: external/SLaTe-PRO/config/default.yaml)
    --activity_avail  Activity availability percentage (default: 100 → no dropout)
    --train_days      Limit number of training routines (default: all)
    --phased          Use phased training (default: false)
    --original_model  Use the original SLaTE model (default: false)
    --name            Run name for logging/output dirs (default: "default_100")
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Path setup: make sure SLaTe-PRO modules are importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SLATEPRO_ROOT = os.path.join(PROJECT_ROOT, 'external', 'SLaTe-PRO')

sys.path.insert(0, SLATEPRO_ROOT)
sys.path.insert(0, os.path.join(SLATEPRO_ROOT, 'helpers'))
sys.path.insert(0, os.path.join(SLATEPRO_ROOT, 'models'))

import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

try:
    import wandb
except ImportError:
    wandb = None

from adict import adict
from MultiModalUserTracking import MultiModalUserTrackingModule
from loader_sequential import RoutinesDataset
from encoders import TimeEncodingOptions

import random
from numpy import random as nrandom

# Reproducibility
SEED = 23435
random.seed(SEED)
nrandom.seed(SEED)
torch.manual_seed(SEED)


HOUSEHOLD_MAP = {
    'A': 'householdA',
    'B': 'householdB',
    'C': 'householdC',
}


def resolve_data_path(household: str) -> str:
    """Return the absolute path to the processed SLaTe-PRO data directory."""
    hh_dir = HOUSEHOLD_MAP.get(household.upper())
    if hh_dir is None:
        raise ValueError(f'Unknown household "{household}". Choose from A, B, C.')
    path = os.path.join(SLATEPRO_ROOT, 'data', 'HOMER+', hh_dir, 'processed_seqLM_coarse')
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Data directory not found: {path}')
    return path


def train(data_path: str, cfg: dict, output_dir: str,
          original_model: bool = False, train_days: int | None = None):
    """Train a SLaTe-PRO model and save the checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    common_data_path = os.path.join(data_path, 'common_data.json')
    with open(common_data_path) as f:
        data_info = json.load(f)

    cfg['DATA_INFO'] = data_info
    cfg['MAX_TRAINING_SAMPLES'] = train_days

    weekend_days = data_info.get('weekend_days') or data_info.get('weeekend_days', None)
    time_options = TimeEncodingOptions(weekend_days)
    time_encoding = time_options('sine_informed')

    data = RoutinesDataset(
        data_path=data_path,
        time_encoder=time_encoding,
        batch_size=cfg['batch_size'],
        use_bert=False,
        activity_dropout=cfg.get('activity_dropout_prob', 0.0),
    )

    cfg.update(data.params)
    model_configs = adict(cfg)

    logger_mode = 'disabled'
    if wandb is not None:
        try:
            wandb_logger = WandbLogger(
                name=cfg.get('NAME', 'pretrain'),
                group=os.path.basename(os.path.dirname(data_path)),
                tags=['pretrain'],
                settings=wandb.Settings(start_method='fork'),
                mode=logger_mode,
            )
            wandb_logger.experiment.config.update(cfg)
        except Exception:
            wandb_logger = None
    else:
        wandb_logger = None

    epochs = cfg.get('epochs', 1000)
    print(f'Training for up to {epochs} epochs ...')
    print(f'  Data:   {data_path}')
    print(f'  Output: {output_dir}')
    print(f'  Nodes:  {model_configs.get("n_nodes", "?")}')
    print(f'  Batch:  {cfg["batch_size"]}')

    if model_configs.get('phased_training', False):
        phase1_dir = output_dir + '_pretr'
        phase1_done_marker = os.path.join(phase1_dir, 'phase1_done')

        if not os.path.isfile(phase1_done_marker):
            print('\n--- Phase 1: Autoencoder pre-training ---')
            os.makedirs(phase1_dir, exist_ok=True)
            ckpt_callback = ModelCheckpoint(dirpath=phase1_dir, save_last=True)
            early_stop = EarlyStopping(
                monitor='Val_ES_accuracy', patience=40, verbose=False, mode='max',
            )
            model_configs.loss_latent_pred = False
            model_configs.loss_object_pred = False
            model_configs.loss_activity_pred = False

            trainer = Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=max(torch.cuda.device_count(), 1),
                logger=wandb_logger,
                max_epochs=epochs,
                log_every_n_steps=1,
                callbacks=[ckpt_callback, early_stop],
                check_val_every_n_epoch=5,
            )
            model = MultiModalUserTrackingModule(
                model_configs=model_configs, original_model=original_model,
            )
            model.set_object_consistency(data.get_object_consistency())
            model.cfg.query_types = []

            last_ckpt_p1 = os.path.join(phase1_dir, 'last.ckpt')
            resume_p1 = last_ckpt_p1 if os.path.isfile(last_ckpt_p1) else None
            if resume_p1:
                print(f'  Resuming Phase 1 from {resume_p1}')
            trainer.fit(model, data.get_train_loader(), data.get_val_loader(),
                        ckpt_path=resume_p1)

            with open(phase1_done_marker, 'w') as f:
                f.write('done\n')
        else:
            print('\n--- Phase 1 already complete, skipping ---')

        model_configs.loss_latent_pred = True
        model_configs.loss_object_pred = True
        model_configs.loss_activity_pred = True
        print('\n--- Phase 2: Full training ---')

    ckpt_callback = ModelCheckpoint(dirpath=output_dir, save_last=True)
    early_stop = EarlyStopping(
        monitor='Val_ES_accuracy', patience=40, verbose=False, mode='max',
    )

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=max(torch.cuda.device_count(), 1),
        logger=wandb_logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        callbacks=[ckpt_callback, early_stop],
        check_val_every_n_epoch=5,
    )
    model = MultiModalUserTrackingModule(
        model_configs=model_configs, original_model=original_model,
    )
    model.set_object_consistency(data.get_object_consistency())
    model.cfg.query_types = []

    last_ckpt = os.path.join(output_dir, 'last.ckpt')
    resume_path = last_ckpt if os.path.isfile(last_ckpt) else None
    if resume_path:
        print(f'Resuming from {resume_path}')
    trainer.fit(model, data.get_train_loader(), data.get_val_loader(),
                ckpt_path=resume_path)

    weights_path = os.path.join(output_dir, 'weights.pt')
    torch.save(model.state_dict(), weights_path)

    config_out = {k: v for k, v in model_configs.items()
                  if not callable(v) and k != 'DATA_INFO'}
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2, default=str)

    print(f'\nTraining complete!')
    print(f'  Checkpoint: {output_dir}/ (*.ckpt)')
    print(f'  Weights:    {weights_path}')
    print(f'  Config:     {config_path}')
    return model, output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Pre-train a SLaTe-PRO model on HOMER+ data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--household', type=str, required=True, choices=['A', 'B', 'C'],
        help='Household to train on (A, B, or C).',
    )
    parser.add_argument(
        '--output_dir', type=str, default='checkpoints/slate_pro/',
        help='Base directory for saving checkpoints (default: checkpoints/slate_pro/).',
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to SLaTe-PRO YAML config (default: external/SLaTe-PRO/config/default.yaml).',
    )
    parser.add_argument('--name', type=str, default='default_100', help='Run name.')
    parser.add_argument('--epochs', type=int, default=None, help='Max training epochs.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    parser.add_argument('--train_days', type=int, default=None,
                        help='Limit training routines (default: all).')
    parser.add_argument('--activity_avail', type=int, default=100,
                        help='Activity availability %% (100 = no dropout).')
    parser.add_argument('--phased', action='store_true',
                        help='Use phased training (AE pre-training then full).')
    parser.add_argument('--original_model', action='store_true',
                        help='Use original SLaTE architecture.')
    args = parser.parse_args()

    config_path = args.config or os.path.join(SLATEPRO_ROOT, 'config', 'default.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg['NAME'] = args.name
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    cfg['activity_dropout_prob'] = 1.0 - (args.activity_avail / 100.0)
    cfg['phased_training'] = args.phased

    data_path = resolve_data_path(args.household)
    output_dir = os.path.join(
        args.output_dir,
        HOUSEHOLD_MAP[args.household.upper()],
        args.name,
    )

    print(f'=== SLaTe-PRO Pre-training ===')
    print(f'  Household: {args.household}')
    print(f'  Data path: {data_path}')
    print(f'  Output:    {output_dir}')
    print()

    torch.cuda.empty_cache()
    train(
        data_path=data_path,
        cfg=cfg,
        output_dir=output_dir,
        original_model=args.original_model,
        train_days=args.train_days,
    )


if __name__ == '__main__':
    main()
