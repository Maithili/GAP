"""Wiring/overfit checks for model/activity_transformer.py on synthetic data.
Tiny CPU run (seconds):

    MAX_TRAIN_EPOCHS=150 python \
        tests/test_activity_transformer.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from model.activity_transformer import (
    train_activity_transformer, load_activity_transformer, build_samples,
    ActivityTransformer)

VOCAB = ['breakfast', 'computer_work', 'dinner', 'wash_dishes']


def make_day(day_idx):
    """Deterministic routine: breakfast -> computer_work -> dinner ->
    wash_dishes, with slight time jitter by day index."""
    j = (day_idx % 3) * 10
    segs = [
        {'activity_name': 'breakfast', 'start_min': 480 + j, 'end_min': 520 + j},
        {'activity_name': 'computer_work', 'start_min': 560 + j, 'end_min': 900 + j},
        {'activity_name': 'dinner', 'start_min': 1080 + j, 'end_min': 1140 + j},
        {'activity_name': 'wash_dishes', 'start_min': 1150 + j, 'end_min': 1180 + j},
    ]
    return {'day': day_idx, 'segments': segs, 'events': []}


def fake_embeddings():
    g = torch.Generator().manual_seed(0)
    return {'model': 'fake-embed', 'dim': 32,
            'vectors': {n: torch.randn(32, generator=g).tolist()
                        for n in VOCAB}}


def run_mode(mode):
    days = [make_day(i) for i in range(12)]
    cfg = {'mode': mode, 'd_proj': 16, 'd_model': 32, 'n_heads': 2,
           'n_layers': 2, 'dim_feedforward': 64, 'dropout': 0.0,
           'lr': 3e-3, 'val_days': 3, 'time_loss_weight': 0.5,
           'early_stop_patience': 100}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ckpt.pt')
        torch.manual_seed(0)
        metrics = train_activity_transformer(days, fake_embeddings(), cfg,
                                             path, device='cpu')
        model = load_activity_transformer(path)
    # deterministic routine must be learnable nearly perfectly
    assert metrics['val_top1'] > 0.9, f"{mode}: val_top1={metrics['val_top1']}"
    # predict_next on a partial day: after breakfast, computer_work is next
    partial = [{'activity_name': 'breakfast', 'start_min': 480, 'end_min': 520}]
    probs, gaps = model.predict_next(partial, now_min=530)
    real = {k: v for k, v in probs.items() if k != '__idle__'}
    assert max(real, key=real.get) == 'computer_work', f"{mode}: {probs}"
    # gaps is per-activity; computer_work starts 560-580 -> gap well under 200
    g = gaps.get('computer_work')
    assert g is not None and 0 <= g < 200, f"{mode}: gaps={gaps}"
    print(f"ok: {mode} (val_top1={metrics['val_top1']:.3f}, "
          f"top_pred=computer_work, gap={g})")


def test_sample_shapes():
    days = [make_day(0)]
    ev = build_samples(days, VOCAB, 'event')[0]
    assert len(ev['act']) == 4 and ev['act'][0] == len(VOCAB) + 1  # BOS first
    assert ev['target'][0] == VOCAB.index('breakfast')
    iv = build_samples(days, VOCAB, 'interval')[0]
    assert (iv['target'] != iv['act']).sum() > 0  # change ticks exist
    assert (iv['weight'] == 5.0).sum() == (iv['target'] != iv['act']).sum()
    print('ok: sample_shapes')


if __name__ == '__main__':
    os.environ.setdefault('MAX_TRAIN_EPOCHS', '150')
    test_sample_shapes()
    run_mode('event')
    run_mode('interval')
    print('ALL activity_transformer checks passed')
