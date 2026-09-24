"""
Visualize benchmark comparison results.

Reads the JSON output from benchmark_comparison.py and generates plots
comparing SLaTe-PRO vs Continual model performance.

Usage:
  python scripts/visualize_comparison.py \
      --results results/benchmark/benchmark_A.json \
      --out_dir results/benchmark/plots
"""

import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Colors ────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    'slatepro':  '#1f77b4',  # blue
    'continual': '#ff7f0e',  # orange
    'noop':      '#7f7f7f',  # grey
}
MODEL_LABELS = {
    'slatepro':  'SLaTe-PRO (batch)',
    'continual': 'Continual (online)',
    'noop':      'No Robot',
}
OUTCOME_COLORS = {
    'skipped':             '#2ca02c',  # green  (TP — robot pre-empted correctly)
    'corrected':           '#d62728',  # red    (FP — robot moved wrong)
    'done_as_is':          '#9467bd',  # purple (FN — robot missed)
    'accepted':            '#17a2b8',  # cyan   (LLM-human went with the robot's move)
    'corrected_despite_correct': '#8c564b',  # brown (moved away though robot was right)
    'redirected':          '#e377c2',  # pink   (moved straight to planned dest, one trip)
    'done_early':          '#bbbbbb',  # grey   (planned line already done by redirect; ignored)
    'skipped_repeated':    '#98df8a',  # light green (dup script TELEPORT after first match)
    'corrected_repeated':  '#ff9896',  # light red
}

METRIC_COLORS = {
    'skipped':                     '#2ca02c',  # green
    'corrected':                   '#d62728',  # red
    'done_as_is':                  '#9467bd',  # purple
    'accepted':                    '#17a2b8',  # cyan
    'skipped_repeated':            '#98df8a',  # light green
    'corrected_repeated':          '#ff9896',  # light red
    'precision_wrong_object':      '#ff7f0e',  # orange
    'wrong_objects_at_end_of_day': '#8c564b',  # brown
    'undone_movements':            '#17becf',  # cyan
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _safe_script_label(day_dict: dict, idx: int) -> str:
    """Best-effort per-day label for plots; never raises.

    Some older/partial benchmark JSONs omit `script` from `per_day` entries.
    """
    script = day_dict.get('script')
    if script:
        try:
            return os.path.basename(str(script)).replace('.txt', '')
        except Exception:
            return str(script).replace('.txt', '')
    return f'day{idx}'


# ── Plot 1: Per-day stacked bar comparison ────────────────────────────────────

def plot_accommodation_comparison(data: dict, out_dir: str) -> str:
    """Side-by-side stacked bar charts of outcomes per day for each model."""
    models = [m for m in ('slatepro', 'continual', 'noop')
              if m in data['results'] and data['results'][m] is not None]
    n_models = len(models)
    if n_models == 0:
        return ''

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        per_day = data['results'][model_name]['per_day']
        days = list(range(len(per_day)))
        # Some older/partial benchmark JSONs may omit these keys per day.
        s = [(d.get('human_metrics') or {}).get('skipped', d.get('skipped', 0)) for d in per_day]
        c = [(d.get('human_metrics') or {}).get('corrected', d.get('corrected', 0)) for d in per_day]
        d_vals = [(d.get('human_metrics') or {}).get('done_as_is', d.get('done_as_is', 0)) for d in per_day]
        s_rep = [(d.get('human_metrics') or {}).get('skipped_repeated', 0) for d in per_day]
        c_rep = [(d.get('human_metrics') or {}).get('corrected_repeated', 0) for d in per_day]

        ax.bar(days, d_vals, label='Done as-is (FN)',
               color=OUTCOME_COLORS['done_as_is'], alpha=0.85)
        ax.bar(days, s, bottom=d_vals, label='Skipped (TP)',
               color=OUTCOME_COLORS['skipped'], alpha=0.85)
        ax.bar(days, c,
               bottom=[dv + sv for dv, sv in zip(d_vals, s)],
               label='Corrected (FP)',
               color=OUTCOME_COLORS['corrected'], alpha=0.85)
        ax.bar(days, s_rep,
               bottom=[dv + sv + cv for dv, sv, cv in zip(d_vals, s, c)],
               label='Skipped (repeat)',
               color=OUTCOME_COLORS['skipped_repeated'], alpha=0.85)
        ax.bar(days, c_rep,
               bottom=[dv + sv + cv + sr for dv, sv, cv, sr in zip(d_vals, s, c, s_rep)],
               label='Corrected (repeat)',
               color=OUTCOME_COLORS['corrected_repeated'], alpha=0.85)

        ax.set_xlabel('Test Day')
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xticks(days)
        ax.set_xticklabels([str(d.get('script', '')).replace('.txt', '') for d in per_day],
                           rotation=45, ha='right', fontsize=7)

    axes[0].set_ylabel('Action Count')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Accommodation Breakdown — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'accommodation_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 2: Precision / Recall / F1 bar chart ────────────────────────────────

def plot_aggregate_metrics(data: dict, out_dir: str) -> str:
    """Grouped bar chart of aggregate precision, recall, F1 for each model."""
    models = [m for m in ('slatepro', 'continual')
              if m in data['results'] and data['results'][m] is not None]
    if not models:
        return ''

    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1']
    x = np.arange(len(metrics))
    bar_w = 0.8 / len(models)

    def _get_metric(totals: dict, name: str) -> float:
        """
        Support both old flattened totals schema (precision/recall/f1 at top-level)
        and the newer structured schema (human_prec/human_rec/human_f1 with nested
        human_metrics/env_metrics).
        """
        if name == "precision":
            return float(totals.get("precision", totals.get("human_prec", 0.0)) or 0.0)
        if name == "recall":
            return float(totals.get("recall", totals.get("human_rec", 0.0)) or 0.0)
        if name == "f1":
            return float(totals.get("f1", totals.get("human_f1", 0.0)) or 0.0)
        return float(totals.get(name, 0.0) or 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model_name in enumerate(models):
        totals = data['results'][model_name]['totals']['test']
        vals = [_get_metric(totals, m) for m in metrics]
        ax.bar(x + i * bar_w, vals, bar_w,
               label=MODEL_LABELS.get(model_name, model_name),
               color=MODEL_COLORS.get(model_name, 'grey'), alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * bar_w, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + bar_w * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(f'Aggregate Metrics — Household {data["household"]}')
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, 'aggregate_metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# ── Plot 3: Per-object success heatmap ────────────────────────────────────────

def plot_object_heatmap(data: dict, out_dir: str) -> str:
    """Heatmap: rows = objects, columns = models, cells = success rate.

    Success rate = skipped / (skipped + corrected + done_as_is) per object.
    Only shows objects that appear in at least one model's results.
    """
    models = [m for m in ('slatepro', 'continual')
              if m in data['results'] and data['results'][m] is not None]
    if not models:
        return ''

    # Aggregate per-object outcomes across all test days for each model
    model_obj_data: dict[str, dict[str, dict]] = {}
    for model_name in models:
        obj_agg: dict[str, dict] = {}
        for day in data['results'][model_name]['per_day']:
            for obj_name, counts in day.get('object_outcomes', {}).items():
                entry = obj_agg.setdefault(obj_name, {'skipped': 0, 'corrected': 0, 'done_as_is': 0})
                for k in ('skipped', 'corrected', 'done_as_is'):
                    entry[k] += counts.get(k, 0)
        model_obj_data[model_name] = obj_agg

    # Collect all object names that have at least one non-done_as_is outcome
    all_objects = set()
    for obj_agg in model_obj_data.values():
        for obj_name, counts in obj_agg.items():
            total = counts['skipped'] + counts['corrected'] + counts['done_as_is']
            if total > 0:
                all_objects.add(obj_name)

    if not all_objects:
        return ''

    objects = sorted(all_objects)

    # Build matrix: rows=objects, cols=[model_skip_rate, model_corr_rate, ...]
    # Show skip rate (green=good) and correction rate (red=bad) side by side
    n_obj = len(objects)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models) + 2, max(4, n_obj * 0.35)),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        obj_agg = model_obj_data[model_name]
        skip_rates = []
        corr_rates = []
        for obj in objects:
            c = obj_agg.get(obj, {'skipped': 0, 'corrected': 0, 'done_as_is': 0})
            total = c['skipped'] + c['corrected'] + c['done_as_is']
            if total == 0:
                skip_rates.append(0)
                corr_rates.append(0)
            else:
                skip_rates.append(c['skipped'] / total)
                corr_rates.append(c['corrected'] / total)

        y = np.arange(n_obj)
        ax.barh(y, skip_rates, height=0.4, color=OUTCOME_COLORS['skipped'],
                alpha=0.85, label='Skip rate (TP)')
        ax.barh(y + 0.4, corr_rates, height=0.4, color=OUTCOME_COLORS['corrected'],
                alpha=0.85, label='Correction rate (FP)')
        ax.set_yticks(y + 0.2)
        ax.set_yticklabels(objects, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Rate')
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(f'Per-Object Outcome Rates — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'object_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

# ── Plot 4: Metrics over time (finetune + test phases) ───────────────────────

def plot_metrics_over_time(data: dict, out_dir: str) -> str:
    """Two-panel line chart of per-day metrics with a finetune/test phase separator.

    Row 1: human interaction outcomes — skipped, corrected, done_as_is
    Row 2: world-state metrics — wrong_object, wrong_objects_at_end_of_day, undone_movements
    """
    models = [m for m in data['results'] if data['results'][m] is not None]
    if not models:
        return ''

    train_days = data.get('train_days', 0)
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(max(14, 5 * n_models), 8),
                             sharex='col', sharey='row')
    # Normalise to 2-D array
    if n_models == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, model_name in enumerate(models):
        per_day = data['results'][model_name]['per_day']
        aed = data['results'][model_name].get('actionwise_evals_daily', [])
        days = list(range(len(per_day)))
        script_labels = [_safe_script_label(d, i) for i, d in enumerate(per_day)]

        # ── Row-1 series ──────────────────────────────────────────────────────
        # Newer schema stores outcomes under d['human_metrics']; older files may be flattened.
        def _hm(d: dict, key: str) -> int:
            hm = d.get('human_metrics') or {}
            return int(d.get(key, hm.get(key, 0)) or 0)

        skipped    = [_hm(d, 'skipped')    for d in per_day]
        corrected  = [_hm(d, 'corrected')  for d in per_day]
        done_as_is = [_hm(d, 'done_as_is') for d in per_day]
        skipped_rep   = [_hm(d, 'skipped_repeated')   for d in per_day]
        corrected_rep = [_hm(d, 'corrected_repeated') for d in per_day]

        # ── Row-2 series ──────────────────────────────────────────────────────
        # Some runs/files may omit `result_metrics` (partial runs or older schema).
        wrong_obj = [(d.get('result_metrics') or {}).get('precision_wrong_object', 0) for d in per_day]
        undone    = [(d.get('result_metrics') or {}).get('undone_movements', 0) for d in per_day]

        wrong_eod = []
        for day_dict in aed:
            count = sum(
                1
                for actions in day_dict.values()
                for a in actions
                if a.get('human_action_outcome') == 'unseen_at_end_of_day'
            )
            wrong_eod.append(count)
        # Pad / trim to match per_day length
        if len(wrong_eod) < len(days):
            wrong_eod += [0] * (len(days) - len(wrong_eod))
        wrong_eod = wrong_eod[:len(days)]

        for row, (series_list, title) in enumerate([
            (
                [
                    (skipped,    'skipped',    'Skipped (TP)'),
                    (corrected,  'corrected',  'Corrected (FP)'),
                    (done_as_is, 'done_as_is', 'Done as-is (FN)'),
                    (skipped_rep,   'skipped_repeated',   'Skipped (repeat)'),
                    (corrected_rep, 'corrected_repeated', 'Corrected (repeat)'),
                ],
                'Human Interaction Outcomes',
            ),
            (
                [
                    (wrong_obj,  'precision_wrong_object',      'Wrong object'),
                    (wrong_eod,  'wrong_objects_at_end_of_day', 'Wrong obj @ end-of-day'),
                    (undone,     'undone_movements',            'Undone movements'),
                ],
                'World-State Metrics',
            ),
        ]):
            ax = axes[row][col]
            for values, key, label in series_list:
                linestyle = '--' if key.endswith('_repeated') else '-'
                ax.plot(days, values, label=label,
                        color=METRIC_COLORS[key], linewidth=1.5, marker='o',
                        markersize=3, alpha=0.85, linestyle=linestyle)

            # Phase separator
            if 0 < train_days <= len(days):
                ax.axvline(x=train_days - 0.5, color='black', linestyle='--',
                           linewidth=1.2, alpha=0.6)
                if row == 0:
                    ymax = ax.get_ylim()[1]
                    ax.text(train_days - 0.5, ymax * 0.98, ' Finetune | Test ',
                            va='top', ha='center', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            ax.set_ylabel('Count')
            if row == 0:
                ax.set_title(f'{MODEL_LABELS.get(model_name, model_name)}\n{title}',
                             fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(axis='y', alpha=0.3)

        # X-axis ticks: show every N-th label to avoid crowding
        n_days = len(days)
        step = max(1, n_days // 20)
        tick_positions = list(range(0, n_days, step))
        axes[1][col].set_xticks(tick_positions)
        axes[1][col].set_xticklabels(
            [script_labels[i] for i in tick_positions],
            rotation=45, ha='right', fontsize=7)
        axes[1][col].set_xlabel('Day (script)')

    fig.suptitle(f'Metrics Over Time — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(out_dir, 'metrics_over_time.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 5: eval_time_from_prediction distribution ────────────────────────────

def plot_eval_time_distribution(data: dict, out_dir: str) -> str:
    """Histogram of eval_time_from_prediction for skipped vs corrected actions."""
    models = [m for m in data['results'] if data['results'][m] is not None]
    if not models:
        return ''

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        aed = data['results'][model_name].get('actionwise_evals_daily', [])
        skipped_times = []
        corrected_times = []
        for day_dict in aed:
            for actions in day_dict.values():
                for a in actions:
                    t = a.get('eval_time_from_prediction')
                    if t is None:
                        continue
                    if a.get('human_action_outcome') == 'skipped':
                        skipped_times.append(t)
                    elif a.get('human_action_outcome') == 'corrected':
                        corrected_times.append(t)

        all_vals = skipped_times + corrected_times
        if not all_vals:
            ax.set_title(f'{MODEL_LABELS.get(model_name, model_name)}\n(no data)')
            continue

        max_val = max(all_vals)
        # Use log-spaced bins if range is large
        if max_val > 100:
            bins = np.unique(np.concatenate([
                np.linspace(0, 50, 26),
                np.logspace(np.log10(50), np.log10(max_val + 1), 30),
            ]))
        else:
            bins = np.linspace(0, max_val + 1, 30)

        for times, key, label in [
            (skipped_times,   'skipped',   'Skipped (TP)'),
            (corrected_times, 'corrected', 'Corrected (FP)'),
        ]:
            if not times:
                continue
            med = np.median(times)
            ax.hist(times, bins=bins, alpha=0.6, color=METRIC_COLORS[key],
                    label=f'{label}  N={len(times)}, med={med:.0f}')
            ax.axvline(med, color=METRIC_COLORS[key], linestyle='--',
                       linewidth=1.5, alpha=0.85)

        if max_val > 100:
            ax.set_xscale('log')
        ax.set_xlabel('eval_time_from_prediction (steps)')
        ax.set_ylabel('Count')
        ax.set_title(f'{MODEL_LABELS.get(model_name, model_name)}\nEval-Time from Prediction')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Eval-Time Distribution — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'eval_time_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 6: Precision & Recall over time ──────────────────────────────────────

def plot_precision_recall_over_time(data: dict, out_dir: str) -> str:
    """Line chart of per-day precision and recall for each model.

    precision = skipped / (skipped + corrected)
    recall    = skipped / (skipped + done_as_is)
    """
    models = [m for m in data['results'] if data['results'][m] is not None]
    if not models:
        return ''

    train_days = data.get('train_days', 0)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(max(10, 5 * n_models), 4.5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        per_day = data['results'][model_name]['per_day']
        days = list(range(len(per_day)))
        script_labels = [_safe_script_label(d, i) for i, d in enumerate(per_day)]

        prec_vals = []
        rec_vals = []
        for d in per_day:
            hm = d.get('human_metrics') or {}
            s = d.get('skipped', hm.get('skipped', 0)) or 0
            c = d.get('corrected', hm.get('corrected', 0)) or 0
            da = d.get('done_as_is', hm.get('done_as_is', 0)) or 0
            prec_vals.append(s / (s + c) if (s + c) > 0 else float('nan'))
            rec_vals.append(s / (s + da) if (s + da) > 0 else float('nan'))

        ax.plot(days, prec_vals, label='Precision', color='#1f77b4',
                linewidth=1.5, marker='o', markersize=3, alpha=0.85)
        ax.plot(days, rec_vals, label='Recall', color='#ff7f0e',
                linewidth=1.5, marker='s', markersize=3, alpha=0.85)

        if 0 < train_days <= len(days):
            ax.axvline(x=train_days - 0.5, color='black', linestyle='--',
                       linewidth=1.2, alpha=0.6)
            ymax = ax.get_ylim()[1]
            ax.text(train_days - 0.5, ymax * 0.98, ' Finetune | Test ',
                    va='top', ha='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel('Score')
        ax.set_xlabel('Day (script)')
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        n_days = len(days)
        step = max(1, n_days // 20)
        tick_positions = list(range(0, n_days, step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([script_labels[i] for i in tick_positions],
                           rotation=45, ha='right', fontsize=7)

    fig.suptitle(f'Precision & Recall Over Time — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'precision_recall_over_time.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 7: Training dataset size over time ──────────────────────────────────

def plot_dataset_size_over_time(data: dict, out_dir: str) -> str:
    """Line chart of training dataset size (number of batches) over finetune days."""
    models = [m for m in data['results'] if data['results'][m] is not None]
    if not models:
        return ''

    has_data = False
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(max(10, 5 * n_models), 4.5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        ds_per_day = data['results'][model_name].get('training_dataset_per_day', {})
        if not ds_per_day:
            ax.set_title(f'{MODEL_LABELS.get(model_name, model_name)}\n(no dataset info)')
            continue

        has_data = True
        day_indices = sorted(int(k) for k in ds_per_day.keys())
        train_sizes = [ds_per_day[str(k) if str(k) in ds_per_day else k]['train']
                       for k in day_indices]
        val_sizes = [ds_per_day[str(k) if str(k) in ds_per_day else k]['val']
                     for k in day_indices]

        ax.plot(day_indices, train_sizes, label='Train batches', color='#2ca02c',
                linewidth=1.5, marker='o', markersize=4, alpha=0.85)
        ax.plot(day_indices, val_sizes, label='Val batches', color='#9467bd',
                linewidth=1.5, marker='s', markersize=4, alpha=0.85)

        ax.set_xlabel('Day index')
        ax.set_ylabel('Dataset size (batches)')
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    if not has_data:
        plt.close(fig)
        return ''

    fig.suptitle(f'Training Dataset Size Over Time — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'dataset_size_over_time.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 8: Metrics vs cumulative training days ──────────────────────────────

def plot_metrics_vs_training_days(data: dict, out_dir: str) -> str:
    """Scatter + line of precision, recall, F1 where X = number of training
    sessions completed by each simulation day.

    Training days are inferred from training_dataset_per_day keys.
    """
    models = [m for m in data['results'] if data['results'][m] is not None]
    if not models:
        return ''

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(max(10, 6 * n_models), 5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    has_data = False

    for ax, model_name in zip(axes, models):
        per_day = data['results'][model_name]['per_day']
        ds_per_day = data['results'][model_name].get('training_dataset_per_day', {})

        trained_day_indices = sorted(int(k) for k in ds_per_day.keys())
        if not trained_day_indices:
            ax.set_title(f'{MODEL_LABELS.get(model_name, model_name)}\n(no training info)')
            continue

        has_data = True

        # Cumulative *days of training data* the active model has seen on this
        # eval day. trained_day_indices keys are the (0-based) day indices on
        # which a training event happened, so the model trained at key k has
        # seen k+1 days of data. Days before the first training event get 0.
        cum_train = []
        for day_idx in range(len(per_day)):
            keys_le = [td for td in trained_day_indices if td <= day_idx]
            cum_train.append((max(keys_le) + 1) if keys_le else 0)

        prec_vals, rec_vals, f1_vals = [], [], []
        for d in per_day:
            hm = d.get('human_metrics') or {}
            s = d.get('skipped', hm.get('skipped', 0)) or 0
            c = d.get('corrected', hm.get('corrected', 0)) or 0
            da = d.get('done_as_is', hm.get('done_as_is', 0)) or 0
            p = s / (s + c) if (s + c) > 0 else float('nan')
            r = s / (s + da) if (s + da) > 0 else float('nan')
            f = 2 * p * r / (p + r) if (p + r) > 0 else float('nan')
            prec_vals.append(p)
            rec_vals.append(r)
            f1_vals.append(f)

        for vals, label, color, marker in [
            (prec_vals, 'Precision', '#1f77b4', 'o'),
            (rec_vals,  'Recall',    '#ff7f0e', 's'),
            (f1_vals,   'F1',        '#2ca02c', '^'),
        ]:
            ax.scatter(cum_train, vals, label=label, color=color,
                       marker=marker, s=20, alpha=0.5, zorder=2)

            x_unique = sorted(set(cum_train))
            y_means = []
            for xu in x_unique:
                ys = [v for ct, v in zip(cum_train, vals)
                      if ct == xu and not (isinstance(v, float) and np.isnan(v))]
                y_means.append(np.mean(ys) if ys else float('nan'))
            ax.plot(x_unique, y_means, color=color, linewidth=1.5,
                    alpha=0.85, zorder=3)

        ax.set_xlabel('Training days completed')
        ax.set_ylabel('Score')
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    if not has_data:
        plt.close(fig)
        return ''

    fig.suptitle(f'Metrics vs Training Days — Household {data["household"]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'metrics_vs_training_days.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Relocation pattern analysis ──────────────────────────────────────────────

def _vh_node_id_to_class_name(dataset_dir: str) -> dict[int, str]:
    """Map VirtualHome node id -> class label from the first ``train/`` or ``test/`` .pt file.

    ``closed_loop_eval`` stores robot matches as ``obj_idx`` / ``loc_idx`` (VH ids), while
    relocation stats from ``_extract_relocations_from_pt`` use ``class_names`` strings aligned
    with ``node_ids`` in the same tensors.
    """
    if not dataset_dir or not os.path.isdir(dataset_dir):
        return {}
    for split in ('train', 'test'):
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for pt_file in sorted(os.listdir(split_dir)):
            if not pt_file.endswith('.pt'):
                continue
            path = os.path.join(split_dir, pt_file)
            try:
                try:
                    d = torch.load(path, map_location='cpu', weights_only=False)
                except TypeError:
                    d = torch.load(path, map_location='cpu')
            except Exception:
                continue
            class_names = d.get('class_names') or []
            node_ids = d.get('node_ids') or []
            if not node_ids or len(node_ids) != len(class_names):
                continue
            out: dict[int, str] = {}
            for nid, cn in zip(node_ids, class_names):
                try:
                    ni = int(nid)
                except (TypeError, ValueError):
                    continue
                if ni == -1:
                    continue
                out[ni] = str(cn)
            if out:
                return out
    return {}


def _actionwise_obj_dest_names(a: dict, vh_id_to_class: dict | None) -> tuple[str | None, str | None]:
    """Object and destination class names for one actionwise eval record (if resolvable)."""
    on = a.get('object_name')
    ln = a.get('location_name')
    if isinstance(on, str) and on.strip() and isinstance(ln, str) and ln.strip():
        return (on.strip(), ln.strip())
    if vh_id_to_class and a.get('obj_idx') is not None and a.get('loc_idx') is not None:
        try:
            oi_i, li_i = int(a['obj_idx']), int(a['loc_idx'])
        except (TypeError, ValueError):
            return (None, None)
        on2, ln2 = vh_id_to_class.get(oi_i), vh_id_to_class.get(li_i)
        if on2 and ln2:
            return (str(on2), str(ln2))
    return (None, None)


def _extract_relocations_from_pt(dataset_dir, split='train'):
    """Extract every object relocation from .pt files in a dataset split.

    Returns list of dicts with keys:
        obj, dest, time, activity, order_in_activity, day_file,
        n_concurrent_in_segment  (total distinct objects relocated in same activity segment)
    """
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        return []
    relocations = []
    for pt_file in sorted(os.listdir(split_dir)):
        if not pt_file.endswith('.pt'):
            continue
        d = torch.load(os.path.join(split_dir, pt_file), map_location='cpu')
        edges = d['edges']          # (T, N, N)
        times = d['times']          # (T,)
        activity = d['activity']    # (T,)
        class_names = d['class_names']
        active_edges = d['active_edges']  # (N, N)

        dyn_mask = active_edges.sum(dim=1) > 0
        dyn_indices = dyn_mask.nonzero(as_tuple=True)[0]
        locs = edges.argmax(dim=-1)  # (T, N)
        act_list = activity.tolist()

        # Two-pass: first collect relocations per activity segment,
        # then annotate each with the segment's total distinct relocated objects.

        # Pass 1: identify activity segments and their relocations
        segments = []  # list of (start_t, end_t, activity_idx)
        seg_start = 0
        for t in range(1, len(act_list)):
            if act_list[t] != act_list[t - 1]:
                segments.append((seg_start, t - 1, act_list[seg_start]))
                seg_start = t
        segments.append((seg_start, len(act_list) - 1, act_list[seg_start]))

        # Pass 2: for each segment, find all relocations and count distinct objects
        file_relocs = []
        for seg_start, seg_end, seg_act in segments:
            seg_relocs = []
            move_count = 0
            last_move_t = -1
            relocated_objects = set()

            for t in range(max(seg_start, 1), seg_end + 1):
                # Skip if we crossed into a different activity at t-1
                if act_list[t] != act_list[t - 1] and t == seg_start:
                    continue
                for oi in dyn_indices.tolist():
                    if locs[t, oi] != locs[t - 1, oi]:
                        if t != last_move_t:
                            move_count += 1
                            last_move_t = t
                        relocated_objects.add(oi)
                        seg_relocs.append({
                            'obj': class_names[oi],
                            'dest': class_names[locs[t, oi].item()],
                            'time': times[t].item(),
                            'activity': seg_act,
                            'order_in_activity': move_count,
                            'day_file': pt_file,
                        })

            n_concurrent = len(relocated_objects)
            for r in seg_relocs:
                r['n_concurrent_in_segment'] = n_concurrent
            file_relocs.extend(seg_relocs)

        relocations.extend(file_relocs)
    return relocations


def _build_relocation_analysis(dataset_dir, result_data, warmup_days,
                               vh_id_to_class: dict | None = None):
    """Cross-reference training-set relocation patterns with eval predictions.

    Returns a list of dicts (one per unique (obj, dest) relocation type):
        obj, dest, recall, time_std, n_concurrent, n_activities, day_fraction,
        mode_order, count_train, count_eval, count_correct
    """
    # --- Training-set statistics ---
    train_relocs = _extract_relocations_from_pt(dataset_dir, 'train')
    n_train_files = len(set(r['day_file'] for r in train_relocs)) if train_relocs else 1
    # Per (obj, dest): collect timestamps, orders, concurrent counts, activities, days
    train_times = defaultdict(list)
    train_orders = defaultdict(list)
    train_concurrent = defaultdict(list)
    train_activities = defaultdict(set)
    train_days = defaultdict(set)
    for r in train_relocs:
        key = (r['obj'], r['dest'])
        train_times[key].append(r['time'])
        train_orders[key].append(r['order_in_activity'])
        train_concurrent[key].append(r['n_concurrent_in_segment'])
        train_activities[key].add(r['activity'])
        train_days[key].add(r['day_file'])

    # --- Eval-phase human movements (from test split .pt files) ---
    test_relocs = _extract_relocations_from_pt(dataset_dir, 'test')
    eval_human = Counter()
    for r in test_relocs:
        eval_human[(r['obj'], r['dest'])] += 1

    # --- Robot correct predictions from actionwise evals ---
    evals = result_data.get('actionwise_evals_daily', [])
    id_map = vh_id_to_class if vh_id_to_class is not None else _vh_node_id_to_class_name(dataset_dir)
    robot_correct = Counter()
    for day_idx in range(warmup_days, len(evals)):
        for actions in evals[day_idx].values():
            for a in actions:
                if a.get('human_action_outcome') != 'skipped':
                    continue
                on, ln = _actionwise_obj_dest_names(a, id_map)
                if on and ln:
                    robot_correct[(on, ln)] += 1

    # --- Join ---
    all_keys = set(train_times.keys()) | set(eval_human.keys())
    rows = []
    for key in all_keys:
        obj, dest = key
        times_list = train_times.get(key, [])
        orders = train_orders.get(key, [])
        concurrents = train_concurrent.get(key, [])
        n_eval = eval_human.get(key, 0)
        n_correct = robot_correct.get(key, 0)
        mode_order = Counter(orders).most_common(1)[0][0] if orders else 0
        rows.append({
            'obj': obj,
            'dest': dest,
            'recall': n_correct / n_eval if n_eval > 0 else float('nan'),
            'time_std': float(np.std(times_list)) if len(times_list) > 1 else 0.0,
            'n_concurrent': float(np.median(concurrents)) if concurrents else 0.0,
            'n_activities': len(train_activities.get(key, set())),
            'day_fraction': len(train_days.get(key, set())) / n_train_files,
            'mode_order': mode_order,
            'count_train': len(times_list),
            'count_eval': n_eval,
            'count_correct': n_correct,
        })
    return rows


def plot_recall_vs_activity_complexity(analysis_rows, out_dir, household=''):
    """Scatter: x = activity complexity (concurrent relocations), y = recall.

    Point size ~ eval count, color ~ number of distinct activities.
    """
    rows = [r for r in analysis_rows if r['count_eval'] >= 2 and r['count_train'] >= 2]
    if not rows:
        return ''

    x = np.array([r['n_concurrent'] for r in rows])
    y = np.array([r['recall'] for r in rows])
    sizes = np.array([r['count_eval'] for r in rows])
    n_acts = np.array([r['n_activities'] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(x, y, s=sizes * 5 + 20, c=n_acts, cmap='viridis',
                    alpha=0.7, edgecolors='k', linewidths=0.3,
                    vmin=max(1, n_acts.min()), vmax=n_acts.max())
    fig.colorbar(sc, ax=ax, label='Number of distinct activities')

    # Label the largest points
    for r in sorted(rows, key=lambda r: r['count_eval'], reverse=True)[:12]:
        ax.annotate(f"{r['obj'][:12]}->{r['dest'][:10]}",
                    (r['n_concurrent'], r['recall']),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 4), textcoords='offset points')

    # Trend line (binned weighted means)
    if len(x) >= 5:
        bin_edges = np.arange(x.min() - 0.5, x.max() + 1.5, max(1, (x.max() - x.min()) / 7))
        bin_centers, bin_means = [], []
        for i in range(len(bin_edges) - 1):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            if mask.sum() >= 2:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means.append(np.average(y[mask], weights=sizes[mask]))
        if bin_centers:
            ax.plot(bin_centers, bin_means, 'r--', linewidth=2, alpha=0.7,
                    label='Weighted mean recall', zorder=5)
            ax.legend(fontsize=9)

    ax.set_xlabel('Activity complexity (concurrent relocations in segment)', fontsize=11)
    ax.set_ylabel('Movement-wise recall (finetune phase)', fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f'Recall vs Activity Complexity — Household {household}', fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, 'recall_vs_activity_complexity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_recall_by_movement_order(analysis_rows, out_dir, household=''):
    """Bar chart: average recall grouped by movement order (mode) within activity,
    split by activity complexity (low/high concurrent relocations).
    """
    rows = [r for r in analysis_rows if r['count_eval'] >= 2 and r['count_train'] >= 2]
    if not rows:
        return ''

    # Bin movement order by mode: 1, 2, 3, 4+
    def order_bin(o):
        return str(o) if o <= 3 else '4+'

    median_concurrent = float(np.median([r['n_concurrent'] for r in rows]))

    bins = ['1', '2', '3', '4+']
    categories = ['Simple activity', 'Complex activity']

    # {order_bin: {category: (weighted_sum_recall, total_weight)}}
    agg = {b: {c: [0.0, 0] for c in categories} for b in bins}
    for r in rows:
        ob = order_bin(r['mode_order'])
        cat = 'Simple activity' if r['n_concurrent'] <= median_concurrent else 'Complex activity'
        w = r['count_eval']
        agg[ob][cat][0] += r['recall'] * w
        agg[ob][cat][1] += w

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bins))
    bar_w = 0.35
    colors = ['#2ca02c', '#d62728']

    for i, cat in enumerate(categories):
        means = []
        for b in bins:
            ws, wt = agg[b][cat]
            means.append(ws / wt if wt > 0 else 0)
        ax.bar(x + i * bar_w, means, bar_w, label=cat, color=colors[i], alpha=0.85)
        for j, (m, b) in enumerate(zip(means, bins)):
            n = agg[b][cat][1]
            if n > 0:
                ax.text(x[j] + i * bar_w, m + 0.02, f'n={n}', ha='center',
                        va='bottom', fontsize=7)

    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels([f'Order {b}' for b in bins])
    ax.set_ylabel('Weighted mean recall')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Recall by Movement Order — Household {household}\n'
                 f'(split by activity complexity, median={median_concurrent:.0f} concurrent objects)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, 'recall_by_movement_order.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Plot 9: Prediction heatmap (time x object, colored by outcome) ──────────

HEATMAP_COLORS = {
    'skipped':   '#2ca02c',   # green  — robot correct
    'corrected': '#d62728',   # red    — robot wrong
    'corrected_through_vigilance': '#ff9999', # light red — robot wrong, but corrected through vigilance
    'other':     '#c4a661',   # dark tan — unseen / undone / none
    'done_as_is': '#7f7f7f',  # grey — human moved, robot did not match (FN)
}


def _extract_day_points(day_data: dict, vh_id_to_class: dict | None = None):
    """Extract (time_minutes, object_name, color) from one day's actionwise dict."""
    points = []
    obj_freq = Counter()
    for ts_str, actions in day_data.items():
        t_min = int(ts_str)
        for a in actions:
            on, _ln = _actionwise_obj_dest_names(a, vh_id_to_class)
            obj = on or (
                f"id_{a['obj_idx']}" if a.get('obj_idx') is not None else '?')
            outcome = a.get('human_action_outcome')
            if outcome == 'skipped':
                color = HEATMAP_COLORS['skipped']
            elif outcome == 'corrected':
                color = HEATMAP_COLORS['corrected']
            elif outcome == 'corrected_through_vigilance':
                color = HEATMAP_COLORS['corrected_through_vigilance']
            else:
                color = HEATMAP_COLORS['other']
            points.append((t_min, obj, color))
            obj_freq[obj] += 1
    return points, obj_freq


def _extract_done_as_is_points(per_day_entry: dict | None) -> list[tuple[int, str]]:
    """Human script teleports with outcome done_as_is (robot false negative), with time."""
    if not per_day_entry:
        return []
    events = per_day_entry.get('human_action_events')
    if not events:
        return []
    out = []
    for e in events:
        if e.get('outcome') != 'done_as_is':
            continue
        obj = e.get('object_name') or e.get('object', '')
        t = e.get('time_minutes', e.get('timestep'))
        if t is None or not obj:
            continue
        out.append((int(t), obj))
    return out
    
def plot_prediction_heatmap(result_data: dict, day_indices: list[int],
                            out_dir: str, household: str = '',
                            condition: str = '',
                            ref_result: dict | None = None,
                            ref_label: str = 'ref',
                            path_suffix: str = '',
                            vh_id_to_class: dict | None = None) -> str:
    """Stacked heatmap of robot predictions across multiple days.

    One subplot per day, stacked vertically. All subplots share the same
    Y axis (union of all objects across days, sorted by total frequency)
    and X axis (clock time).

    If ``ref_result`` is supplied, each day panel also gets a twin y-axis
    step line overlay showing edit distance between cur and ref world-state
    timelines across the day.

    Parameters
    ----------
    result_data : dict
        Benchmark dict with 'actionwise_evals_daily' and 'per_day'.
    day_indices : list[int]
        Indices into actionwise_evals_daily to plot.
    ref_result : dict, optional
        Reference benchmark dict (e.g. Cond 1) whose per-day
        ``world_state_timeline`` is diffed against cur.
    vh_id_to_class : dict[int, str], optional
        Maps VH node id -> class name for actionwise rows that only store ``obj_idx`` /
        ``loc_idx`` (from ``_vh_node_id_to_class_name`` on the dataset directory).
    """
    aed = result_data.get('actionwise_evals_daily', [])
    per_day = result_data.get('per_day', [])

    # Filter to valid day indices that have predictions
    valid_days = []
    for di in day_indices:
        if 0 <= di < len(aed) and aed[di]:
            valid_days.append(di)
    if not valid_days:
        print(f'  [WARNING] No valid days with predictions in {day_indices}')
        return ''

    # First pass: collect all objects across all days for a shared Y axis
    all_obj_freq = Counter()
    day_points = {}
    day_done_as_is = {}
    for di in valid_days:
        points, obj_freq = _extract_day_points(aed[di], vh_id_to_class)
        if points:
            day_points[di] = points
            all_obj_freq.update(obj_freq)
        pde = per_day[di] if di < len(per_day) else None
        dai = _extract_done_as_is_points(pde)
        if dai:
            day_done_as_is[di] = dai
            for t, obj in dai:
                all_obj_freq[obj] += 1

    valid_days = [di for di in valid_days if di in day_points]
    if not valid_days:
        return ''

    # Edit-distance overlays: per-day (times, eds) keyed by cur day index.
    ed_overlays: dict[int, tuple[list[int], list[int]]] = {}
    ed_peak: dict[int, int] = {}
    if ref_result is not None:
        ref_pd = ref_result.get('per_day', [])
        script_to_ref_idx = {s: i for i, p in enumerate(ref_pd) if (s := p.get('script'))}
        for di in valid_days:
            cur_day = per_day[di] if di < len(per_day) else None
            if not cur_day:
                continue
            cur_tl = cur_day.get('world_state_timeline') or []
            ri = script_to_ref_idx.get(cur_day.get('script'))
            if ri is None:
                continue
            ref_tl = ref_pd[ri].get('world_state_timeline') or []
            if not cur_tl or not ref_tl:
                continue
            times = sorted({s['time_minutes'] for s in cur_tl} |
                           {s['time_minutes'] for s in ref_tl})
            eds = []
            for t in times:
                sc = _state_at_time(cur_tl, t)
                sr = _state_at_time(ref_tl, t)
                d, _ = _edit_distance(sc, sr)
                eds.append(d)
            ed_overlays[di] = (times, eds)
            ed_peak[di] = max(eds) if eds else 0

    global_ed_max = max(ed_peak.values()) if ed_peak else 0

    # Shared Y axis: keep top objects by frequency, sorted most-frequent at top
    max_objects = 20
    objects_sorted = [obj for obj, _ in all_obj_freq.most_common(max_objects)]
    objects_sorted.reverse()  # most frequent at top in plot
    obj_to_y = {obj: i for i, obj in enumerate(objects_sorted)}
    n_objects = len(objects_sorted)

    n_days = len(valid_days)
    # Readability tweaks: bigger fonts/markers + wider figure to reduce overlap.
    marker_size = 130
    ytick_fontsize = 16
    daylabel_fontsize = 16
    xtick_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 10
    legend_markersize = 10

    row_h = max(2.2, n_objects * 0.28 + 0.6)
    fig_width = 28
    fig, axes = plt.subplots(n_days, 1, figsize=(fig_width, row_h * n_days + 1.8),
                              sharex=True, sharey=True)
    if n_days == 1:
        axes = [axes]

    dai_marker_size = max(55, int(marker_size * 0.55))
    ref_cross_size = max(80, int(marker_size * 0.6))

    # Pre-compute ref condition points per day (keyed by cur day index)
    ref_day_points: dict[int, list] = {}
    if ref_result is not None:
        ref_pd_list = ref_result.get('per_day', [])
        ref_aed_list = ref_result.get('actionwise_evals_daily', [])
        ref_script_map = {s: i for i, p in enumerate(ref_pd_list) if (s := p.get('script'))}
        for di in valid_days:
            cur_script = per_day[di].get('script') if di < len(per_day) else None
            if not cur_script or cur_script not in ref_script_map:
                continue
            ri = ref_script_map[cur_script]
            if ri < len(ref_aed_list) and ref_aed_list[ri]:
                pts, _ = _extract_day_points(ref_aed_list[ri], vh_id_to_class)
                if pts:
                    ref_day_points[di] = pts

    ed_axes: list = []
    for ax, di in zip(axes, valid_days):
        points = day_points[di]
        # Human done-as-is (no matching robot prediction): draw first, under robot dots
        if di in day_done_as_is:
            dai_pts = day_done_as_is[di]
            dx = [t for t, obj in dai_pts if obj in obj_to_y]
            dy = [obj_to_y[obj] for t, obj in dai_pts if obj in obj_to_y]
            if dx:
                ax.scatter(
                    dx, dy, c=HEATMAP_COLORS['done_as_is'], s=dai_marker_size,
                    alpha=0.85, edgecolors='#4a4a4a', linewidths=0.6, zorder=2,
                    marker='o')

        # Filter to objects in our Y axis
        xs = [p[0] for p in points if p[1] in obj_to_y]
        ys = [obj_to_y[p[1]] for p in points if p[1] in obj_to_y]
        colors = [p[2] for p in points if p[1] in obj_to_y]
        alphas = [0.5 if p[2] == HEATMAP_COLORS['corrected_through_vigilance'] else 1 for p in points if p[1] in obj_to_y]

        try:
            ax.scatter(xs, ys, c=colors, s=marker_size, alpha=alphas, edgecolors='none',
                       zorder=3)
        except Exception as e:
            print(f'Error plotting points: {e}')
            print(f'xs: {xs}')
            print(f'ys: {ys}')
            print(f'colors: {colors}')
            ax.scatter(xs, ys, c=colors, s=marker_size, alpha=0.85, edgecolors='none',
                       zorder=3)

        # Ref condition predictions as crosses (offset +0.2 on Y so both visible)
        if di in ref_day_points:
            rpts = ref_day_points[di]
            rxs = [p[0] for p in rpts if p[1] in obj_to_y]
            rys = [obj_to_y[p[1]] + 0.2 for p in rpts if p[1] in obj_to_y]
            rcolors = [p[2] for p in rpts if p[1] in obj_to_y]
            ax.scatter(rxs, rys, c=rcolors, s=ref_cross_size, alpha=0.7,
                       marker='x', linewidths=1.5, zorder=2.5)

        # Y axis
        ax.set_yticks(range(n_objects))
        ax.set_yticklabels(objects_sorted, fontsize=ytick_fontsize)
        ax.set_ylim(-0.5, n_objects - 0.5)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='y', length=3, pad=2, labelsize=ytick_fontsize)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)

        # Row label as right-side annotation
        day_num = per_day[di]['day'] if di < len(per_day) else di
        total = len(points)
        n_skip = sum(1 for _, _, c in points if c == HEATMAP_COLORS['skipped'])
        n_corr = sum(1 for _, _, c in points if c == HEATMAP_COLORS['corrected'])
        n_corr_vig = sum(1 for _, _, c in points if c == HEATMAP_COLORS['corrected_through_vigilance'])
        n_dai = len(day_done_as_is.get(di, []))
        dai_part = f', {n_dai}d' if n_dai else ''
        ed_part = f', {ed_peak[di]}ed' if di in ed_peak else ''
        label = f'Day {day_num} ({total}pts, {n_skip}skip, {n_corr}corr, {n_corr_vig}cv{dai_part}{ed_part})'
        ax.set_ylabel(label, fontsize=daylabel_fontsize)

        # Edit-distance overlay on twin y-axis
        if di in ed_overlays:
            ts_ed, eds_ed = ed_overlays[di]
            ax_ed = ax.twinx()
            ax_ed.step(ts_ed, eds_ed, where='post', color='#6a3d9a',
                       linewidth=1.8, alpha=0.9, zorder=1)
            ax_ed.fill_between(ts_ed, eds_ed, step='post', alpha=0.10,
                               color='#6a3d9a')
            if global_ed_max > 0:
                ax_ed.set_ylim(0, global_ed_max * 1.05)
            ax_ed.tick_params(axis='y', labelcolor='#6a3d9a',
                              labelsize=ytick_fontsize - 2)
            ax_ed.set_ylabel('edit dist', color='#6a3d9a',
                             fontsize=daylabel_fontsize - 2)
            ed_axes.append(ax_ed)

    # X axis on bottom subplot
    all_xs = [p[0] for pts in day_points.values() for p in pts]
    x_min, x_max = min(all_xs), max(all_xs)
    hour_start = (x_min // 60) * 60
    hour_end = ((x_max // 60) + 1) * 60
    hour_ticks = list(range(int(hour_start), int(hour_end) + 1, 60))
    axes[-1].set_xticks(hour_ticks)
    axes[-1].set_xticklabels([f'{(h // 60) % 24:02d}:00' for h in hour_ticks],
                              fontsize=xtick_fontsize)
    axes[-1].set_xlim(x_min - 10, x_max + 10)
    axes[-1].set_xlabel('Time of day', fontsize=xtick_fontsize + 1)

    # Legend on top subplot
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HEATMAP_COLORS['skipped'],
               markersize=legend_markersize, label='Skipped (robot correct)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HEATMAP_COLORS['corrected'],
               markersize=legend_markersize, label='Corrected (robot wrong)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HEATMAP_COLORS['other'],
               markersize=legend_markersize, label='Other (unseen/undone)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HEATMAP_COLORS['corrected_through_vigilance'],
               markersize=legend_markersize, label='Corrected through vigilance (robot wrong, but corrected through vigilance)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=HEATMAP_COLORS['done_as_is'],
               markeredgecolor='#4a4a4a', markeredgewidth=0.6,
               markersize=legend_markersize, label='Done as-is (human, no robot match)'),
    ]
    if ref_day_points:
        legend_handles.extend([
            Line2D([0], [0], marker='x', color='w', markerfacecolor=HEATMAP_COLORS['skipped'],
                   markeredgecolor=HEATMAP_COLORS['skipped'], markeredgewidth=1.5,
                   markersize=legend_markersize, label=f'{ref_label} Skipped'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor=HEATMAP_COLORS['corrected'],
                   markeredgecolor=HEATMAP_COLORS['corrected'], markeredgewidth=1.5,
                   markersize=legend_markersize, label=f'{ref_label} Corrected'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor=HEATMAP_COLORS['other'],
                   markeredgecolor=HEATMAP_COLORS['other'], markeredgewidth=1.5,
                   markersize=legend_markersize, label=f'{ref_label} Other'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor=HEATMAP_COLORS['corrected_through_vigilance'],
                   markeredgecolor=HEATMAP_COLORS['corrected_through_vigilance'], markeredgewidth=1.5,
                   markersize=legend_markersize, label=f'{ref_label} Corrected through vigilance'),
        ])
    if ed_overlays:
        legend_handles.append(
            Line2D([0], [0], color='#6a3d9a', linewidth=2,
                   label=f'edit dist (cur vs {ref_label})'))
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=legend_fontsize)

    # Suptitle
    title_parts = []
    if condition:
        title_parts.append(condition)
    if household:
        title_parts.append(f'HH {household}')
    title_parts.append(f'Robot Predictions — {n_days} finetune days')
    fig.suptitle(' '.join(title_parts), fontsize=title_fontsize, fontweight='bold')
    # Extra left margin for larger y tick labels.
    fig.tight_layout(rect=[0.11, 0, 1, 0.97])

    day_range = f'{valid_days[0]}-{valid_days[-1]}'
    if ed_overlays:
        slug = ref_label.replace(' ', '_')
        fname = f'prediction_heatmap_days{day_range}_HH{household}_vs_{slug}.png'
    else:
        fname = f'prediction_heatmap_days{day_range}_HH{household}.png'
    base, ext = os.path.splitext(fname)
    path = os.path.join(out_dir, f'{base}_{path_suffix}{ext}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Cross-condition state divergence (vs reference condition, e.g. Cond 1) ──

def _load_ref_result(ref_path: str) -> dict:
    with open(ref_path) as f:
        d = json.load(f)
    return d.get('results', {}).get('slatepro', d)


def _state_at_time(timeline: list[dict], t: int) -> dict:
    """Last snapshot <= t. Returns vh_obj_to_loc (VH obj ID -> VH loc ID).

    Uses vh_obj_to_loc rather than slate_obj_to_loc to avoid counting
    vocabulary-padding nodes that aren't real household objects.
    """
    last = {}
    for snap in timeline:
        if snap['time_minutes'] > t:
            break
        last = snap.get('vh_obj_to_loc', {})
    return last


def _edit_distance(s1: dict, s2: dict) -> tuple[int, set[str]]:
    """Hamming on shared keys. Returns (count, set_of_differing_obj_idx_str)."""
    keys = set(s1) | set(s2)
    diff = {k for k in keys if s1.get(k) != s2.get(k)}
    return len(diff), diff


def _build_idx_to_name(aed: list[dict]) -> dict[str, str]:
    """Walk actionwise_evals_daily to recover obj_idx -> object_name mapping."""
    m: dict[str, str] = {}
    for day in aed:
        if not day:
            continue
        for actions in day.values():
            for a in actions:
                oi = a.get('obj_idx')
                nm = a.get('object_name')
                if oi is not None and nm and str(oi) not in m:
                    m[str(oi)] = nm
    return m


def _day_pairs_by_script(cur_pd: list[dict], ref_pd: list[dict]):
    """Yield (cur_idx, ref_idx, script) for days present in both, warmup days skipped
    (detected as days with no robot actions in cur)."""
    ref_map = {s: i for i, p in enumerate(ref_pd) if (s := p.get('script'))}
    for ci, cp in enumerate(cur_pd):
        if cp.get('total_robot', 0) == 0:
            continue
        s = cp.get('script')
        if s and s in ref_map:
            yield ci, ref_map[s], s


def plot_edit_distance_vs_outcomes(cur_data: dict, ref_data: dict,
                                    out_dir: str, household: str,
                                    cur_label: str, ref_label: str) -> str:
    """Per-day mean & end-of-day edit distance (cur vs ref world-state timeline)
    overlaid with cur's TP/FP counts + 'missed vs ref' (objects ref skipped but
    cur done_as_is). Pearson correlations footed at the bottom."""
    cur_pd = cur_data.get('per_day', [])
    ref_pd = ref_data.get('per_day', [])
    cur_aed = cur_data.get('actionwise_evals_daily', [])
    ref_aed = ref_data.get('actionwise_evals_daily', [])

    days, labels, ed_mean, ed_end = [], [], [], []
    tp, fp, missed_ref = [], [], []
    for ci, ri, script in _day_pairs_by_script(cur_pd, ref_pd):
        cur_tl = cur_pd[ci].get('world_state_timeline') or []
        ref_tl = ref_pd[ri].get('world_state_timeline') or []
        if not cur_tl or not ref_tl:
            continue
        times = sorted({s['time_minutes'] for s in cur_tl} |
                       {s['time_minutes'] for s in ref_tl})
        if not times:
            continue
        eds = []
        for t in times:
            d, _ = _edit_distance(_state_at_time(cur_tl, t),
                                  _state_at_time(ref_tl, t))
            eds.append(d)
        ed_mean.append(float(np.mean(eds)))
        ed_end.append(float(eds[-1]))

        days.append(ci)
        labels.append(script.replace('.txt', ''))
        hm = cur_pd[ci].get('human_metrics') or {}
        tp.append(hm.get('skipped', 0))
        fp.append(hm.get('corrected', 0))

        ref_skipped: set = set()
        if ri < len(ref_aed) and ref_aed[ri]:
            for actions in ref_aed[ri].values():
                for a in actions:
                    if a.get('human_action_outcome') == 'skipped':
                        ref_skipped.add(a.get('object_name'))
        cur_done: set = set()
        for e in cur_pd[ci].get('human_action_events', []):
            if e.get('outcome') == 'done_as_is':
                obj = e.get('object_name')
                if obj:
                    cur_done.add(obj)
        missed_ref.append(len(ref_skipped & cur_done))

    if not days:
        return ''

    x = np.arange(len(days))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(days) * 0.35), 7.5),
                              sharex=True,
                              gridspec_kw={'height_ratios': [1.2, 1.6], 'hspace': 0.08})
    ax0 = axes[0]
    ax0.plot(x, ed_mean, color='#6a3d9a', linewidth=1.7, marker='o',
             markersize=3, label='mean edit dist')
    ax0.plot(x, ed_end, color='#9e68c9', linewidth=1.0, marker='x',
             markersize=4, linestyle='--', alpha=0.8, label='end-of-day edit dist')
    ax0.fill_between(x, ed_mean, alpha=0.12, color='#6a3d9a')
    ax0.set_ylabel(f'Edit dist\n({cur_label} vs {ref_label})')
    ax0.legend(fontsize=8, loc='upper left')
    ax0.grid(alpha=0.3)
    ax0.set_title(f'State divergence vs outcomes — HH {household}  '
                  f'{cur_label} vs {ref_label}', fontsize=12, fontweight='bold')

    ax1 = axes[1]
    ax1.bar(x, tp, color=OUTCOME_COLORS['skipped'], alpha=0.85,
            label='Skipped (TP)')
    ax1.bar(x, fp, bottom=tp, color=OUTCOME_COLORS['corrected'], alpha=0.85,
            label='Corrected (FP)')
    ax1.set_ylabel(f'{cur_label} counts')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(x, missed_ref, color='#ff7f0e', linewidth=1.5, marker='s',
              markersize=4, label=f'Missed vs {ref_label}')
    ax1b.set_ylabel(f'Missed (obj) vs {ref_label}', color='#ff7f0e')
    ax1b.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1b.legend(fontsize=8, loc='upper right')

    step = max(1, len(days) // 25)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([labels[i] for i in range(0, len(days), step)],
                         rotation=45, ha='right', fontsize=7)
    ax1.set_xlabel('Day (script)')

    def _r(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])
    fig.text(0.5, 0.01,
             f'Pearson(edit_dist, TP)={_r(ed_mean, tp):.2f}   '
             f'Pearson(edit_dist, FP)={_r(ed_mean, fp):.2f}   '
             f'Pearson(edit_dist, Missed)={_r(ed_mean, missed_ref):.2f}',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    slug = ref_label.replace(' ', '_')
    path = os.path.join(out_dir, f'edit_distance_vs_outcomes_HH{household}_{slug}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_state_diff_object_correlation(cur_data: dict, ref_data: dict,
                                        out_dir: str, household: str,
                                        cur_label: str, ref_label: str,
                                        top_k: int = 20) -> str:
    """Heatmap: rows = objects whose location differs (cur vs ref) on day d,
    cols = objects wrongly moved or missed-vs-ref on day d. Cell = P(col wrong
    on day | row differed on day) plus a 'lift' panel = P(wrong|diff)/P(wrong).
    """
    cur_pd = cur_data.get('per_day', [])
    ref_pd = ref_data.get('per_day', [])
    cur_aed = cur_data.get('actionwise_evals_daily', [])
    ref_aed = ref_data.get('actionwise_evals_daily', [])

    idx_to_name = _build_idx_to_name(cur_aed)
    idx_to_name.update(_build_idx_to_name(ref_aed))

    diff_days = Counter()
    wrong_days = Counter()
    joint = Counter()
    total_days = 0
    for ci, ri, _ in _day_pairs_by_script(cur_pd, ref_pd):
        cur_tl = cur_pd[ci].get('world_state_timeline') or []
        ref_tl = ref_pd[ri].get('world_state_timeline') or []
        if not cur_tl or not ref_tl:
            continue

        day_diff: set = set()
        times = sorted({s['time_minutes'] for s in cur_tl} |
                       {s['time_minutes'] for s in ref_tl})
        for t in times:
            _, d = _edit_distance(_state_at_time(cur_tl, t),
                                  _state_at_time(ref_tl, t))
            day_diff.update(d)
        diff_names = {idx_to_name.get(oi, f'obj_{oi}') for oi in day_diff}

        wrong_names: set = set()
        if ci < len(cur_aed) and cur_aed[ci]:
            for actions in cur_aed[ci].values():
                for a in actions:
                    outcome = a.get('human_action_outcome', '')
                    label = str(a.get('label', ''))
                    if outcome == 'corrected' or label == 'wrong_object':
                        wn = a.get('object_name')
                        if wn:
                            wrong_names.add(wn)

        ref_skip_obj: set = set()
        if ri < len(ref_aed) and ref_aed[ri]:
            for actions in ref_aed[ri].values():
                for a in actions:
                    if a.get('human_action_outcome') == 'skipped':
                        on = a.get('object_name')
                        if on:
                            ref_skip_obj.add(on)
        cur_done_obj: set = set()
        for e in cur_pd[ci].get('human_action_events', []):
            if e.get('outcome') == 'done_as_is':
                on = e.get('object_name')
                if on:
                    cur_done_obj.add(on)
        wrong_names |= (ref_skip_obj & cur_done_obj)

        total_days += 1
        for dn in diff_names:
            diff_days[dn] += 1
        for wn in wrong_names:
            wrong_days[wn] += 1
        for dn in diff_names:
            for wn in wrong_names:
                joint[(dn, wn)] += 1

    if not joint or total_days == 0:
        return ''

    rows = [o for o, n in diff_days.most_common() if n >= 2][:top_k]
    cols = [o for o, n in wrong_days.most_common() if n >= 2][:top_k]
    if not rows or not cols:
        return ''

    M = np.zeros((len(rows), len(cols)))
    L = np.zeros_like(M)
    for i, dn in enumerate(rows):
        denom = max(diff_days[dn], 1)
        for j, wn in enumerate(cols):
            n = joint.get((dn, wn), 0)
            M[i, j] = n / denom
            base = wrong_days[wn] / total_days
            L[i, j] = (M[i, j] / base) if base > 0 else 0.0

    fig, axes = plt.subplots(
        1, 2, figsize=(max(12, len(cols) + len(cols) * 0.5 + 3),
                       max(6, len(rows) * 0.38 + 2)))
    for ax, mat, title, cmap, vmax in [
        (axes[0], M, 'P(wrong | state-diff)', 'YlOrRd', 1.0),
        (axes[1], L, 'Lift = P(wrong|diff) / P(wrong)', 'RdBu_r', 3.0),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat[i, j]
                if v > (0.15 if cmap == 'YlOrRd' else 1.3):
                    col = 'white' if v > (0.5 if cmap == 'YlOrRd' else 2.0) else 'black'
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=6, color=col)

    axes[0].set_ylabel(f'State-diff object ({cur_label} vs {ref_label})')
    axes[0].set_xlabel(f'Wrong / missed object ({cur_label})')
    axes[1].set_xlabel(f'Wrong / missed object ({cur_label})')
    fig.suptitle(f'State divergence → output errors  — HH {household}  '
                 f'{cur_label} vs {ref_label}  (n_days={total_days})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    slug = ref_label.replace(' ', '_')
    path = os.path.join(out_dir, f'state_diff_vs_wrong_obj_HH{household}_{slug}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize benchmark comparison results')
    parser.add_argument('--results', required=True,
                        help='Path to benchmark JSON (from benchmark_comparison.py)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for plots (default: same dir as results)')
    parser.add_argument('--dataset_dir', default=None,
                        help='Path to generated dataset (e.g. external/gen_.../HouseholdB/processed_seqLM_coarse). '
                             'Required for relocation-pattern plots. Auto-detected from status file if omitted.')
    parser.add_argument('--warmup_days', type=int, default=None,
                        help='Number of warmup days (auto-detected from results if omitted)')
    parser.add_argument('--condition', default='',
                        help='Condition label for plot titles (e.g. "Cond 5")')
    parser.add_argument('--ref_results', default=None,
                        help='Path to reference benchmark JSON (e.g. Cond 1) to '
                             'compare against for state-divergence plots')
    parser.add_argument('--ref_label', default='Cond 1',
                        help='Label for reference condition in plots')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.results), 'plots')
    os.makedirs(out_dir, exist_ok=True)

    data = load_results(args.results)

    # Detect format: comparison (has 'results' + 'household') vs flat (has 'per_day' at top level)
    is_comparison = 'results' in data and 'household' in data
    household = data.get('household', '')
    if not household:
        # Try to infer from filename (e.g. benchmark_slatepro_B.json -> B)
        fname = os.path.basename(args.results)
        for suffix in ['_A.json', '_B.json', '_C.json']:
            if fname.endswith(suffix):
                household = suffix[1]
                break

    print(f'Loaded results for Household {household or "?"}')

    paths = []
    if is_comparison:
        paths.append(plot_accommodation_comparison(data, out_dir))
        paths.append(plot_aggregate_metrics(data, out_dir))
        paths.append(plot_object_heatmap(data, out_dir))
        paths.append(plot_metrics_over_time(data, out_dir))
        paths.append(plot_eval_time_distribution(data, out_dir))
        paths.append(plot_precision_recall_over_time(data, out_dir))
        paths.append(plot_dataset_size_over_time(data, out_dir))
        paths.append(plot_metrics_vs_training_days(data, out_dir))
    else:
        print('  (flat format detected — skipping comparison plots, generating heatmap only)')

    # ── Relocation pattern analysis (requires dataset_dir) ───────────────
    dataset_dir = args.dataset_dir
    # Auto-detect from status file next to the results JSON
    if dataset_dir is None:
        results_dir = os.path.dirname(args.results)
        status_file = os.path.join(results_dir, f'benchmark_slatepro_{household}_status.txt')
        if os.path.isfile(status_file):
            for line in open(status_file):
                if line.startswith('dataset_dir:'):
                    dataset_dir = line.split(':', 1)[1].strip()
                    break

    warmup_days = args.warmup_days
    if warmup_days is None:
        warmup_days = data.get('train_days', 65)

    vh_id_to_class: dict[int, str] = {}
    if dataset_dir and os.path.isdir(dataset_dir):
        vh_id_to_class = _vh_node_id_to_class_name(dataset_dir)
        print(f'\nAnalyzing relocation patterns from {dataset_dir} (warmup={warmup_days})')
        # result_data should have actionwise_evals_daily at top level
        result_data = data.get('results', {}).get('slatepro', data)
        analysis = _build_relocation_analysis(
            dataset_dir, result_data, warmup_days, vh_id_to_class)
        print(f'  Found {len(analysis)} unique (obj, dest) relocation types')
        n_with_eval = sum(1 for r in analysis if r['count_eval'] >= 2)
        print(f'  {n_with_eval} types with >= 2 eval occurrences')
        paths.append(plot_recall_vs_activity_complexity(analysis, out_dir, household))
        paths.append(plot_recall_by_movement_order(analysis, out_dir, household))
    else:
        if dataset_dir:
            print(f'\nWARNING: dataset_dir not found: {dataset_dir}')
        else:
            print('\nSkipping relocation pattern plots (no --dataset_dir and no status file found)')

    # ── Prediction heatmap ────────────────────────────────────────────────
    # Get result data (handle both comparison and flat formats)
    result_data = data.get('results', {}).get('slatepro', data)
    per_day_list = result_data.get('per_day', [])

    # First 'warm'-prefixed dir token holds the warmup-day count, named either
    # 'warm5' or 'warmup5' — pull the digits out (stripping the literal prefix
    # turns 'warmup5' into the un-int-able 'up5').
    warm_token = [w for w in args.results.split('_') if w.startswith('warm')][0]
    warm_start_days = int(re.search(r'\d+', warm_token).group())
    # Auto-detect finetune days (days with robot predictions)
    ft_days = [i for i, d in enumerate(per_day_list) if (d.get('total_robot', 0) or 0) > 0 and i >= warm_start_days]
    # Cap at 15 panels — sample evenly if too many
    max_panels = 15
    if len(ft_days) > max_panels:
        step = len(ft_days) / max_panels
        sampled = [ft_days[int(i * step)] for i in range(max_panels - 1)]
        sampled.append(ft_days[-1])  # always include last
        print(f'\n{len(ft_days)} finetune days — sampling {len(sampled)} evenly ...')
        ft_days = sampled
    else:
        print(f'\nGenerating stacked heatmap for {len(ft_days)} finetune days ...')

    # Load reference condition once — used by both the heatmap overlay and
    # the cross-condition plots below.
    ref_result = None
    if args.ref_results:
        if not os.path.isfile(args.ref_results):
            print(f'\nWARNING: ref_results not found: {args.ref_results}')
        else:
            print(f'\nLoading reference condition: {args.ref_results}')
            ref_result = _load_ref_result(args.ref_results)

    p = plot_prediction_heatmap(result_data, ft_days, out_dir,
                                household=household,
                                condition=args.condition,
                                ref_result=ref_result,
                                ref_label=args.ref_label,
                                vh_id_to_class=vh_id_to_class)
    if p:
        paths.append(p)
        print(f'  Saved: {os.path.basename(p)}')

    # ── Cross-condition state divergence plots ──────────────────────────
    if ref_result is not None:
        cur_label = args.condition or 'Current'
        p = plot_edit_distance_vs_outcomes(
            result_data, ref_result, out_dir, household,
            cur_label, args.ref_label)
        if p:
            paths.append(p)
            print(f'  Saved: {os.path.basename(p)}')
        p = plot_state_diff_object_correlation(
            result_data, ref_result, out_dir, household,
            cur_label, args.ref_label)
        if p:
            paths.append(p)
            print(f'  Saved: {os.path.basename(p)}')

    print(f'\nGenerated {len([p for p in paths if p])} plots in {out_dir}/')
    for p in paths:
        if p:
            print(f'  {os.path.basename(p)}')


if __name__ == '__main__':
    main()
