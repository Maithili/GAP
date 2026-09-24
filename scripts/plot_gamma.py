"""Plot the v2-PF continual-learning gamma trajectory for a run.

Reads <memory_dir>/gamma_log.jsonl (one line per retrain round: chosen gamma
+ per-candidate prequential scores) and writes gamma_plot.png next to it:
top panel = chosen gamma per round (the 'memory tightens when the routine
shifts' figure), bottom panel = prequential log-lik per candidate per round.

Usage:
  python scripts/plot_gamma.py --memory_dir results/<run>/hybrid_memory_A
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory_dir', required=True)
    args = ap.parse_args()

    log_p = os.path.join(args.memory_dir, 'gamma_log.jsonl')
    rows = [json.loads(l) for l in open(log_p)]
    if not rows:
        raise RuntimeError(f"empty {log_p}")

    rounds = list(range(len(rows)))
    chosen = [r['gamma'] for r in rows]
    n_days = [r['n_days'] for r in rows]
    grid = sorted({float(g) for r in rows for g in r['scores']})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.step(rounds, chosen, where='mid', marker='o')
    ax1.set_ylabel('chosen gamma (per-day decay)')
    ax1.set_ylim(min(grid) - 0.02, 1.02)
    ax1.grid(alpha=0.3)
    half = [0.693 / max(1e-9, 1 - g) for g in chosen]
    for x, (g, h, n) in enumerate(zip(chosen, half, n_days)):
        ax1.annotate(f"{h:.0f}d" if g < 1 else "inf",
                     (x, g), textcoords='offset points', xytext=(0, 8),
                     fontsize=7, ha='center')
    ax1.set_title('v2-PF memory half-life per retrain round '
                  '(annotation = half-life in days)')

    for g in grid:
        ys = [r['scores'].get(str(g), r['scores'].get(g)) for r in rows]
        ax2.plot(rounds, ys, marker='.', label=f'gamma={g}')
    ax2.set_xlabel('retrain round')
    ax2.set_ylabel('prequential log-lik (training-side)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    out = os.path.join(args.memory_dir, 'gamma_plot.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
