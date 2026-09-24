#!/usr/bin/env python
"""The paper's three metrics over the course of a run, one figure per user model.

Every reported run retrains every 5 days, and the benchmark json records a
`finetune_by_ckpt` block per 5-day window. This plots those windows, so the x
axis is the evaluation day and each point is that window's metrics. Values are
macro means over households A/B/C -- per household first, then averaged, as in
the paper's tables -- and the shaded band is the sample sd across households,
which is what the paper's +/- intervals are.

Source is `results_final/`, not the run directories: the exported slim totals
carry `finetune_by_ckpt`, so the figure can be rebuilt from what the repo ships
after the full runs are archived.

Colour: the validated categorical palette, slots 1-5 in a fixed order that does
not depend on how the methods rank. The oracle is a ceiling rather than a
competing method, so it takes neutral ink and a dashed line -- role encoded by
shape, not by a sixth hue.

Run: .../python scripts/plot_over_time.py [--out figures/over_time]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from effort_ledger import ledger_from_sources

# (label, cell prefix, colour) -- fixed order, assigned by entity.
METHODS = [
    ("GAP (ours)", "hybrid",   "#2a78d6"),
    ("SLaTe-PRO",  "slatepro", "#eb6834"),
    ("STREAK",     "streak",   "#1baf7a"),
    ("Plain LLM",  "plainllm", "#eda100"),
    ("NOPA",       "nopa",     "#e87ba4"),
]
ORACLE = ("Oracle (ceiling)", "oracle", "#52514e")
USERS = [("scripted", "scripted", "scripted user"),
         ("llm", "llm_human", "LLM user")]
PANELS = [("net_pct", "net actions saved (%)", "Net actions saved"),
          ("hard_pct", "perturbed actions (%)", "Perturbed actions"),
          ("f1", "F1", "F1")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def window_metrics(entry):
    """-> {net_pct, hard_pct, f1} for one 5-day window, or None if it is empty."""
    hm = entry.get("human_metrics") or {}
    em = entry.get("env_metrics") or {}
    led = ledger_from_sources(hm, em)
    if led is None:
        return None
    g = lambda d, k: int((d or {}).get(k, 0) or 0)
    tp = g(hm, "skipped") + g(hm, "accepted")
    fp = (g(hm, "corrected") + g(hm, "corrected_despite_correct")
          + g(hm, "vigilance_corrections") + g(hm, "redirected"))
    fn = g(hm, "done_as_is")
    return {"net_pct": led["net_pct"], "hard_pct": led["harder_pct"],
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-9)}


def series(cell):
    """-> {day: {metric: (mean, sd)}} across households for one exported cell."""
    per_hh = {}
    for hh in "ABC":
        p = f"results_final/{cell}/results_{hh}.json"
        if not os.path.isfile(p):
            continue
        blocks = (json.load(open(p)).get("totals") or {}).get("finetune_by_ckpt") or {}
        for ck, entry in blocks.items():
            m = window_metrics(entry)
            if m is None:
                continue
            day = entry.get("window_start_day_idx")
            if day is None:
                day = int(ck)
            per_hh.setdefault(day, []).append(m)
    out = {}
    for day, ms in sorted(per_hh.items()):
        if len(ms) < 2:                      # a single household is not a mean
            continue
        out[day] = {k: (st.mean([m[k] for m in ms]), st.stdev([m[k] for m in ms]))
                    for k in ("net_pct", "hard_pct", "f1")}
    return out


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def draw(user_suffix, title, out_path):
    curves = []
    for label, cell, colour in METHODS + [ORACLE]:
        data = series(f"{cell}_{user_suffix}")
        if not data:
            print(f"  skipped {label}: no exported windows")
            continue
        curves.append((label, colour, data, label == ORACLE[0]))
    if not curves:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    fig.patch.set_facecolor("white")
    for ax, (key, ylabel, panel_title) in zip(axes, PANELS):
        for label, colour, data, is_oracle in curves:
            xs = sorted(data)
            ys = [data[d][key][0] for d in xs]
            sd = [data[d][key][1] for d in xs]
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                            [y + s for y, s in zip(ys, sd)],
                            color=colour, alpha=0.10, linewidth=0)
            ax.plot(xs, ys, color=colour, linewidth=2.0,
                    linestyle=(0, (5, 2)) if is_oracle else "-",
                    marker="o", markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.8, label=label, zorder=3)
        if key != "f1":
            ax.axhline(0, color=INK2, linewidth=1.0, alpha=0.45, zorder=1)
        style(ax)
        ax.set_xlabel("evaluation day", color=INK2, fontsize=10)
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
        ax.set_title(panel_title, color=INK, fontsize=11, loc="left", fontweight="medium")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=len(labels),
                     loc="upper center", bbox_to_anchor=(0.5, 1.06))
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.012, ha="left", y=1.14,
                 fontweight="medium")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path} ({len(curves)} methods, "
          f"{len(next(iter(curves))[2])} windows)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/over_time")
    a = ap.parse_args()
    ok = True
    for name, suffix, title in USERS:
        print(f"{title}:")
        ok &= draw(suffix, f"Per-window metrics, {title} (5-day retraining windows)",
                   f"{a.out}/retrain_{name}.png")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
