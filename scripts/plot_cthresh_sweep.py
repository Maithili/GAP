#!/usr/bin/env python
"""Two-panel figure for the GAP confidence-threshold (c_thresh) sweep.

Left : Net Saved % and Perturbed % vs c_thresh. Both are percentages of the same
       solo-effort denominator, so they share ONE y axis (never a dual axis).
Right: the same sweep as a precision-recall curve, in threshold order.

Points are macro-averages over households A/B/C; error bars / the shaded band are
the sample std (ddof=1) across the three households, matching the paper tables.
A threshold is plotted only if its run dir has all 3 .done_<HH> markers.

Run: .../python scripts/plot_cthresh_sweep.py [--out figures/cthresh_sweep]
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from final_results_tables import metrics_from_finetune

# The sweep runs are archived (cthresh_sweep.tar.zst), not kept on disk: restore
# them beside this path to regenerate the figure. The plotted numbers are in the
# CSVs next to it, so the figure can be read without them.
SWEEP = "results/hybrid_v2_sweep/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent"
# c_thresh 0.1 is the config default: the main GAP run carries no _pt tag.
BASE = ("full_results/hybrid_retrain_warmup5_test0_every5_all1_noreset1"
        "_modeevent_gate_pbg0.001_noleadgate")
OPERATING_POINT = 0.1

# Categorical slots 1-2 of the validated reference palette, in fixed order.
# Adjacent-pair separation checked (OKLab dE x100): normal 33.6, protan 26.5,
# deutan 33.3, tritan 31.0 -- all above the 15 / 8 floors.
# Colour carries ONE meaning across both panels: blue = Net Saved, orange =
# Perturbed, neutral ink = "the sweep itself" / the chosen operating point.
# The right panel is a single series (one curve through the same runs), so it
# takes neutral ink rather than reusing a hue that means a measure on the left.
C_NET, C_PERT = "#2a78d6", "#eb6834"          # marks
# Text variants, darkened to clear 4.5:1 on white (marks need no text contrast):
# #2a78d6 -> 4.42 fails, #2468b8 -> 5.60; #eb6834 -> 3.20 fails, #b8431a -> 5.45.
T_NET, T_PERT = "#2468b8", "#b8431a"
C_CURVE = "#52514e"                            # PR curve: single series, neutral
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def run_dir(t):
    return BASE if abs(t - OPERATING_POINT) < 1e-9 else f"{SWEEP}_gate_pt{t}_pbg0.001_noleadgate"


# Columns of the raw per-household dump: the component counts the metrics are
# built from, so every plotted number can be recomputed from this file alone.
PER_HH_FIELDS = ("tp", "skipped", "accepted", "fp", "corrected", "cdc", "undo",
                 "redir", "fn", "done_as_is", "unseen", "net_abs", "net_pct",
                 "hard_abs", "hard_pct", "prec", "recall", "f1")


def write_csvs(rows, raw, out):
    """Raw per-household rows and the plotted macro summary, beside the figure."""
    with open(f"{out}_per_household.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c_thresh", "household", "run_dir"] + list(PER_HH_FIELDS))
        for t, hh, d, mt in raw:
            w.writerow([t, hh, d] + [mt[k] for k in PER_HH_FIELDS])
    keys = ("net_pct", "hard_pct", "prec", "recall", "f1")
    with open(f"{out}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c_thresh", "n_households"]
                   + [f"{k}_{s}" for k in keys for s in ("mean", "sd")])
        for r in rows:
            w.writerow([r["t"], 3] + [r[k + suf] for k in keys for suf in ("", "_sd")])
    print(f"wrote {out}_per_household.csv ({len(raw)} rows) and {out}_summary.csv ({len(rows)} rows)")


def collect(thresholds):
    """-> (rows, skipped, raw). Each row: threshold + mean/std of the metrics."""
    rows, skipped, raw = [], [], []
    for t in thresholds:
        d = run_dir(t)
        n_mk = len(glob.glob(f"{d}/.done_?"))
        if n_mk != 3:
            skipped.append((t, "missing dir" if not os.path.isdir(d) else f"{n_mk}/3 markers"))
            continue
        per_hh = [metrics_from_finetune(f"{d}/benchmark_hybrid_{h}.json") for h in "ABC"]
        if any(m is None for m in per_hh):
            skipped.append((t, "totals unreadable")); continue
        raw += [(t, h, d, mt) for h, mt in zip("ABC", per_hh)]
        row = {"t": t}
        for k in ("net_pct", "hard_pct", "prec", "recall", "f1"):
            vals = [m[k] for m in per_hh]
            row[k], row[k + "_sd"] = st.mean(vals), st.stdev(vals)
        rows.append(row)
    return rows, skipped, raw


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/cthresh_sweep")
    ap.add_argument("--thresholds", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    a = ap.parse_args()
    ts = [float(x) for x in a.thresholds.split(",")]
    rows, skipped, raw = collect(ts)

    print(f"{'c_thresh':>9}  {'Net Saved %':>16}  {'Perturbed %':>16}  {'Prec':>6}  {'Recall':>7}")
    for r in rows:
        print(f"{r['t']:>9.1f}  {r['net_pct']:>7.1f} ± {r['net_pct_sd']:<6.1f}  "
              f"{r['hard_pct']:>7.1f} ± {r['hard_pct_sd']:<6.1f}  {r['prec']:>6.3f}  {r['recall']:>7.3f}")
    if skipped:
        print("\nnot plotted (incomplete):")
        for t, why in skipped:
            print(f"  c_thresh={t}: {why}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_csvs(rows, raw, a.out)
    if len(rows) < 2:
        print("\nfewer than 2 complete thresholds - no figure written.")
        return 1

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    fig.patch.set_facecolor("white")

    # ---- left: both percentages against one shared y axis -------------------
    x = [r["t"] for r in rows]
    for key, color, tcolor, label in (("net_pct", C_NET, T_NET, "Net Saved %"),
                                      ("hard_pct", C_PERT, T_PERT, "Perturbed %")):
        y = [r[key] for r in rows]
        sd = [r[key + "_sd"] for r in rows]
        axl.fill_between(x, [v - s for v, s in zip(y, sd)], [v + s for v, s in zip(y, sd)],
                         color=color, alpha=0.13, linewidth=0)
        axl.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5.5,
                 markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
    if any(abs(r["t"] - OPERATING_POINT) < 1e-9 for r in rows):
        axl.axvline(OPERATING_POINT, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)),
                    alpha=0.55, zorder=1)
        axl.annotate(f"operating point {OPERATING_POINT}", (OPERATING_POINT, axl.get_ylim()[1]),
                     textcoords="offset points", xytext=(5, -10), color=INK2, fontsize=8)
    style(axl)
    axl.set_xlabel("confidence threshold $c_{thresh}$", color=INK2, fontsize=10)
    axl.set_ylabel("% of solo human effort", color=INK2, fontsize=10)
    axl.set_title("Effort saved vs. effort disturbed", color=INK, fontsize=11,
                  loc="left", fontweight="medium")
    leg = axl.legend(frameon=False, fontsize=9.5, loc="upper right")
    for txt, tc in zip(leg.get_texts(), (T_NET, T_PERT)):
        txt.set_color(tc)
    axl.margins(x=0.06)

    # ---- right: precision-recall in threshold order -------------------------
    rec = [r["recall"] for r in rows]
    pre = [r["prec"] for r in rows]
    axr.plot(rec, pre, color=C_CURVE, linewidth=2.0, marker="o", markersize=5.5,
             markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    # selective labels only: the ends and the operating point
    to_label = {rows[0]["t"], rows[-1]["t"], OPERATING_POINT}
    for r in rows:
        if r["t"] in to_label:
            is_op = abs(r["t"] - OPERATING_POINT) < 1e-9
            # the operating point is called out by SHAPE (hollow ring), not by a
            # second hue -- blue/orange already mean the two measures on the left.
            axr.plot([r["recall"]], [r["prec"]], marker="o", markersize=11 if is_op else 8.5,
                     markerfacecolor="white" if is_op else C_CURVE,
                     markeredgecolor=INK if is_op else "white",
                     markeredgewidth=2.0 if is_op else 1.4, zorder=4)
            axr.annotate(f"$c_{{thresh}}$={r['t']:.1f}" + ("  (used)" if is_op else ""),
                         (r["recall"], r["prec"]), textcoords="offset points",
                         xytext=(9, 7), fontsize=8.5,
                         color=INK if is_op else INK2,
                         fontweight="medium" if is_op else "normal")
    style(axr)
    axr.set_xlabel("recall", color=INK2, fontsize=10)
    axr.set_ylabel("precision", color=INK2, fontsize=10)
    axr.set_title("Precision-recall across the sweep", color=INK, fontsize=11,
                  loc="left", fontweight="medium")
    axr.margins(0.17)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    print(f"\nwrote {a.out}.pdf and {a.out}.png ({len(rows)} thresholds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
