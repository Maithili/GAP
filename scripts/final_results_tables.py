#!/usr/bin/env python
"""Final results tables for the rerun_Jun15 grid.

Produces two artifacts (printed + written to results/):

A) FINAL table — one block per setting {scripted,llm} x {act+reset,retrain}, methods as
   rows, averaged over households (A,B,C with a .done_<HH> marker). Columns:
     TP (skipped, accepted) | FP (corrected, cdc, undo, redir) | FN (done_as_is)
     | precision | recall | Human F1 | unseen | net effort (abs, %) | harder (abs, %)

B) DELTA table — all four settings stacked, rows = the four base methods that have a
   PRIOR (full_set_Jun15) counterpart. Columns: F1, precision, recall, net%, hard%, each as
   prev -> cur with relative % change. '% chg' = 100*(cur-prev)/|prev|.

Definitions are the canonical ones from closed_loop_eval.py:_metrics:
  TP = skipped + accepted
  FP = corrected + corrected_despite_correct + vigilance_corrections(undo) + redirected
  FN = done_as_is
  precision = TP/(TP+FP) ; recall = TP/(TP+FN) ; F1 = 2TP/(2TP+FP+FN)
net/harder come from effort_ledger (canonical effort accounting).

Run: python scripts/final_results_tables.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import consolidated_table as ct  # dir-resolution + PRIOR mapping (reused, not duplicated)
from effort_ledger import ledger_from_sources
from final_deliverable import _extract_totals


def metrics_from_finetune(jf):
    """Rich per-run metric dict from the finetune totals block, or None if empty."""
    t = _extract_totals(jf).get("finetune") or {}
    hm = t.get("human_metrics") or {}
    em = t.get("env_metrics") or {}
    led = ledger_from_sources(hm, em)
    if led is None:
        return None
    g = lambda d, k: int((d or {}).get(k, 0) or 0)
    # Compute TP/FP/FN from component counts (robust to old jsons that predate the
    # precomputed tp/fp/fn keys). Identical to closed_loop_eval.py:_metrics.
    skipped, accepted = g(hm, "skipped"), g(hm, "accepted")
    corrected, cdc = g(hm, "corrected"), g(hm, "corrected_despite_correct")
    undo, redir = g(hm, "vigilance_corrections"), g(hm, "redirected")
    tp = skipped + accepted
    fp = corrected + cdc + undo + redir
    fn = g(hm, "done_as_is")
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    unseen = g(em, "unseen") + g(em, "unseen_at_end_of_day")
    return {
        "tp": tp, "skipped": g(hm, "skipped"), "accepted": g(hm, "accepted"),
        "fp": fp, "corrected": g(hm, "corrected"), "cdc": g(hm, "corrected_despite_correct"),
        "undo": g(hm, "vigilance_corrections"), "redir": g(hm, "redirected"),
        "fn": fn, "done_as_is": g(hm, "done_as_is"),
        "prec": prec, "recall": rec, "f1": f1, "unseen": unseen,
        "net_abs": led["net_abs"], "net_pct": led["net_pct"],
        "hard_abs": led["harder_abs"], "hard_pct": led["harder_pct"],
    }


def _avg(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def collect_cur(label, kind, root, pt, human, setting):
    if kind == "base":
        method = ct.base_method(label)
        dirs = ct.base_dirs(method, human, setting)
        jn = method
    else:
        dirs = [ct.gate_dir(root, pt, human, setting)]
        jn = "hybrid"
    rows, any_dir = [], False
    for d in dirs:
        if not os.path.isdir(d):
            continue
        any_dir = True
        for hh in "ABC":
            if os.path.isfile(f"{d}/.done_{hh}"):
                jf = f"{d}/benchmark_{jn}_{hh}.json"
                if os.path.isfile(jf):
                    m = metrics_from_finetune(jf)
                    if m is not None:
                        rows.append(m)
    if not any_dir:
        return None, 0, "na"
    if not rows:
        return None, 0, "running"
    return _avg(rows), len(rows), ("ok" if len(rows) >= 3 else "partial")


def collect_old(method, human, setting):
    pref = ct.OLD_BASE.get((method, human, setting))
    if not pref:
        return None, 0, "na"
    ds = sorted(glob.glob(f"{ct.FULL}/{pref}*"))
    if not ds:
        return None, 0, "na"
    rows = []
    for hh in "ABC":
        cands = glob.glob(f"{ds[0]}/benchmark_*_{hh}.json")
        if cands:
            m = metrics_from_finetune(cands[0])
            if m is not None:
                rows.append(m)
    if not rows:
        return None, 0, "na"
    return _avg(rows), len(rows), ("ok" if len(rows) >= 3 else "partial")


# ── A) FINAL table ──────────────────────────────────────────────────────────
FCOLS = [
    ("TP", "tp", 0), ("skip", "skipped", 0), ("acc", "accepted", 0),
    ("FP", "fp", 0), ("corr", "corrected", 0), ("cdc", "cdc", 0),
    ("undo", "undo", 0), ("redir", "redir", 0),
    ("FN", "fn", 0),
    ("prec", "prec", 3), ("rec", "recall", 3), ("F1", "f1", 3),
    ("unseen", "unseen", 0),
    ("net", "net_abs", 0), ("net%", "net_pct", 1),
    ("hard", "hard_abs", 0), ("hard%", "hard_pct", 1),
]


def fmt_final_row(name, avg, n, status):
    tag = "*" if status == "partial" else ""
    nm = f"{name}{tag}"
    if status == "na":
        return f"{nm:<20}{'-':>3}" + "".join(f"{'-':>8}" for _ in FCOLS)
    if status == "running":
        return f"{nm:<20}{'0':>3}  (running - no household finished)"
    cells = ""
    for _, key, dec in FCOLS:
        v = avg[key]
        cells += f"{v:>8.{dec}f}"
    return f"{nm:<20}{n:>3}{cells}"


def final_tables():
    w = 20 + 3 + 8 * len(FCOLS)
    out = ["=" * w,
           "FINAL RESULTS  (rerun_Jun15, finetune phase, averaged over .done households)",
           "TP=skip+acc | FP=corr+cdc+undo+redir | FN=done_as_is | "
           "net=savings-(undo+redir) | hard=corr+corr_rep+cdc+redir",
           "'*'=<3 HH done | 'n'=households averaged | '-'=not run",
           "=" * w]
    grp = (f"{'':<23}{'':>8}" + f"{'<------ TP ------>':>24}"
           + f"{'<------------ FP ------------>':>32}" + f"{'<-FN->':>8}")
    hdr = f"{'method':<20}{'n':>3}" + "".join(f"{h:>8}" for h, _, _ in FCOLS)
    for title, human, setting in ct.SETTINGS:
        out += ["", f"### {title}", grp, "-" * w, hdr]
        for label, kind, root, pt in ct.ROWS:
            avg, n, st = collect_cur(label, kind, root, pt, human, setting)
            out.append(fmt_final_row(label, avg, n, st))
    return "\n".join(out)


# ── B) DELTA table ──────────────────────────────────────────────────────────
DELTA_METRICS = [("F1", "f1"), ("prec", "prec"), ("recall", "recall"),
                 ("net%", "net_pct"), ("hard%", "hard_pct")]
# delta rows = base methods with a PRIOR counterpart (label, ct-method-key)
DELTA_ROWS = [("slatepro", "slatepro"), ("oracle", "oracle"),
              ("plainllm", "plainllm"), ("HybridPF", "hybrid")]
# map the delta method -> current-grid ROWS entry (hybrid PRIOR=v6 -> current nogate)
CUR_OF = {
    "slatepro": ("slatepro", "base", None, None),
    "oracle": ("oracle", "base", None, None),
    "plainllm": ("plainllm", "base", None, None),
    "hybrid": ("HybridPF nogate", "base", None, None),
}


def _pct_chg(prev, cur):
    if prev is None or cur is None:
        return None
    if abs(prev) < 1e-9:
        return None
    return 100.0 * (cur - prev) / abs(prev)


def delta_tables():
    cell_w = 26  # "prev->cur (+x%)"
    w = 20 + len(DELTA_METRICS) * cell_w
    out = ["", "=" * w,
           "DELTA vs PRIOR (full_set_Jun15).  cell = prev -> cur  (rel % change)",
           "rel % chg = 100*(cur-prev)/|prev|.  net%/hard% are themselves percentages.",
           "=" * w]
    hdr = f"{'method':<20}" + "".join(f"{h:>{cell_w}}" for h, _ in DELTA_METRICS)
    for title, human, setting in ct.SETTINGS:
        out += ["", f"### {title}", "-" * w, hdr]
        for label, mkey in DELTA_ROWS:
            prev, _, pst = collect_old(mkey, human, setting)
            cl, knd, rt, pt = CUR_OF[mkey]
            cur, _, cst = collect_cur(cl, knd, rt, pt, human, setting)
            cells = ""
            for _, key in DELTA_METRICS:
                pv = prev[key] if prev else None
                cv = cur[key] if cur else None
                if pv is None and cv is None:
                    cells += f"{'-':>{cell_w}}"
                    continue
                ch = _pct_chg(pv, cv)
                pvs = f"{pv:.2f}" if pv is not None else "—"
                cvs = f"{cv:.2f}" if cv is not None else "—"
                chs = f"({ch:+.0f}%)" if ch is not None else ""
                cells += f"{pvs}->{cvs}{chs}".rjust(cell_w)
            out.append(f"{label:<20}{cells}")
    return "\n".join(out)


def main():
    text = final_tables() + "\n" + delta_tables() + "\n"
    os.makedirs("results", exist_ok=True)
    with open("results/FINAL_RESULTS_TABLES.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
