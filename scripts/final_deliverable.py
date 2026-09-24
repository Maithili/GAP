#!/usr/bin/env python
"""FINAL deliverable: 4 per-setting result tables + a stacked delta-vs-baseline table.

Settings = {scripted, llm(redirect)} x {act+reset, retrain}. Methods as rows:
slatepro, oracle, plainllm, hybrid (canonical non-gate, from the 48-grid), and
hybrid+gate (the validated headline config, prob_threshold=0.1, from hybrid_gate).
Averaged over households A/B/C that are done; '*' marks n<3 (provisional).

Baselines for the delta table come from results/full_set_Jun15 (the prior numbers):
  slatepro: cond5/cond4/cond21/cond22 ; plainllm: cond27/cond29/cond28/cond30
  oracle (pure): cond31/cond33/cond35/cond37 ; hybrid (v6): cond23pf6/cond25pf6/cond24pf6/cond26pf6
Writes everything to results/FINAL_TABLES.txt.
"""
import glob
import json
import os

GRID = "full_results"
GATE = "results/hybrid_gate"
FULL = "results/full_set_Jun15"
OUT = "results/FINAL_TABLES.txt"

SETTINGS = [("scripted", "actreset"), ("scripted", "retrain"),
            ("llm", "actreset"), ("llm", "retrain")]

# (display label, source root, method-token, gate-flag) for the new-grid rows
ROWS = [
    ("slatepro", GRID, "slatepro", False),
    ("oracle",   GRID, "oracle",   False),
    ("plainllm", GRID, "plainllm", False),
    ("hybrid",   GRID, "hybrid",   False),
    ("hybrid+gate", GATE, "hybrid", True),
]

# baseline condition prefix in full_set_Jun15, keyed (method, human, setting)
BASE = {
    ("slatepro", "scripted", "actreset"): "cond5_",
    ("slatepro", "scripted", "retrain"):  "cond4_",
    ("slatepro", "llm", "actreset"):      "cond21_",
    ("slatepro", "llm", "retrain"):       "cond22_",
    ("oracle", "scripted", "actreset"):   "cond31_oracle_pure_warmup65",
    ("oracle", "scripted", "retrain"):    "cond33_oracle_pure_retrain_warmup5",
    ("oracle", "llm", "actreset"):        "cond35_oracle_pure_llmhuman",
    ("oracle", "llm", "retrain"):         "cond37_oracle_pure_retrain_llmhuman",
    ("plainllm", "scripted", "actreset"): "cond27_",
    ("plainllm", "scripted", "retrain"):  "cond29_",
    ("plainllm", "llm", "actreset"):      "cond28_",
    ("plainllm", "llm", "retrain"):       "cond30_",
    ("hybrid", "scripted", "actreset"):   "cond23pf6_",
    ("hybrid", "scripted", "retrain"):    "cond25pf6_",
    ("hybrid", "llm", "actreset"):        "cond24pf6_",
    ("hybrid", "llm", "retrain"):         "cond26pf6_",
}


def _extract_totals(path):
    """Read only the leading chunk and decode just the `totals` object.
    `totals` precedes the multi-GB per_day/actionwise arrays, so a few MB
    suffices — avoids loading ~1GB files fully."""
    with open(path) as f:
        data = f.read(16_000_000)
    i = data.find('"totals"')
    if i < 0:
        return json.load(open(path))["totals"]
    c = data.find(':', i) + 1
    while c < len(data) and data[c] in ' \t\r\n':
        c += 1
    try:
        obj, _ = json.JSONDecoder().raw_decode(data, c)
        return obj
    except ValueError:
        return json.load(open(path))["totals"]


def cell_metrics(jf):
    t = _extract_totals(jf)["finetune"]
    hm = t.get("human_metrics", {}) or {}
    em = t.get("env_metrics", {}) or {}
    g = lambda k: hm.get(k, 0) or 0
    skipped, acc = g("skipped"), g("accepted")
    corr, cdc = g("corrected"), g("corrected_despite_correct")
    vig = hm.get("vigilance_corrections", em.get("vigilance_corrections", 0)) or 0
    red = hm.get("redirected", em.get("redirected", 0)) or 0
    done = g("done_as_is")
    unseen = em.get("unseen", 0) or 0
    tp, fp, fn = skipped + acc, corr + cdc + vig + red, done
    solo = skipped + acc + corr + cdc + red + done
    net = (skipped + acc) - vig
    harder = corr + cdc + red
    return dict(skipped=skipped, accepted=acc, tp=tp, corr=corr, cdc=cdc,
                undo=vig, redir=red, fp=fp, fn=fn,
                prec=tp/(tp+fp) if tp+fp else 0.0,
                rec=tp/(tp+fn) if tp+fn else 0.0,
                f1=2*tp/(2*tp+fp+fn) if tp else 0.0, unseen=unseen,
                net_abs=net, net_pct=100*net/solo if solo else 0.0,
                hard_abs=harder, hard_pct=100*harder/solo if solo else 0.0)


def hh_dirs(root, method, human, setting, gate):
    out = []
    for d in sorted(glob.glob(f"{root}/{method}_{setting}_*")):
        is_llm = "humanllm" in d
        is_red = "redirect" in d
        is_gate = d.endswith("_gate")
        if human == "scripted" and is_llm:
            continue
        if human == "llm" and not (is_llm and is_red):
            continue
        if gate != is_gate:
            continue
        out.append(d)
    return out


def avg_new(label, root, method, human, setting, gate):
    rows = []
    for d in hh_dirs(root, method, human, setting, gate):
        for hh in "ABC":
            if os.path.isfile(f"{d}/.done_{hh}"):
                jf = f"{d}/benchmark_{method}_{hh}.json"
                if os.path.isfile(jf):
                    try:
                        rows.append(cell_metrics(jf))
                    except (json.JSONDecodeError, KeyError):
                        pass
    if not rows:
        return None, 0
    return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}, len(rows)


def avg_base(method, human, setting):
    pref = BASE.get((method, human, setting))
    if not pref:
        return None, 0
    ds = sorted(glob.glob(f"{FULL}/{pref}*"))
    if not ds:
        return None, 0
    d = ds[0]
    rows = []
    for hh in "ABC":
        cands = glob.glob(f"{d}/benchmark_*_{hh}.json")
        if cands:
            try:
                rows.append(cell_metrics(cands[0]))
            except (json.JSONDecodeError, KeyError):
                pass
    if not rows:
        return None, 0
    return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}, len(rows)


COLS = [("skip", "skipped", 5, 0), ("acc", "accepted", 4, 0), ("TP", "tp", 5, 0),
        ("corr", "corr", 5, 0), ("cdc", "cdc", 4, 0), ("undo", "undo", 5, 0),
        ("redir", "redir", 6, 0), ("FP", "fp", 5, 0), ("FN", "fn", 5, 0),
        ("prec", "prec", 6, 3), ("rec", "rec", 6, 3), ("F1", "f1", 6, 3),
        ("unseen", "unseen", 7, 0), ("netAbs", "net_abs", 7, 0), ("net%", "net_pct", 6, 1),
        ("hardAbs", "hard_abs", 8, 0), ("hard%", "hard_pct", 6, 1)]


def pct_change(new, base):
    if base is None or base == 0:
        return "  n/a"
    return f"{100*(new-base)/abs(base):+6.0f}%"


def main():
    L = []
    L.append("=" * 132)
    L.append("FINAL RESULTS — 48-cell grid (full_results) + hybrid+gate (results/hybrid_gate).")
    L.append("hybrid = canonical non-gate (grid); hybrid+gate = prob_threshold=0.1 accept-gate (headline). "
             "* = n<3 HH (provisional).")
    L.append("=" * 132)
    for human, setting in SETTINGS:
        title = f"{human.upper()} human  |  {'act+reset' if setting=='actreset' else 'retrain'}"
        L.append(f"\n### {title}  (avg over done households)")
        hdr = f"{'method':12}{'n':>4} " + " ".join(f"{c[0]:>{c[2]}}" for c in COLS)
        L.append(hdr)
        L.append("-" * len(hdr))
        for label, root, method, gate in ROWS:
            a, n = avg_new(label, root, method, human, setting, gate)
            star = "" if n == 3 else "*"
            lbl = f"{label}{star}"
            if n == 0:
                L.append(f"{lbl:12}{'0/3':>4} " + " ".join(f"{'*':>{c[2]}}" for c in COLS))
                continue
            L.append(f"{lbl:12}{str(n)+'/3':>4} " + " ".join(
                f"{a[c[1]]:>{c[2]}.{c[3]}f}" for c in COLS))

    # ---- delta table ----
    L.append("\n\n" + "=" * 120)
    L.append("DELTA vs PRIOR NUMBERS (results/full_set_Jun15).  new value (Δ% = relative change from baseline).")
    L.append("baseline conds: slatepro cond5/4/21/22 | plainllm cond27/29/28/30 | oracle(pure) cond31/33/35/37 "
             "| hybrid(v6) cond23pf6/25pf6/24pf6/26pf6")
    L.append("=" * 120)
    dh = f"{'setting':22}{'method':12}" + "".join(f"{m:>22}" for m in ["F1", "precision", "recall", "net%", "harder%"])
    L.append(dh)
    L.append("-" * len(dh))
    for human, setting in SETTINGS:
        sname = f"{human}/{'act+reset' if setting=='actreset' else 'retrain'}"
        for label, root, method, gate in ROWS:
            a, n = avg_new(label, root, method, human, setting, gate)
            # baseline only defined for non-gate canonical methods
            b, bn = (avg_base(method, human, setting) if not gate else (None, 0))
            if n == 0:
                L.append(f"{sname:22}{label:12}" + f"{'(pending)':>22}")
                continue
            cells = []
            for key in ["f1", "prec", "rec", "net_pct", "hard_pct"]:
                nv = a[key]
                if b is not None:
                    cells.append(f"{nv:7.3f} ({pct_change(nv, b[key])})")
                else:
                    cells.append(f"{nv:7.3f} ( base?)" if gate else f"{nv:7.3f} (  n/a)")
            star = "" if n == 3 else "*"
            L.append(f"{sname:22}{(label+star):12}" + "".join(f"{c:>22}" for c in cells))
        L.append("")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"WROTE {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
