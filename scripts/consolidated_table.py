#!/usr/bin/env python
"""Consolidated results table using the clean effort ledger.

Rows = methods {slatepro, oracle, plainllm} + HybridPF variants
{nogate, v1-gate, v2-gate} x {prob_threshold 0.1, 0.5}.
Sections = the 4 settings {scripted, llm-redirect} x {act+reset, retrain}.
Averages over households (A,B,C) with a `.done_<HH>` marker.
'*' = fewer than 3 households done; '(running)' = dirs exist but none finished;
'-' = variant not run.

Ledger columns (finetune phase, avg over HH):
  solo, human, robot, savings, undo, redirected,
  net saved (abs), net saved %, harder (abs), hard%

Run: python scripts/consolidated_table.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from effort_ledger import ledger_from_finetune

RERUN = "full_results"
FULL = "results/full_set_Jun15"
INFIX = {
    "actreset": "hybrid_actreset_warmup65_test0_every99999_all1_noresetoff_modeevent",
    "retrain": "hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent",
}

ROWS = [
    ("slatepro", "base", None, None),
    ("oracle", "base", None, None),
    ("plainllm", "base", None, None),
    ("HybridPF nogate", "base", None, None),
    ("nopa", "base", None, None),
    ("HybridPF v1 pt0.1", "gate", "results/hybrid_gate", ""),
    ("HybridPF v1 pt0.5", "gate", "results/hybrid_pt05", "_pt0.5"),
    ("HybridPF v2 pt0.1", "gate", "results/hybrid_v2", ""),
    ("HybridPF v2 pt0.5", "gate", "results/hybrid_v2", "_pt0.5"),
]

SETTINGS = [
    ("SCRIPTED human  |  act+reset", "scripted", "actreset"),
    ("SCRIPTED human  |  retrain", "scripted", "retrain"),
    ("LLM human (redirect)  |  act+reset", "llm", "actreset"),
    ("LLM human (redirect)  |  retrain", "llm", "retrain"),
]

OLD_ROWS = [
    ("slatepro", "slatepro"),
    ("oracle", "oracle"),
    ("plainllm", "plainllm"),
    ("HybridPF (v6)", "hybrid"),
]

OLD_BASE = {
    ("slatepro", "scripted", "actreset"): "cond5_",
    ("slatepro", "scripted", "retrain"): "cond4_",
    ("slatepro", "llm", "actreset"): "cond21_",
    ("slatepro", "llm", "retrain"): "cond22_",
    ("oracle", "scripted", "actreset"): "cond31_oracle_pure_warmup65",
    ("oracle", "scripted", "retrain"): "cond33_oracle_pure_retrain_warmup5",
    ("oracle", "llm", "actreset"): "cond35_oracle_pure_llmhuman",
    ("oracle", "llm", "retrain"): "cond37_oracle_pure_retrain_llmhuman",
    ("plainllm", "scripted", "actreset"): "cond27_",
    ("plainllm", "scripted", "retrain"): "cond29_",
    ("plainllm", "llm", "actreset"): "cond28_",
    ("plainllm", "llm", "retrain"): "cond30_",
    ("hybrid", "scripted", "actreset"): "cond23pf6_",
    ("hybrid", "scripted", "retrain"): "cond25pf6_",
    ("hybrid", "llm", "actreset"): "cond24pf6_",
    ("hybrid", "llm", "retrain"): "cond26pf6_",
}

COLS = [
    ("solo", "solo"),
    ("human", "human"),
    ("robot", "robot"),
    ("save", "savings"),
    ("undo", "undo"),
    ("redir", "redirected"),
    ("saved", "net_abs"),
    ("saved%", "net_pct"),
    ("hard", "harder_abs"),
    ("hard%", "harder_pct"),
]
INT = {"solo", "human", "robot", "savings", "undo", "redirected", "net_abs", "harder_abs"}

# Ablation suffixes emitted by scripts/run_benchmark_nopa.sh.
NOPA_ABLATION_TAGS = ("_oursrg", "_noinvplan", "_noreturn", "_restoreall",
                      "_hpgpn", "_ftdays", "_noplangate")


def base_method(label):
    return {
        "slatepro": "slatepro",
        "oracle": "oracle",
        "plainllm": "plainllm",
        "HybridPF nogate": "hybrid",
        "nopa": "nopa",
    }[label]


def base_dirs(method, human, setting):
    out = []
    for d in sorted(glob.glob(f"{RERUN}/{method}_{setting}_*")):
        is_llm = "humanllm" in d
        is_red = "redirect" in d
        if d.endswith("_gate") or "_pt" in os.path.basename(d):
            continue
        # NOPA ablation variants share the base dir prefix - exclude them from
        # the headline row (see scripts/run_benchmark_nopa.sh ABL_TAG).
        if any(tag in os.path.basename(d) for tag in NOPA_ABLATION_TAGS):
            continue
        if human == "scripted" and is_llm:
            continue
        if human == "llm" and not (is_llm and is_red):
            continue
        out.append(d)
    return out


def gate_dir(root, pt_suffix, human, setting):
    hs = "_humanllm_redirect" if human == "llm" else ""
    return f"{root}/{INFIX[setting]}{hs}_gate{pt_suffix}"


def collect(label, kind, root, pt, human, setting):
    rows, n_done, any_dir = [], 0, False
    if kind == "base":
        method = base_method(label)
        dirs = base_dirs(method, human, setting)
        jsonname = method
    else:
        dirs = [gate_dir(root, pt, human, setting)]
        jsonname = "hybrid"
    for d in dirs:
        if not os.path.isdir(d):
            continue
        any_dir = True
        for hh in "ABC":
            if os.path.isfile(f"{d}/.done_{hh}"):
                jf = f"{d}/benchmark_{jsonname}_{hh}.json"
                if os.path.isfile(jf):
                    try:
                        m = ledger_from_finetune(jf)
                        if m is not None:
                            rows.append(m)
                            n_done += 1
                    except Exception:
                        pass
    if not any_dir:
        return None, 0, "na"
    if not rows:
        return None, 0, "running"
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return avg, n_done, ("ok" if n_done >= 3 else "partial")


def collect_old(method, human, setting):
    pref = OLD_BASE.get((method, human, setting))
    if not pref:
        return None, 0, "na"
    ds = sorted(glob.glob(f"{FULL}/{pref}*"))
    if not ds:
        return None, 0, "na"
    rows = []
    for hh in "ABC":
        cands = glob.glob(f"{ds[0]}/benchmark_*_{hh}.json")
        if cands:
            try:
                m = ledger_from_finetune(cands[0])
                if m is not None:
                    rows.append(m)
            except Exception:
                pass
    if not rows:
        return None, 0, "na"
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return avg, len(rows), ("ok" if len(rows) >= 3 else "partial")


def fmt_row(name, avg, n, status):
    tag = {"ok": "", "partial": "*", "running": "", "na": ""}[status]
    nm = f"{name}{tag}"
    if status == "na":
        return f"{nm:<20}{'-':>4}" + "".join(f"{'-':>8}" for _ in COLS)
    if status == "running":
        return f"{nm:<20}{'0':>4}" + f"{'(running - partial json, no .done)':>40}"
    cells = ""
    for _, key in COLS:
        v = avg[key]
        cells += f"{v:>8.0f}" if key in INT else f"{v:>8.2f}"
    return f"{nm:<20}{n:>4}{cells}"


def main():
    w = 20 + 4 + 8 * len(COLS)
    out = []
    out.append("=" * w)
    out.append("CONSOLIDATED RESULTS - effort ledger (finetune, avg over .done households)")
    out.append(
        "solo = script TELEPORT inventory | human = corrected*+cdc+done_as_is+redirect+undo"
    )
    out.append(
        "robot = tagged robot TELEPORTs | saved = savings-(redirect+undo) | hard = corrected*+cdc+redirect"
    )
    out.append(
        "'*' = <3 HH done | '(running)' = no HH finished | '-' = variant not run"
    )
    out.append("HybridPF v2-gate was run only for LLM-human (gate targets LLM churn).")
    out.append("=" * w)
    hdr = f"{'method':<20}{'n':>4}" + "".join(f"{h:>8}" for h, _ in COLS)
    for title, human, setting in SETTINGS:
        out.append("")
        out.append(f"### {title}   - PRIOR (full_set_Jun15)")
        out.append("-" * w)
        out.append(hdr)
        for label, method in OLD_ROWS:
            avg, n, status = collect_old(method, human, setting)
            out.append(fmt_row(label, avg, n, status))
        out.append("")
        out.append(f"### {title}   - CURRENT (rerun_Jun15 + gate variants)")
        out.append("-" * w)
        out.append(hdr)
        for label, kind, root, pt in ROWS:
            avg, n, status = collect(label, kind, root, pt, human, setting)
            out.append(fmt_row(label, avg, n, status))
    text = "\n".join(out)
    os.makedirs("results", exist_ok=True)
    with open("results/CONSOLIDATED_TABLE.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
