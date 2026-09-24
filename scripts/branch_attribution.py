#!/usr/bin/env python
"""Attribute every emitted robot action to the branch that produced it.

The hybrid policy emits from two branches and tags each candidate `source` in
`hybrid_memory_<HH>/policy_diag.jsonl`:

  current  PF / task-graph assistance for the activity in progress
  prep     transformer-predicted next-activity pre-staging

The human-side outcome of an action lives in the benchmark json's
`actionwise_evals_daily`, keyed by the action's emission time. Joining the two
on (day, emission tick, object) gives precision per branch -- the numbers in
the paper's "source" tables.

DAY INDEX: `policy_diag` counts `day_in_process` from the policy's own first
eval day, while `actionwise_evals_daily` is indexed by absolute day, so the two
differ by a constant per run (0 when the policy was constructed on day 0, else
the warm-start length). The offset is not recorded anywhere, so this script
RECOVERS it: it tries every offset and keeps the one that joins best, and
refuses to report unless that winner is decisive (>= 3x the runner-up). A wrong
offset still joins ~15% of actions by coincidence, which is exactly the failure
this guard exists to catch.

Metrics, matching the tables:
  TP        = skipped + accepted
  FP        = corrected + corrected_through_vigilance + redirected + undone
  precision = TP / (TP + FP)
  unseen (robot self-churn, the robot moved the object again before the human
  ever saw it) is excluded from precision -- it is not a human-tagged error.

With --csv-dir, the joined actions and the summary are written beside the
figures so the published table can be recomputed without the run directories
(which are archived, not shipped): <cell>_branch_actions.csv.gz holds one row
per emitted action (household, day, tick, object, source, outcome).

Run: .../python scripts/branch_attribution.py <run_dir> [<run_dir> ...]
     [--policy hybrid] [--per-household] [--csv-dir results_final/source_tables]
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

TP_OUTCOMES = {"skipped", "accepted"}
FP_OUTCOMES = {"corrected", "corrected_repeatedly", "corrected_through_vigilance",
               "redirected", "undone", "undo"}
MAX_OFFSET = 10
SOURCES = ("current", "prep")


def load_outcomes(bench_json):
    """(day, tick, object) -> [outcome, ...] from actionwise_evals_daily."""
    data = json.load(open(bench_json))
    out = defaultdict(list)
    for day, rec in enumerate(data.get("actionwise_evals_daily") or []):
        if not rec:
            continue
        for tick, actions in rec.items():
            for a in actions:
                out[(day, int(tick), a["object_name"])].append(a["human_action_outcome"])
    return out


def load_emissions(diag_jsonl):
    """[(day_in_process, tick, object, source), ...] in file order."""
    emits = []
    with open(diag_jsonl) as f:
        for line in f:
            r = json.loads(line)
            day, tick = r.get("day_in_process"), r.get("t_min")
            for c in r.get("candidates", []):
                if c.get("outcome") == "emitted":
                    emits.append((int(day), int(tick), c.get("obj"), c.get("source")))
    return emits


def best_offset(emits, outcomes):
    """-> (offset, rate, runner_up_rate). See the day-index note above."""
    scored = []
    for off in range(MAX_OFFSET + 1):
        hits = sum(1 for d, t, o, _ in emits if (d + off, t, o) in outcomes)
        scored.append((hits / len(emits) if emits else 0.0, off))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0], scored[1][0]


def tally(run, policy, households="ABC"):
    """-> per-household {source: Counter(outcome)} plus emission counts."""
    per_hh = {}
    for hh in households:
        bench = f"{run}/benchmark_{policy}_{hh}.json"
        diag = f"{run}/hybrid_memory_{hh}/policy_diag.jsonl"
        if not (os.path.isfile(bench) and os.path.isfile(diag)):
            raise SystemExit(f"missing input for HH-{hh}: {bench} / {diag}")
        outcomes = load_outcomes(bench)
        emits = load_emissions(diag)
        off, rate, runner = best_offset(emits, outcomes)
        if rate < 3 * max(runner, 1e-9):
            raise SystemExit(
                f"{run} HH-{hh}: day offset is ambiguous (best {off} at {rate:.0%}, "
                f"runner-up {runner:.0%}); refusing to report a join this weak.")
        pending = {k: list(v) for k, v in outcomes.items()}
        counts = defaultdict(Counter)
        emitted = Counter()
        rows = []
        for d, t, o, src in emits:
            emitted[src] += 1
            got = pending.get((d + off, t, o))
            outcome = got.pop(0) if got else "<unjoined>"
            counts[src][outcome] += 1
            rows.append((hh, d + off, t, o, src, outcome))
        per_hh[hh] = {"counts": counts, "emitted": emitted, "rows": rows,
                      "offset": off, "join_rate": rate}
    return per_hh


def summarise(counts, emitted):
    total = sum(emitted.values())
    tp = {s: sum(v for k, v in counts[s].items() if k in TP_OUTCOMES) for s in SOURCES}
    fp = {s: sum(v for k, v in counts[s].items() if k in FP_OUTCOMES) for s in SOURCES}
    tp_all = sum(tp.values()) or 1
    rows = {}
    for s in SOURCES:
        denom = tp[s] + fp[s]
        rows[s] = {"actions_pct": 100.0 * emitted[s] / (total or 1),
                   "tp_pct": 100.0 * tp[s] / tp_all,
                   "precision": (tp[s] / denom) if denom else float("nan"),
                   "n_actions": emitted[s], "n_tp": tp[s], "n_fp": fp[s]}
    return rows


def write_csvs(run, per_hh, pooled_rows, csv_dir):
    """Joined actions + the summary, so the table survives without the runs."""
    os.makedirs(csv_dir, exist_ok=True)
    stem = os.path.join(csv_dir, os.path.basename(run))
    with gzip.open(f"{stem}_branch_actions.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["household", "day", "tick", "object", "source", "outcome"])
        for hh in sorted(per_hh):
            w.writerows(per_hh[hh]["rows"])
    with open(f"{stem}_branch_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "source", "pct_total_actions", "pct_true_positives",
                    "precision", "n_actions", "n_tp", "n_fp"])
        for scope, rows in [("pooled", pooled_rows)] + [
                (f"HH-{hh}", summarise(d["counts"], d["emitted"])) for hh, d in sorted(per_hh.items())]:
            for s, r in rows.items():
                w.writerow([scope, s, f"{r['actions_pct']:.2f}", f"{r['tp_pct']:.2f}",
                            f"{r['precision']:.4f}", r["n_actions"], r["n_tp"], r["n_fp"]])
    print(f"  wrote {stem}_branch_actions.csv.gz and {stem}_branch_summary.csv")


def report(run, policy, per_household, csv_dir=None):
    per_hh = tally(run, policy, "ABC")
    offs = {hh: (d["offset"], d["join_rate"]) for hh, d in per_hh.items()}
    print(f"\n=== {os.path.basename(run)} ===")
    print("  day offset / join rate: " +
          "  ".join(f"HH-{hh}: +{o} {r:.0%}" for hh, (o, r) in offs.items()))
    pooled_counts, pooled_emitted = defaultdict(Counter), Counter()
    for hh, d in per_hh.items():
        for s in SOURCES:
            pooled_counts[s] += d["counts"][s]
        pooled_emitted += d["emitted"]
    unjoined = sum(c["<unjoined>"] for c in pooled_counts.values())
    print(f"  emitted {sum(pooled_emitted.values())}, unjoined {unjoined} "
          f"({100.0 * unjoined / max(sum(pooled_emitted.values()), 1):.1f}%)")
    pooled_rows = summarise(pooled_counts, pooled_emitted)
    print(f"  {'source':<9}{'% total actions':>17}{'% true positives':>18}{'precision':>12}")
    for s, r in pooled_rows.items():
        print(f"  {s:<9}{r['actions_pct']:>16.1f}%{r['tp_pct']:>17.1f}%{r['precision']:>12.3f}")
    if csv_dir:
        write_csvs(run, per_hh, pooled_rows, csv_dir)
    if per_household:
        for hh, d in per_hh.items():
            print(f"  -- HH-{hh}")
            for s, r in summarise(d["counts"], d["emitted"]).items():
                print(f"     {s:<8}{r['actions_pct']:>15.1f}%{r['tp_pct']:>17.1f}%"
                      f"{r['precision']:>12.3f}   (TP {r['n_tp']}, FP {r['n_fp']})")
    outs = Counter()
    for s in SOURCES:
        outs += pooled_counts[s]
    print("  outcomes:", dict(outs.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories to analyse")
    ap.add_argument("--policy", default="hybrid")
    ap.add_argument("--per-household", action="store_true")
    ap.add_argument("--csv-dir", default=None,
                    help="write <run>_branch_actions.csv.gz and _branch_summary.csv here")
    a = ap.parse_args()
    for run in a.runs:
        report(run, a.policy, a.per_household, a.csv_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
