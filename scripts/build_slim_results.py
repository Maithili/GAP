#!/usr/bin/env python
"""Extract the slim `totals` block from every benchmark json under the
ablation / dist-shift result dirs, so those cells survive deletion of the
raw multi-GB traces.

Companion to build_public_results.sh, which does the same for results/final
(-> results_final/). This covers the dirs that had no slim export:
rerun_Jun15, hybrid_v2, archives, discriminator.

Output: results_slim/<src_dir>/<run_dir>/<benchmark_name>.json  (~12 KB each)
        plus .done_<HH> markers and gamma_log.jsonl where present.

Reads only the leading 16 MB of each json (totals precedes the per_day /
actionwise arrays). Never falls back to a full read -- a 24 GB file would
OOM the node; such files are reported as SKIPPED instead.

Run: python scripts/build_slim_results.py
"""
import json
import os
import shutil
import sys

SRC_DIRS = ["rerun_Jun15", "hybrid_v2", "archives", "discriminator"]
OUT_ROOT = "results_slim"
HEAD_BYTES = 16_000_000


def extract_totals(path):
    """Decode just the leading `totals` object. Returns None if not in the head."""
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(HEAD_BYTES)
    i = head.find('"totals"')
    if i < 0:
        return None
    c = head.find(":", i) + 1
    while c < len(head) and head[c] in " \t\r\n":
        c += 1
    try:
        obj, _ = json.JSONDecoder().raw_decode(head, c)
        return obj
    except ValueError:
        return None


def main():
    n_ok = n_skip = 0
    skipped = []
    for src in SRC_DIRS:
        root = os.path.join("results", src)
        if not os.path.isdir(root):
            print(f"[warn] missing {root}", flush=True)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, "results")
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if fn.startswith("benchmark_") and fn.endswith(".json"):
                    totals = extract_totals(p)
                    if totals is None:
                        n_skip += 1
                        skipped.append(p)
                        continue
                    outdir = os.path.join(OUT_ROOT, rel)
                    os.makedirs(outdir, exist_ok=True)
                    with open(os.path.join(outdir, fn), "w") as f:
                        json.dump({"totals": totals}, f)
                    n_ok += 1
                elif fn.startswith(".done_") or fn == "gamma_log.jsonl":
                    outdir = os.path.join(OUT_ROOT, rel)
                    os.makedirs(outdir, exist_ok=True)
                    shutil.copy2(p, os.path.join(outdir, fn))
        print(f"[done] {src}", flush=True)

    print(f"\nextracted {n_ok} totals blocks, skipped {n_skip}")
    for s in skipped:
        print(f"  SKIPPED (no totals in first 16MB): {s}")
    return 0 if n_skip == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
