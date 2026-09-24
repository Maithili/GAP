#!/bin/bash
# Assemble a GitHub-shareable results_final/ from results/final/.
# Per model_[human] dir we keep:
#   - results_<HH>.json  : the slim ~12KB 'totals' block (all reported metrics +
#                          per-5-day-window trajectory) extracted from the multi-GB
#                          raw benchmark json (raw jsons are NOT shipped: 1-25 GB each).
#   - checkpoints_<HH>/  : weights_day*.pt only (inference weights; all <3 MB).
#                          Lightning pl_checkpoints_* and *_resume.pt are skipped.
#   - memory_<HH>/       : hybrid curator artifacts (task_graphs, activities,
#                          tagged_days, gamma_log, ...). Hybrid only.
#   - heatmaps/          : SUBSAMPLED — combined per-window prediction heatmaps
#                          (one PNG per checkpoint window per HH); per-day frames dropped.
#   - llm_audit_<HH>.txt : full human-LLM transcripts. LLM-human dirs only.
# Skipped entirely: raw benchmark_*.json, *_status.txt, llm_budget_*.txt, other debug plots.
set -uo pipefail
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
PY="${COADAPT_PYTHON:-python}"
SRC=results/final
OUT=results_final
mkdir -p "$OUT"

for s in "$SRC"/*/; do
  model=$(basename "$s")                      # e.g. hybrid_scripted
  method="${model%_scripted}"; method="${method%_llm_human}"   # -> hybrid
  d="$OUT/$model"; mkdir -p "$d"
  echo "### $model (method=$method)"

  for HH in A B C; do
    jf="$s/benchmark_${method}_${HH}.json"
    [ -f "$jf" ] || continue
    # slim totals -> results_<HH>.json
    "$PY" - "$jf" "$d/results_${HH}.json" <<'PYEOF'
import sys, json; sys.path.insert(0,'scripts')
from benchmark_plot_io import load_benchmark_plot_data
pd = load_benchmark_plot_data(sys.argv[1])
json.dump({"totals": pd["totals"]}, open(sys.argv[2],"w"))
PYEOF
    # inference weights only
    ck="$s/model_checkpoints_benchmark_${method}_${HH}"
    if ls "$ck"/weights_day*.pt >/dev/null 2>&1; then
      mkdir -p "$d/checkpoints_${HH}"; cp "$ck"/weights_day*.pt "$d/checkpoints_${HH}/"
    fi
    # hybrid curator memory
    if [ -d "$s/hybrid_memory_${HH}" ]; then
      mkdir -p "$d/memory_${HH}"; cp -r "$s/hybrid_memory_${HH}"/. "$d/memory_${HH}/"
    fi
    # llm audit transcript (LLM-human dirs only)
    [ -f "$s/llm_audit_benchmark_${method}_${HH}.txt" ] && cp "$s/llm_audit_benchmark_${method}_${HH}.txt" "$d/llm_audit_${HH}.txt"
  done

  # subsampled heatmaps: combined per-window PNGs, HH-A only (one household across
  # every training checkpoint). Drops per-day debug_*_dayN frames and HH-B/C.
  if ls "$s"/prediction_heatmap_*_HHA_*.png >/dev/null 2>&1; then
    mkdir -p "$d/heatmaps"; cp "$s"/prediction_heatmap_*_HHA_*.png "$d/heatmaps/"
  fi
  echo "   -> $(du -sh "$d" | cut -f1)"
done

echo "=== results_final total: $(du -sh "$OUT" | cut -f1) ==="
find "$OUT" -type f | sed -E 's#.*/##; s/_[ABC]([._])/_HH\1/; s/[0-9]{3,}/N/g' | sort | uniq -c | sort -rn | head -20
