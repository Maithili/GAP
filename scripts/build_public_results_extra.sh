#!/bin/bash
# Extend results_final/ with the "maybe relevant" cells (ablations, act+reset
# dist-shift, AB0) using the SAME slimming as build_public_results.sh:
#   results_<HH>.json (12KB totals) + weights_day*.pt + hybrid memory + llm audit + combined heatmaps.
# Cell map: outname | srcdir | method
set -uo pipefail
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"
PY="${COADAPT_PYTHON:-python}"
OUT=results_final

HV=results/hybrid_v2/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent
HVL=results/hybrid_v2/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent_humanllm_redirect_gate_pbg0.001
HVS=results/hybrid_v2/hybrid_retrain_warmup5_test0_every5_all1_noreset1_modeevent_gate_pbg0.001
AR_SLATE=full_results/slatepro_actreset_warmup65_test0_every99999_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days99999_noresetoff_vigoff
RT_SLATE=results/archives/slatepro_retrain_warmup5_test0_every5_look1_eval100000_multidefault_threshold0.0_suppressoff_generateoff_all1_train_every_n_days5_noreset1_vigoff

MAP="
ablation_onlyCurrent_scripted|${HVS}_noprep|hybrid
ablation_onlyCurrent_llm_human|${HVL}_noprep|hybrid
ablation_onlyFuture_scripted|${HVS}_nocurrent|hybrid
ablation_onlyFuture_llm_human|${HVL}_nocurrent|hybrid
ablation_singleGreedy_scripted|${HVS}_pfsingle|hybrid
ablation_singleGreedy_llm_human|${HVL}_pfsingle|hybrid
actreset_hybrid_scripted|results/hybrid_v2/hybrid_actreset_warmup65_test0_every99999_all1_noresetoff_modeevent_gate_pbg0.001|hybrid
actreset_streak_scripted|full_results/streak_noPrivilege_invban_cll2_warmup65_test0_every99999_look3_threshold0.0_addedBad100_actreset|streak
actreset_slatepro_scripted|$AR_SLATE|slatepro
actreset_plainllm_scripted|full_results/plainllm_actreset_warmup65_test0_every99999_all1_noresetoff|plainllm
actreset_oracle_scripted|full_results/oracle_actreset_warmup65_test0_every99999_all1_noresetoff|oracle
ab0_slatepro_scripted|$RT_SLATE|slatepro
ab0_slatepro_llm_human|${RT_SLATE}_humanllm_gpt-4o-mini_redirect|slatepro
ab0_streak_scripted|full_results/streak_noPrivilege_invban_cll2_warmup5_test0_every5_look3_threshold0.0_addedBad0|streak
"

while IFS='|' read -r model s method; do
  [ -z "$model" ] && continue
  if [ ! -d "$s" ]; then echo "!! MISSING srcdir for $model: $s"; continue; fi
  d="$OUT/$model"; mkdir -p "$d"
  echo "### $model (method=$method)  <- $s"
  for HH in A B C; do
    jf="$s/benchmark_${method}_${HH}.json"
    [ -f "$jf" ] || continue
    "$PY" - "$jf" "$d/results_${HH}.json" <<'PYEOF'
import sys, json; sys.path.insert(0,'scripts')
from benchmark_plot_io import load_benchmark_plot_data
pd = load_benchmark_plot_data(sys.argv[1])
json.dump({"totals": pd["totals"]}, open(sys.argv[2],"w"))
PYEOF
    ck="$s/model_checkpoints_benchmark_${method}_${HH}"
    if ls "$ck"/weights_day*.pt >/dev/null 2>&1; then
      mkdir -p "$d/checkpoints_${HH}"; cp "$ck"/weights_day*.pt "$d/checkpoints_${HH}/"
    fi
    if [ -d "$s/hybrid_memory_${HH}" ]; then
      mkdir -p "$d/memory_${HH}"; cp -r "$s/hybrid_memory_${HH}"/. "$d/memory_${HH}/"
    fi
    [ -f "$s/llm_audit_benchmark_${method}_${HH}.txt" ] && cp "$s/llm_audit_benchmark_${method}_${HH}.txt" "$d/llm_audit_${HH}.txt"
  done
  if ls "$s"/prediction_heatmap_*_HHA_*.png >/dev/null 2>&1; then
    mkdir -p "$d/heatmaps"; cp "$s"/prediction_heatmap_*_HHA_*.png "$d/heatmaps/"
  fi
  echo "   -> $(du -sh "$d" | cut -f1)"
done <<< "$MAP"
echo "=== results_final total: $(du -sh "$OUT" | cut -f1) ==="
