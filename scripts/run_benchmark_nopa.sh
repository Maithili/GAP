#!/bin/bash
#SBATCH --job-name=bench_nopa
#SBATCH --partition=rail-lab
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_nopa_%j.out
#SBATCH --error=logs/bench_nopa_%j.err
#SBATCH --exclude=gundam,qt-1,clippy,chitti,conroy,nestor
#SBATCH --signal=B:USR1@120

# Closed-loop benchmark wrapper for the NOPA baseline (Puig et al., ICRA 2023).
# CPU-only, like run_benchmark_oracle.sh / run_benchmark_plainllm.sh: the goal
# proposal network is a ~279k-param MLP and everything else is simulator work,
# so NOPA never needs to occupy the GPU QOS cap. One job per household.
# Same env-var -> CLI plumbing, USR1 timeout auto-resubmit and .done_<HH> marker
# as run_benchmark_hybrid_packed.sh / run_benchmark_streak.sh.

set -euo pipefail

PYTHON="${COADAPT_PYTHON:-python}"
HOUSEHOLD=${HOUSEHOLD:-A}
TRAIN_DAYS=${TRAIN_DAYS:-65}
OUT_DIR=${OUT_DIR:-results/benchmark_nopa}
ALL_SCRIPTS=${ALL_SCRIPTS:-1}
WARMUP_DAYS=${WARMUP_DAYS:-65}
TEST_DAYS=${TEST_DAYS:-0}
TRAIN_EVERY_N_DAYS=${TRAIN_EVERY_N_DAYS:-99999}
NO_RESET_ENV=${NO_RESET_ENV:-}
MAX_TRAIN_EPOCHS=${MAX_TRAIN_EPOCHS:-400}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-1}
MAX_RESUBMITS=${MAX_RESUBMITS:-40}

# Human model
HUMAN_POLICY=${HUMAN_POLICY:-}
LLM_MODEL=${LLM_MODEL:-}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-}
LLM_MAX_USD_PER_DAY=${LLM_MAX_USD_PER_DAY:-}
LLM_OBSERVE_MODE=${LLM_OBSERVE_MODE:-}

# NOPA knobs (empty -> config/base.yaml default)
NOPA_SOLO_DAYS=${NOPA_SOLO_DAYS:-$WARMUP_DAYS}
NOPA_K=${NOPA_K:-20}
NOPA_TPROP=${NOPA_TPROP:-15}
NOPA_WM=${NOPA_WM:-5.0}
NOPA_MAX_ACTIONS=${NOPA_MAX_ACTIONS:-1}

export MAX_TRAIN_EPOCHS

# Ablation tag so each variant lands in its own dir (and its own .done marker).
ABL_TAG=""
[ -n "${NOPA_GPN_LOSS:-}" ] && [ "${NOPA_GPN_LOSS}" != "ce" ] && ABL_TAG="${ABL_TAG}_${NOPA_GPN_LOSS}"

if [ "$TRAIN_EVERY_N_DAYS" = "99999" ]; then SHAPE=actreset; else SHAPE=retrain; fi
OUT_DIR_NOPA="${OUT_DIR}/nopa_${SHAPE}_warmup${WARMUP_DAYS}_test${TEST_DAYS}_every${TRAIN_EVERY_N_DAYS}_K${NOPA_K}_tprop${NOPA_TPROP}_wm${NOPA_WM}_max${NOPA_MAX_ACTIONS}${ABL_TAG}${HUMAN_POLICY:+_human${HUMAN_POLICY}${LLM_OBSERVE_MODE:+_${LLM_OBSERVE_MODE}}}"
COMPLETION_MARKER="$OUT_DIR_NOPA/.done_${HOUSEHOLD}"

# ── Auto-resubmit on SLURM timeout ──────────────────────────────────────
resubmit_self() {
    echo "Caught timeout signal (USR1). Attempt $RESUBMIT_COUNT/$MAX_RESUBMITS."
    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "ERROR: Max resubmits ($MAX_RESUBMITS) reached. Giving up."
        exit 1
    fi
    echo "Resubmitting..."
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) \
    OUT_DIR="$OUT_DIR" HOUSEHOLD="$HOUSEHOLD" TRAIN_DAYS="$TRAIN_DAYS" \
    ALL_SCRIPTS="$ALL_SCRIPTS" WARMUP_DAYS="$WARMUP_DAYS" TEST_DAYS="$TEST_DAYS" \
    TRAIN_EVERY_N_DAYS="$TRAIN_EVERY_N_DAYS" NO_RESET_ENV="$NO_RESET_ENV" \
    MAX_TRAIN_EPOCHS="$MAX_TRAIN_EPOCHS" MAX_RESUBMITS="$MAX_RESUBMITS" \
    HUMAN_POLICY="$HUMAN_POLICY" LLM_MODEL="$LLM_MODEL" \
    LLM_CACHE_DIR="$LLM_CACHE_DIR" LLM_MAX_USD_PER_DAY="$LLM_MAX_USD_PER_DAY" \
    LLM_OBSERVE_MODE="$LLM_OBSERVE_MODE" \
    NOPA_SOLO_DAYS="$NOPA_SOLO_DAYS" NOPA_K="$NOPA_K" NOPA_TPROP="$NOPA_TPROP" \
    NOPA_WM="$NOPA_WM" NOPA_MAX_ACTIONS="$NOPA_MAX_ACTIONS" \
    sbatch scripts/run_benchmark_nopa.sh
    exit 0
}
trap 'resubmit_self' USR1

echo "=== NOPA Benchmark (attempt $RESUBMIT_COUNT) ==="
echo "Household:           $HOUSEHOLD"
echo "Output:              $OUT_DIR_NOPA"
echo "Shape:               $SHAPE (warmup=$WARMUP_DAYS every=$TRAIN_EVERY_N_DAYS test=$TEST_DAYS)"
echo "GPN solo days:       $NOPA_SOLO_DAYS"
echo "K / T_prop / w_m:    $NOPA_K / $NOPA_TPROP / $NOPA_WM"
echo "Max actions/tick:    $NOPA_MAX_ACTIONS"
echo "Human policy:        ${HUMAN_POLICY:-scripted}"
echo "Ablation tag:        ${ABL_TAG:-<none>}"

if [ -f "$COMPLETION_MARKER" ]; then
  echo "Completion marker found: $COMPLETION_MARKER - skipping."
  exit 0
fi

mkdir -p logs "$OUT_DIR_NOPA"
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

# Run in background so the USR1 trap can fire during execution
$PYTHON scripts/benchmark_comparison.py \
    --household "$HOUSEHOLD" \
    --train_days "$TRAIN_DAYS" \
    --out_dir "$OUT_DIR_NOPA" \
    --run_nopa \
    --skip_slatepro \
    --skip_noop \
    --time_step_minutes 10 \
    ${ALL_SCRIPTS:+--all_scripts} \
    --warmup_days "$WARMUP_DAYS" \
    --test_days "$TEST_DAYS" \
    --train_every_n_days "$TRAIN_EVERY_N_DAYS" \
    ${NO_RESET_ENV:+--no_reset_env} \
    ${HUMAN_POLICY:+--human_policy "$HUMAN_POLICY"} \
    ${LLM_MODEL:+--llm_model "$LLM_MODEL"} \
    ${LLM_CACHE_DIR:+--llm_cache_dir "$LLM_CACHE_DIR"} \
    ${LLM_MAX_USD_PER_DAY:+--llm_max_usd_per_day "$LLM_MAX_USD_PER_DAY"} \
    ${LLM_OBSERVE_MODE:+--llm_observe_mode "$LLM_OBSERVE_MODE"} \
    --nopa_solo_days "$NOPA_SOLO_DAYS" \
    --nopa_k "$NOPA_K" \
    --nopa_tprop "$NOPA_TPROP" \
    --nopa_wm "$NOPA_WM" \
    --nopa_max_actions "$NOPA_MAX_ACTIONS" \
    ${NOPA_GPN_LOSS:+--nopa_gpn_loss "$NOPA_GPN_LOSS"} &
PY_PID=$!
set +e
wait $PY_PID
PY_EXIT=$?
set -e
if [ $PY_EXIT -ne 0 ]; then
    echo "Job failed (exit=$PY_EXIT) on node: $SLURMD_NODENAME"
    echo "Variables: HOUSEHOLD=$HOUSEHOLD OUT_DIR_NOPA=$OUT_DIR_NOPA"
    exit 1
fi

touch "$COMPLETION_MARKER"
echo "Completion marker: $COMPLETION_MARKER"
echo "Done. Results: $OUT_DIR_NOPA/benchmark_nopa_${HOUSEHOLD}.json"
