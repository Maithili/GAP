#!/bin/bash
#SBATCH --job-name=bench_plainllm
#SBATCH --partition=rail-lab
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_plainllm_%j.out
#SBATCH --error=logs/bench_plainllm_%j.err
#SBATCH --exclude=gundam,qt-1,clippy,chitti,conroy,nestor
#SBATCH --signal=B:USR1@120

# Closed-loop benchmark wrapper for the plain-LLM baseline robot policy.
# Mirrors run_benchmark_hybrid_packed.sh (USR1 auto-resubmit, .done_<HH> marker).

set -euo pipefail

PYTHON="${COADAPT_PYTHON:-python}"
HOUSEHOLD=${HOUSEHOLD:-A}
TRAIN_DAYS=${TRAIN_DAYS:-65}
OUT_DIR=${OUT_DIR:-results/benchmark_plainllm}
ALL_SCRIPTS=${ALL_SCRIPTS:-}
WARMUP_DAYS=${WARMUP_DAYS:-65}
TEST_DAYS=${TEST_DAYS:-0}
TRAIN_EVERY_N_DAYS=${TRAIN_EVERY_N_DAYS:-99999}
NO_RESET_ENV=${NO_RESET_ENV:-}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-30}
HUMAN_POLICY=${HUMAN_POLICY:-}
LLM_MODEL=${LLM_MODEL:-}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-}
LLM_MAX_USD_PER_DAY=${LLM_MAX_USD_PER_DAY:-}
LLM_OBSERVE_MODE=${LLM_OBSERVE_MODE:-}
# redirect is the ONLY LLM human we report: default any LLM-human run to it so a run can
# never silently use the superseded undo_allow mode (scripted runs stay untagged).
if [ -n "$HUMAN_POLICY" ] && [ -z "$LLM_OBSERVE_MODE" ]; then LLM_OBSERVE_MODE=redirect; fi
PLAINLLM_CACHE_DIR=${PLAINLLM_CACHE_DIR:-cache/llm_plainllm/${SLURM_JOB_NAME:-plainllm}_${HOUSEHOLD}}

OUT_DIR_PL=$OUT_DIR"_warmup"$WARMUP_DAYS"_test"$TEST_DAYS"_every"$TRAIN_EVERY_N_DAYS"_all"${ALL_SCRIPTS:-off}"_noreset"${NO_RESET_ENV:-off}${HUMAN_POLICY:+_human$HUMAN_POLICY}${LLM_OBSERVE_MODE:+_$LLM_OBSERVE_MODE}

COMPLETION_MARKER="$OUT_DIR_PL/.done_${HOUSEHOLD}"

resubmit_self() {
    echo "Caught timeout signal (USR1). Attempt $RESUBMIT_COUNT/$MAX_RESUBMITS."
    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "ERROR: Max resubmits reached."; exit 1
    fi
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) \
    OUT_DIR="$OUT_DIR" HOUSEHOLD="$HOUSEHOLD" TRAIN_DAYS="$TRAIN_DAYS" \
    ALL_SCRIPTS="$ALL_SCRIPTS" \
    WARMUP_DAYS="$WARMUP_DAYS" TEST_DAYS="$TEST_DAYS" \
    TRAIN_EVERY_N_DAYS="$TRAIN_EVERY_N_DAYS" NO_RESET_ENV="$NO_RESET_ENV" \
    MAX_RESUBMITS="$MAX_RESUBMITS" \
    HUMAN_POLICY="$HUMAN_POLICY" LLM_MODEL="$LLM_MODEL" \
    LLM_CACHE_DIR="$LLM_CACHE_DIR" LLM_MAX_USD_PER_DAY="$LLM_MAX_USD_PER_DAY" \
    LLM_OBSERVE_MODE="$LLM_OBSERVE_MODE" \
    PLAINLLM_CACHE_DIR="$PLAINLLM_CACHE_DIR" \
    sbatch scripts/run_benchmark_plainllm.sh
    exit 0
}
trap 'resubmit_self' USR1

echo "=== Plain-LLM Benchmark (attempt $RESUBMIT_COUNT) ==="
echo "Household: $HOUSEHOLD   Output: $OUT_DIR_PL"
echo "$SLURMD_NODENAME"

if [ -f "$COMPLETION_MARKER" ]; then
    echo "Completion marker found — skipping."; exit 0
fi

mkdir -p logs "$OUT_DIR_PL"
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

$PYTHON scripts/benchmark_comparison.py \
    --household "$HOUSEHOLD" \
    --train_days "$TRAIN_DAYS" \
    --out_dir "$OUT_DIR_PL" \
    --run_plainllm \
    --skip_slatepro --skip_noop \
    --time_step_minutes 10 \
    ${ALL_SCRIPTS:+--all_scripts} \
    ${WARMUP_DAYS:+--warmup_days "$WARMUP_DAYS"} \
    ${TEST_DAYS:+--test_days "$TEST_DAYS"} \
    ${TRAIN_EVERY_N_DAYS:+--train_every_n_days "$TRAIN_EVERY_N_DAYS"} \
    ${NO_RESET_ENV:+--no_reset_env} \
    ${HUMAN_POLICY:+--human_policy "$HUMAN_POLICY"} \
    ${LLM_MODEL:+--llm_model "$LLM_MODEL"} \
    ${LLM_CACHE_DIR:+--llm_cache_dir "$LLM_CACHE_DIR"} \
    ${LLM_MAX_USD_PER_DAY:+--llm_max_usd_per_day "$LLM_MAX_USD_PER_DAY"} \
    ${LLM_OBSERVE_MODE:+--llm_observe_mode "$LLM_OBSERVE_MODE"} \
    ${PLAINLLM_CACHE_DIR:+--plainllm_cache_dir "$PLAINLLM_CACHE_DIR"} &
PY_PID=$!
set +e
wait $PY_PID
PY_EXIT=$?
set -e
if [ $PY_EXIT -ne 0 ]; then
    echo "Job failed (exit=$PY_EXIT) on $SLURMD_NODENAME"; exit 1
fi

touch "$COMPLETION_MARKER"
echo "Completion marker: $COMPLETION_MARKER"
echo "Done."
