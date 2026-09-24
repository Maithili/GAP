#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --exclude=gundam,qt-1,clippy,chitti,conroy,nestor
#SBATCH --signal=B:USR1@120

set -euo pipefail

PYTHON="${COADAPT_PYTHON:-python}"
HOUSEHOLD=${HOUSEHOLD:-A}
TRAIN_DAYS=${TRAIN_DAYS:-65}
SLATEPRO_CKPT=${SLATEPRO_CKPT-checkpoints/slate-pro}   # empty (set but blank) => train from scratch
OUT_DIR=${OUT_DIR:-results/benchmark}
LOOKAHEAD_MODEL=${LOOKAHEAD_MODEL:-}
LOOKAHEAD_EVAL=${LOOKAHEAD_EVAL:-}
THRESHOLD=${THRESHOLD:-}
ALL_SCRIPTS=${ALL_SCRIPTS:-}
WARMUP_DAYS=${WARMUP_DAYS:-5}
TEST_DAYS=${TEST_DAYS:-10}
TRAIN_EVERY_N_DAYS=${TRAIN_EVERY_N_DAYS:-1}
NO_RESET_ENV=${NO_RESET_ENV:-}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-30}
ADDED_BAD_PERC=${ADDED_BAD_PERC:-}
HUMAN_POLICY=${HUMAN_POLICY:-}
LLM_MODEL=${LLM_MODEL:-}
LLM_CACHE_DIR=${LLM_CACHE_DIR:-}
LLM_MAX_USD_PER_DAY=${LLM_MAX_USD_PER_DAY:-}
LLM_OBSERVE_MODE=${LLM_OBSERVE_MODE:-}
# redirect is the ONLY LLM human we report: default any LLM-human run to it so a run can
# never silently use the superseded undo_allow mode (scripted runs stay untagged).
if [ -n "$HUMAN_POLICY" ] && [ -z "$LLM_OBSERVE_MODE" ]; then LLM_OBSERVE_MODE=redirect; fi

OUT_DIR_SLATEPRO=$OUT_DIR"_warmup"$WARMUP_DAYS"_test"$TEST_DAYS"_every"$TRAIN_EVERY_N_DAYS"_look"${LOOKAHEAD_MODEL:-default}"_eval"${LOOKAHEAD_EVAL:-default}"_multi"default"_threshold"${THRESHOLD:-default}"_suppressoff""_generate"${GENERATE_DATASET:-off}"_all"${ALL_SCRIPTS:-off}"_train_every_n_days"$TRAIN_EVERY_N_DAYS"_noreset"${NO_RESET_ENV:-off}"_vigoff"${ADDED_BAD_PERC:+_addedBad$ADDED_BAD_PERC}${HUMAN_POLICY:+_human$HUMAN_POLICY}${LLM_MODEL:+_$LLM_MODEL}${LLM_OBSERVE_MODE:+_$LLM_OBSERVE_MODE}

COMPLETION_MARKER="$OUT_DIR_SLATEPRO/.done_${HOUSEHOLD}"

# ── Auto-resubmit on SLURM timeout ──────────────────────────────────────
resubmit_self() {
    echo "Caught timeout signal (USR1). Attempt $RESUBMIT_COUNT/$MAX_RESUBMITS."
    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "ERROR: Max resubmits ($MAX_RESUBMITS) reached. Giving up."
        exit 1
    fi
    echo "Resubmitting..."
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) \
    SLATEPRO_CKPT="$SLATEPRO_CKPT" \
    OUT_DIR="$OUT_DIR" \
    HOUSEHOLD="$HOUSEHOLD" \
    TRAIN_DAYS="$TRAIN_DAYS" \
    LOOKAHEAD_MODEL="$LOOKAHEAD_MODEL" \
    LOOKAHEAD_EVAL="$LOOKAHEAD_EVAL" \
    THRESHOLD="$THRESHOLD" \
    ALL_SCRIPTS="$ALL_SCRIPTS" \
    WARMUP_DAYS="$WARMUP_DAYS" \
    TEST_DAYS="$TEST_DAYS" \
    TRAIN_EVERY_N_DAYS="$TRAIN_EVERY_N_DAYS" \
    NO_RESET_ENV="$NO_RESET_ENV" \
    MAX_RESUBMITS="$MAX_RESUBMITS" \
    ADDED_BAD_PERC="$ADDED_BAD_PERC" \
    HUMAN_POLICY="$HUMAN_POLICY" \
    LLM_MODEL="$LLM_MODEL" \
    LLM_CACHE_DIR="$LLM_CACHE_DIR" \
    LLM_MAX_USD_PER_DAY="$LLM_MAX_USD_PER_DAY" \
    LLM_OBSERVE_MODE="$LLM_OBSERVE_MODE" \
    sbatch scripts/run_benchmark_slatepro.sh
    exit 0
}
trap 'resubmit_self' USR1

echo "=== Benchmark Comparison (attempt $RESUBMIT_COUNT) ==="
echo "Household:                $HOUSEHOLD"
echo "Train days:               $TRAIN_DAYS"
echo "Output:                   $OUT_DIR_SLATEPRO"
echo "Lookahead model:          ${LOOKAHEAD_MODEL:-default}"
echo "Lookahead eval:           ${LOOKAHEAD_EVAL:-default}"
echo "All scripts:              ${ALL_SCRIPTS:-off}"
echo "Warmup days:              ${WARMUP_DAYS:-5}"
echo "Test days:                ${TEST_DAYS:-10}"
echo "Train every n days:       ${TRAIN_EVERY_N_DAYS:-1}"
echo "No reset env:             ${NO_RESET_ENV:-off}"
echo "Human policy:             ${HUMAN_POLICY:-scripted}"
echo "LLM model:                ${LLM_MODEL:-default}"
echo "LLM cache dir:            ${LLM_CACHE_DIR:-default}"
echo "Added bad %:              ${ADDED_BAD_PERC:-0}"
echo "Resubmit count:           $RESUBMIT_COUNT"
echo $SLURMD_NODENAME
echo ""

# Skip if already completed (bypassed when re-running with prior checkpoints)
if [ -f "$COMPLETION_MARKER" ]; then
    echo "Completion marker found: $COMPLETION_MARKER — skipping."
    exit 0
fi

mkdir -p logs "$OUT_DIR_SLATEPRO"

# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"


# Run benchmark in background so trap can fire during execution
$PYTHON scripts/benchmark_comparison.py \
    ${SLATEPRO_CKPT:+--slatepro_checkpoint "$SLATEPRO_CKPT"} \
    --household "$HOUSEHOLD" \
    --train_days "$TRAIN_DAYS" \
    --out_dir "$OUT_DIR_SLATEPRO" \
    --skip_noop \
    ${LOOKAHEAD_MODEL:+--lookahead_steps "$LOOKAHEAD_MODEL"} \
    ${LOOKAHEAD_EVAL:+--lookahead_eval "$LOOKAHEAD_EVAL"} \
    ${THRESHOLD:+--threshold "$THRESHOLD"} \
    ${ALL_SCRIPTS:+--all_scripts} \
    ${WARMUP_DAYS:+--warmup_days "$WARMUP_DAYS"} \
    ${TEST_DAYS:+--test_days "$TEST_DAYS"} \
    ${TRAIN_EVERY_N_DAYS:+--train_every_n_days "$TRAIN_EVERY_N_DAYS"} \
    ${NO_RESET_ENV:+--no_reset_env} \
    ${HUMAN_POLICY:+--human_policy "$HUMAN_POLICY"} \
    ${LLM_MODEL:+--llm_model "$LLM_MODEL"} \
    ${LLM_CACHE_DIR:+--llm_cache_dir "$LLM_CACHE_DIR"} \
    ${LLM_MAX_USD_PER_DAY:+--llm_max_usd_per_day "$LLM_MAX_USD_PER_DAY"} \
    ${LLM_OBSERVE_MODE:+--llm_observe_mode "$LLM_OBSERVE_MODE"} &
PY_PID=$!
# wait is interruptible by signals, so USR1 trap can fire
# Disable set -e around wait so we can check exit code manually
set +e
wait $PY_PID
PY_EXIT=$?
set -e
if [ $PY_EXIT -ne 0 ]; then
    echo "Job failed (exit=$PY_EXIT) on node: $SLURMD_NODENAME"
    echo Variables: HOUSEHOLD=$HOUSEHOLD TRAIN_DAYS=$TRAIN_DAYS OUT_DIR_SLATEPRO=$OUT_DIR_SLATEPRO
    exit 1
fi

# Generate plots
$PYTHON scripts/visualize_comparison.py \
    --results "$OUT_DIR_SLATEPRO/benchmark_${HOUSEHOLD}.json" \
    --out_dir "$OUT_DIR_SLATEPRO/plots/household${HOUSEHOLD}"

# Mark as complete
touch "$COMPLETION_MARKER"
echo "Completion marker: $COMPLETION_MARKER"

echo ""
echo "Done. Plots at: $OUT_DIR_SLATEPRO/plots/household${HOUSEHOLD}/"
