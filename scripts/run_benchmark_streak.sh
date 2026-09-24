#!/bin/bash
#SBATCH --job-name=bench_streak
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/bench_streak_%j.out
#SBATCH --error=logs/bench_streak_%j.err
#SBATCH --exclude=gundam,qt-1,clippy,chitti,conroy,nestor
#SBATCH --signal=B:USR1@120

set -euo pipefail

PYTHON="${COADAPT_PYTHON:-python}"
HOUSEHOLD=${HOUSEHOLD:-A}
TRAIN_DAYS=${TRAIN_DAYS:-65}
OUT_DIR=${OUT_DIR:-results/benchmark_streak}
TIME_STEP=${TIME_STEP:-10}
WARMUP_DAYS=${WARMUP_DAYS:-5}
TEST_DAYS=${TEST_DAYS:-0}
TRAIN_EVERY_N_DAYS=${TRAIN_EVERY_N_DAYS:-5}
NO_RESET_ENV=${NO_RESET_ENV:-1}
ACTRESET=${ACTRESET:-}          # non-empty => act+reset (env RESETS each day; overrides NO_RESET_ENV); tags dir _actreset
ADDED_BAD_PERC=${ADDED_BAD_PERC:-100}
MAX_TRAIN_EPOCHS=${MAX_TRAIN_EPOCHS:-500}
HUMAN_POLICY=${HUMAN_POLICY:-}
CL_MODE=${CL_MODE:-none}            # none | l2 | ewc (continual learning)
LOOKAHEAD=${LOOKAHEAD:-3}          # STREAK lookahead_steps (1 for 1-step paper parity)

export ADDED_BAD_PERC
export MAX_TRAIN_EPOCHS

# CL tag only when CL is on, so base STREAK dir names stay unchanged.
CL_TAG=""
if [ "$CL_MODE" != "none" ]; then CL_TAG="_cl${CL_MODE}"; fi
CONF_THRESH=${CONF_THRESH:-0.0}    # STREAK confidence_threshold (per-prediction emit cutoff)
MAX_ACTIONS=${MAX_ACTIONS:-}       # STREAK per-tick top-N action cap (empty = uncapped paper default); tags out dir *_max<N>
unset STREAK_SEED   # reported STREAK runs are unseeded (see DESIGN.md: seed-driven multimodality)
OUT_DIR_STREAK="${OUT_DIR}${CL_TAG}_warmup${WARMUP_DAYS}_test${TEST_DAYS}_every${TRAIN_EVERY_N_DAYS}_look${LOOKAHEAD}_threshold${CONF_THRESH}_addedBad${ADDED_BAD_PERC}${ACTRESET:+_actreset}${MAX_ACTIONS:+_max${MAX_ACTIONS}}${HUMAN_POLICY:+_human${HUMAN_POLICY}}"
COMPLETION_MARKER="$OUT_DIR_STREAK/.done_${HOUSEHOLD}"

echo "=== STREAK benchmark (train-from-scratch) HH=${HOUSEHOLD} OUT=${OUT_DIR_STREAK} ADDED_BAD=${ADDED_BAD_PERC} ==="

if [ -f "$COMPLETION_MARKER" ]; then
  echo "Completion marker found: $COMPLETION_MARKER — skipping."
  exit 0
fi

mkdir -p logs "$OUT_DIR_STREAK"
# Resolve the repo root without hardcoding it. Under sbatch the script is
# spooled, so $BASH_SOURCE does not point into the repo -- SLURM_SUBMIT_DIR
# does (submit from the repo root). REPO_ROOT can also be set explicitly.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

$PYTHON scripts/benchmark_comparison.py \
    --household "$HOUSEHOLD" \
    --train_days "$TRAIN_DAYS" \
    --out_dir "$OUT_DIR_STREAK" \
    --run_streak \
    --skip_slatepro \
    --skip_noop \
    --all_scripts \
    --time_step_minutes "$TIME_STEP" \
    --warmup_days "$WARMUP_DAYS" \
    --test_days "$TEST_DAYS" \
    --train_every_n_days "$TRAIN_EVERY_N_DAYS" \
    $([ -z "$ACTRESET" ] && echo "${NO_RESET_ENV:+--no_reset_env}") \
    --streak_lookahead_steps "$LOOKAHEAD" \
    --threshold "$CONF_THRESH" \
    ${MAX_ACTIONS:+--streak_max_actions "$MAX_ACTIONS"} \
    --streak_cl_mode "$CL_MODE" \
    ${HUMAN_POLICY:+--human_policy "$HUMAN_POLICY"} \
    ${HUMAN_POLICY:+--llm_cache_dir "${LLM_CACHE_DIR:-cache/llm_human/streak_${CL_MODE}_${HOUSEHOLD}}"}

touch "$COMPLETION_MARKER"
echo "Completion marker: $COMPLETION_MARKER"
echo "Done. Results: $OUT_DIR_STREAK/benchmark_streak_${HOUSEHOLD}.json"
