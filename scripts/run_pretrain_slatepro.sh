#!/bin/bash
#SBATCH --job-name=pretrain_sp
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/pretrain_slatepro_%j.out
#SBATCH --error=logs/pretrain_slatepro_%j.err

# Step 0: Batch-train SLaTe-PRO on HOMER+ data (one household per job).
#
# Usage:
#   sbatch scripts/run_pretrain_slatepro.sh A
#   sbatch scripts/run_pretrain_slatepro.sh B
#   sbatch scripts/run_pretrain_slatepro.sh C
#
# Resumable: if killed by SLURM timeout, resubmit the same command and
# training resumes from the last checkpoint automatically.
#
# NOTE: SLaTe-PRO requires the pyml conda env, not coadapt.

set -euo pipefail

HH=${1:?"Usage: sbatch scripts/run_pretrain_slatepro.sh <A|B|C>"}

case "$HH" in
    A|B|C) ;;
    *) echo "Error: household must be A, B, or C (got: $HH)" >&2; exit 1 ;;
esac

# SLaTe-PRO needs pyml env (different deps from coadapt)
PYTHON="${SLATEPRO_PYTHON:-python}"
# Same resolution as the benchmark wrappers: under sbatch the script is spooled,
# so SLURM_SUBMIT_DIR (submit from the repo root) is the reliable handle.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO"

OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/slate_pro}
NAME=${NAME:-default_100}
TRAIN_DAYS=${TRAIN_DAYS:-}  # empty = use all available data

mkdir -p logs "$OUTPUT_DIR"

echo "=== SLaTe-PRO: Household $HH ==="

EXTRA_ARGS=""
if [ -n "$TRAIN_DAYS" ]; then
    EXTRA_ARGS="--train_days $TRAIN_DAYS"
fi

CUDA_LAUNCH_BLOCKING=1 $PYTHON scripts/pretrain_slatepro.py \
    --household "$HH" \
    --output_dir "$OUTPUT_DIR" \
    --name "$NAME" \
    $EXTRA_ARGS

echo "Done: Household $HH. Checkpoint at: $OUTPUT_DIR/"
