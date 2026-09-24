#!/bin/bash
#SBATCH --job-name=pretrain_slatepro
#SBATCH --output=logs/pretrain_slatepro_%A_%a.out
#SBATCH --error=logs/pretrain_slatepro_%A_%a.err
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2

# Pre-train SLaTe-PRO on each persona (Household A, B, C).
# Submit: sbatch scripts/pretrain_slatepro.sbatch
#
# Array index -> household: 0=A, 1=B, 2=C
# To run a single persona, e.g. B: sbatch --array=1 scripts/pretrain_slatepro.sbatch

set -e

# Resolve the repo root without hardcoding it: under sbatch the script is
# spooled, so SLURM_SUBMIT_DIR (submit from the repo root) is the reliable
# handle. REPO_ROOT can also be set explicitly.
PROJECT_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p checkpoints/slate_pro

HOUSEHOLDS=(A B C)
HH=${HOUSEHOLDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Pretraining SLaTe-PRO for Household $HH (array id $SLURM_ARRAY_TASK_ID) ==="
"${SLATEPRO_PYTHON:-python}" scripts/pretrain_slatepro.py \
  --household "$HH" \
  --output_dir checkpoints/slate_pro/ \
  --name default_100

echo "Done: Household $HH"
