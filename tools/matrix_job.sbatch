#!/bin/bash
#SBATCH --job-name=dinocular-matrix
#SBATCH --partition=sgpu_short
#SBATCH --time=07:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/matrix.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/matrix.%x.%j.err

set -euo pipefail

: "${RUN_CARD:?RUN_CARD is required}"
: "${RUN_CARD_FILE_SHA256:?RUN_CARD_FILE_SHA256 is required}"
: "${RUN_CARD_SHA256:?RUN_CARD_SHA256 is required}"

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
RUNNER=$PROJECT/code/dino_wm/tools/run_matrix_card.py
python3 "$RUNNER" verify --run-card "$RUN_CARD"
SIF=$(python3 "$RUNNER" emit --run-card "$RUN_CARD" --field container)
ARM=$(python3 "$RUNNER" emit --run-card "$RUN_CARD" --field arm)
CACHE_ROOT=$PROJECT/cache/matrix/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export TORCH_HOME=$PROJECT/env/torch
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
test -w "$TRITON_CACHE_DIR" -a -w "$XDG_CACHE_HOME" -a -w "$HF_HOME"
source "$PROJECT/code/dino_wm/tools/dinocular_container_env.sh"
build_dinocular_container_env "$ARM"

/usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env RUN_CARD_FILE_SHA256="$RUN_CARD_FILE_SHA256" \
    --env RUN_CARD_SHA256="$RUN_CARD_SHA256" \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env TORCH_HOME="$TORCH_HOME" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    "${DINOCULAR_CONTAINER_ENV[@]}" \
    --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
    "$SIF" \
    python "$RUNNER" execute --run-card "$RUN_CARD"
