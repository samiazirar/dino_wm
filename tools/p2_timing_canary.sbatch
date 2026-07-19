#!/bin/bash
#SBATCH --job-name=p2-dinov2
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p2_dinov2.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p2_dinov2.%x.%j.err

set -euo pipefail

: "${P2_ENV:?set P2_ENV to pusht, wall, rope, or granular}"
PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RESULT_DIR=$PROJECT/results/p2/dinov2_vits14/$P2_ENV
CACHE_DIR=$PROJECT/cache/p2-$SLURM_JOB_ID
mkdir -p "$RESULT_DIR" "$CACHE_DIR/triton" "$CACHE_DIR/xdg" "$CACHE_DIR/hf"

export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export HF_HOME=$CACHE_DIR/hf
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

nvidia-smi
sha256sum "$SIF"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$PROJECT/code/dinov2" \
    --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
    --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
    "$SIF" \
    python "$PROJECT/code/dino_wm/tools/p2_timing_canary.py" \
        --env "$P2_ENV" \
        --warmup-steps 20 \
        --measured-steps 200 \
        --output "$RESULT_DIR/timing.$SLURM_JOB_ID.json"

printf 'P2_DINOV2_TIMING=PASS env=%s job=%s\n' "$P2_ENV" "$SLURM_JOB_ID"
