#!/bin/bash
#SBATCH --job-name=p3-resume-det
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/resume_determinism.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/resume_determinism.%j.err

set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RESULT_DIR=${RESUME_PROOF_DIR:-$PROJECT/results/resume/determinism-$SLURM_JOB_ID}
CACHE_ROOT=$PROJECT/cache/resume-proof/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

nvidia-smi
sha256sum "$SIF"

/usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env TORCH_HOME="$PROJECT/env/torch" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$PROJECT/code/dinov2" \
    --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
    --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
    "$SIF" \
    python "$PROJECT/code/dino_wm/tools/resume_determinism_proof.py" \
        --output-dir "$RESULT_DIR" \
        --first-steps 3 \
        --resume-steps 2 \
        --batch-size 4

printf 'RESUME_DETERMINISM=PASS result=%s/result.json\n' "$RESULT_DIR"
