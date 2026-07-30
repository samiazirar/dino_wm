#!/bin/bash
#SBATCH --job-name=depth-validity
#SBATCH --partition=mlgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/depth-validity/depth-validity-20260730c/logs/job.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/depth-validity/depth-validity-20260730c/logs/job.%j.err
set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
ROOT=$PROJECT/outputs/depth-validity/depth-validity-20260730c
CODE_ROOT=$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/code/dino_wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
TOOL=$ROOT/code/measure_consumed_depth.py
OUT=$ROOT/measurements

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TORCH_HOME=$PROJECT/env/torch

/usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    --env TORCH_HOME="$TORCH_HOME" \
    --env PYTHONPATH="$CODE_ROOT:/opt/dinov2" \
    "$SIF" python "$TOOL" --output "$OUT" \
    --task pusht \
      "$PROJECT/outputs/campaign-seed1/pusht-seed1-countable-20260728a/dinocular/run/hydra.yaml" \
      "$PROJECT/outputs/campaign-seed1/pusht-seed1-countable-20260728a/cards/direct-pusht-dinocular-s1.yaml" \
      51e125b930bf7f7ed7906189f813849adc35a229ec41c75f27bfba03aa981d8c \
    --task wall \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/wall/dinocular/run/hydra.yaml" \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/cards/p3-wall-dinocular-s1.yaml" \
      4e3e6b2f1bdf05a1cf73f274befd0e1c6e79a644f245bb0c0b18927f3497ef86 \
    --task rope \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/rope/dinocular/run/hydra.yaml" \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/cards/p3-rope-dinocular-s1.yaml" \
      8f41b1cc8cf31ce0cdcbb457742869636f8b9880b7585ff90f2180194105359f \
    --task granular \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/granular/dinocular/run/hydra.yaml" \
      "$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/cards/p3-granular-dinocular-s1.yaml" \
      699c012d9e744890c7319a9c275a828d2f901dd187ad881a8c2094230be41faa
