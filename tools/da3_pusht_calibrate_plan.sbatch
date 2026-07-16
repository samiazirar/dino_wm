#!/bin/bash
#SBATCH --job-name=dinocular-depth-da3-pusht-calibrate-plan
#SBATCH --partition=sgpu_short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_calibrate_plan.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_calibrate_plan.%j.err

set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
CALIBRATION=$PROJECT/data/depth_cache_calibration/pusht.json
PLAN=$PROJECT/data/depth_cache_da3_pusht_shards/plan.json
test ! -e "$CALIBRATION"
test ! -e "$PLAN"

CACHE_ROOT=$PROJECT/cache/da3-pusht-calibrate/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
test -w "$TRITON_CACHE_DIR" -a -w "$XDG_CACHE_HOME" -a -w "$HF_HOME"

APPTAINER=(/usr/bin/apptainer exec --nv --writable-tmpfs
    --bind "$PROJECT:/workspace"
    --env NUMBA_CACHE_DIR=/tmp/dinocular-numba-$SLURM_JOB_ID
    --env TRITON_CACHE_DIR=/workspace/cache/da3-pusht-calibrate/$SLURM_JOB_ID/triton
    --env XDG_CACHE_HOME=/workspace/cache/da3-pusht-calibrate/$SLURM_JOB_ID/xdg
    --env HF_HOME=/workspace/cache/da3-pusht-calibrate/$SLURM_JOB_ID/hf
    --env TORCH_HOME=/workspace/env/torch
    --env PYTHONPATH=/workspace/env/da3-runtime-site:/workspace/code/dino_wm:/workspace/code/depth-anything-3/src:/workspace/code/depth-anything-3/da3_streaming
    "$SIF")

"${APPTAINER[@]}" python /workspace/code/dino_wm/tools/precompute_depth_da3_shards.py calibrate \
    --root /workspace/data/raw \
    --model-dir /workspace/models/da3nested_giant_large_1_1 \
    --da3-root /workspace/code/depth-anything-3 \
    --json /workspace/data/depth_cache_calibration/pusht.json

"${APPTAINER[@]}" python /workspace/code/dino_wm/tools/precompute_depth_da3_shards.py plan \
    --root /workspace/data/raw \
    --calibration-manifest /workspace/data/depth_cache_calibration/pusht.json \
    --json /workspace/data/depth_cache_da3_pusht_shards/plan.json
