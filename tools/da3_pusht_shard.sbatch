#!/bin/bash
#SBATCH --job-name=dinocular-depth-da3-pusht-shard
#SBATCH --partition=sgpu_medium
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_shard.%A.%a.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_shard.%A.%a.err

set -euo pipefail

: "${DA3_SHARD_INDEX:?DA3_SHARD_INDEX=0..7 is required}"
: "${DA3_PLAN_FILE_SHA256:?DA3_PLAN_FILE_SHA256 is required}"
case "$DA3_SHARD_INDEX" in 0|1|2|3|4|5|6|7) ;; *) exit 2 ;; esac
case "$DA3_PLAN_FILE_SHA256" in [0-9a-f][0-9a-f]*) ;; *) exit 2 ;; esac
test "${#DA3_PLAN_FILE_SHA256}" -eq 64

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
PLAN=$PROJECT/data/depth_cache_da3_pusht_shards/plan.json
SHARD_ROOT=$(printf '%s/data/depth_cache_da3_pusht_shards/shard-%03d-of-008' "$PROJECT" "$DA3_SHARD_INDEX")
test -f "$PLAN"
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$DA3_PLAN_FILE_SHA256"
test ! -e "$SHARD_ROOT"

CACHE_ROOT=$PROJECT/cache/da3-pusht-shards/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
test -w "$TRITON_CACHE_DIR" -a -w "$XDG_CACHE_HOME" -a -w "$HF_HOME"

/usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:/workspace" \
    --env NUMBA_CACHE_DIR=/tmp/dinocular-numba-$SLURM_JOB_ID \
    --env TRITON_CACHE_DIR=/workspace/cache/da3-pusht-shards/$SLURM_JOB_ID/triton \
    --env XDG_CACHE_HOME=/workspace/cache/da3-pusht-shards/$SLURM_JOB_ID/xdg \
    --env HF_HOME=/workspace/cache/da3-pusht-shards/$SLURM_JOB_ID/hf \
    --env TORCH_HOME=/workspace/env/torch \
    --env PYTHONPATH=/workspace/env/da3-runtime-site:/workspace/code/dino_wm:/workspace/code/depth-anything-3/src:/workspace/code/depth-anything-3/da3_streaming \
    "$SIF" \
    python /workspace/code/dino_wm/tools/precompute_depth_da3_shards.py build \
        --root /workspace/data/raw \
        --shards-root /workspace/data/depth_cache_da3_pusht_shards \
        --plan /workspace/data/depth_cache_da3_pusht_shards/plan.json \
        --shard-index "$DA3_SHARD_INDEX" \
        --model-dir /workspace/models/da3nested_giant_large_1_1 \
        --da3-root /workspace/code/depth-anything-3
