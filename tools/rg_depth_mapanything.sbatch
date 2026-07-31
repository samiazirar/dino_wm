#!/bin/bash
#SBATCH --job-name=rg-depth-mapanything
#SBATCH --partition=sgpu_devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/rg-depth-repair/rg-depth-repair-20260730a/logs/mapanything.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/rg-depth-repair/rg-depth-repair-20260730a/logs/mapanything.%j.err
set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
REPAIR_ROOT=$PROJECT/outputs/rg-depth-repair/rg-depth-repair-20260730a
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
MODEL_REVISION=a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa
CODE_ROOT=${RG_CODE_ROOT:-$REPAIR_ROOT/code}
ENVIRONMENT=${RG_ENVIRONMENT:?RG_ENVIRONMENT is required}
STAGE=${RG_STAGE:-build}
OUTPUT_DIR=${RG_OUTPUT_DIR:?RG_OUTPUT_DIR is required}
SHARDS_ROOT=${RG_SHARDS_ROOT:-}
SHARD_COUNT=${RG_SHARD_COUNT:-}
SHARD_INDEX=${RG_SHARD_INDEX:-}
CALIBRATION_JSON=${RG_CALIBRATION_JSON:-}
MAX_TRAJECTORIES=${RG_MAX_TRAJECTORIES:-}
SPOT_FRAMES=${RG_SPOT_FRAMES:-32}
# Production builds are pinned to the literal-singleton producer (batch size 1).
# `validate` instantiates the live producer at batch size 1 and requires the
# cached manifest to carry that exact identity, so a cache built at any other
# batch size records `candidate_independent_singleton_scenes_on_batch_axis` and
# can never be admitted.  Default `build` to 1 and refuse anything else; the
# batch knob remains meaningful only for the validator's independent-batch
# equivalence probe, which defaults to 8 as the canaries used.
case "$STAGE" in
    build) BATCH_SIZE=${RG_BATCH_SIZE:-1} ;;
    *) BATCH_SIZE=${RG_BATCH_SIZE:-8} ;;
esac
VALIDATION_JSON=${RG_VALIDATION_JSON:-$OUTPUT_DIR/$ENVIRONMENT/validation.json}

case "$ENVIRONMENT" in rope|granular) ;; *) exit 2 ;; esac
if [ "$STAGE" = build ] && [ "$BATCH_SIZE" != 1 ]; then
    echo "REFUSING build at batch size $BATCH_SIZE: the pinned producer contract is literal-singleton (1)" >&2
    exit 2
fi
case "$OUTPUT_DIR" in "$REPAIR_ROOT"/*) ;; *) exit 2 ;; esac
case "$CODE_ROOT" in "$REPAIR_ROOT"/code|"$REPAIR_ROOT"/code-*) ;; *) exit 2 ;; esac
mkdir -p "$REPAIR_ROOT/logs" "$OUTPUT_DIR"

export TRITON_CACHE_DIR=$PROJECT/cache/mapanything-triton/$SLURM_JOB_ID
export XDG_CACHE_HOME=$PROJECT/cache/mapanything-xdg/$SLURM_JOB_ID
export HF_HOME=$PROJECT/cache/mapanything-hf/$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"

APPTAINER_ARGS=(
    --nv
    --writable-tmpfs
    --bind "$PROJECT:/workspace"
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env NUMBA_CACHE_DIR=/tmp/dinocular-rg-ma-numba-$SLURM_JOB_ID
    --env TRITON_CACHE_DIR="/workspace/cache/mapanything-triton/$SLURM_JOB_ID"
    --env XDG_CACHE_HOME="/workspace/cache/mapanything-xdg/$SLURM_JOB_ID"
    --env HF_HOME="/workspace/cache/mapanything-hf/$SLURM_JOB_ID"
    --env TORCH_HOME=/workspace/env/torch
    --env PYTHONPATH=/workspace/env/da3-runtime-site:/workspace/env/mapanything-site:/workspace/code/map-anything:${CODE_ROOT/$PROJECT/\/workspace}
)
TOOL=${CODE_ROOT/$PROJECT/\/workspace}/tools/precompute_depth_mapanything.py
ROOT=/workspace/data/raw
OUT=${OUTPUT_DIR/$PROJECT/\/workspace}
MODEL=/workspace/models/map-anything/$MODEL_REVISION
MAPANYTHING=/workspace/code/map-anything

case "$STAGE" in
    build)
        EXTRA=()
        if [ -n "$SHARD_COUNT" ]; then
            test -n "$SHARD_INDEX"
            EXTRA+=(--shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX")
        fi
        if [ -n "$CALIBRATION_JSON" ]; then
            EXTRA+=(--calibration-json "${CALIBRATION_JSON/$PROJECT/\/workspace}")
        fi
        if [ -n "$MAX_TRAJECTORIES" ]; then
            EXTRA+=(--max-trajectories "$MAX_TRAJECTORIES")
        fi
        /usr/bin/apptainer exec "${APPTAINER_ARGS[@]}" "$SIF" \
            python "$TOOL" build --environment "$ENVIRONMENT" --root "$ROOT" \
            --out "$OUT" --model-dir "$MODEL" --mapanything-root "$MAPANYTHING" \
            --batch-size "$BATCH_SIZE" "${EXTRA[@]}"
        ;;
    merge)
        test -n "$SHARD_COUNT" -a -n "$SHARDS_ROOT"
        /usr/bin/apptainer exec "${APPTAINER_ARGS[@]}" "$SIF" \
            python "$TOOL" merge --environment "$ENVIRONMENT" --root "$ROOT" \
            --shards-root "${SHARDS_ROOT/$PROJECT/\/workspace}" --out "$OUT" \
            --shard-count "$SHARD_COUNT"
        ;;
    validate)
        EXTRA=()
        if [ -n "$MAX_TRAJECTORIES" ]; then
            EXTRA+=(--max-trajectories "$MAX_TRAJECTORIES")
        fi
        mkdir -p "$(dirname "$VALIDATION_JSON")"
        /usr/bin/apptainer exec "${APPTAINER_ARGS[@]}" "$SIF" \
            python "$TOOL" validate --environment "$ENVIRONMENT" --root "$ROOT" \
            --cache "$OUT" --model-dir "$MODEL" --mapanything-root "$MAPANYTHING" \
            --batch-size "$BATCH_SIZE" --spot-frames "$SPOT_FRAMES" \
            --json "${VALIDATION_JSON/$PROJECT/\/workspace}" "${EXTRA[@]}"
        ;;
    *) exit 2 ;;
esac
