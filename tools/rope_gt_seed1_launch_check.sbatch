#!/bin/bash
#SBATCH --job-name=rope-gt-launch-s1
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/rope_gt_launch.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/rope_gt_launch.%j.err

set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
CODE=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/code/dino_wm_gt_rope_seed1
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RUNFILES=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/runfiles
CACHE_DIR=$PROJECT/data/depth_cache_gt/rope.lmdb
VALIDATION_PATH=$PROJECT/data/depth_cache_validation/rope.gt_hdf5_cam1.json
CONTRACT_PATH=$PROJECT/manifests/native_depth_contract_rope_hdf5_camera1_v1.json
PROVENANCE_PATH=$RUNFILES/gt_depth_provenance.json

test -s "$PROVENANCE_PATH"
python3 - "$PROVENANCE_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("done") is not True:
    raise SystemExit("full frozen HDF5 provenance measurement did not pass")
PY

MANIFEST_SHA256=$(sha256sum "$CACHE_DIR/manifest.json" | awk '{print $1}')
VALIDATION_SHA256=$(sha256sum "$VALIDATION_PATH" | awk '{print $1}')
CONTRACT_SHA256=$(sha256sum "$CONTRACT_PATH" | awk '{print $1}')
PRODUCER_SHA256=$(python3 - "$VALIDATION_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["results"]["rope"]["producer_sha256"])
PY
)
CHECKPOINT_SHA256=decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc

export DATASET_DIR=$PROJECT/data/raw
export DINOV2_REPO=$PROJECT/code/dinov2
export DINOV2_VITS14_WEIGHTS=$PROJECT/models/dinov2_vits14_pretrain.pth
export DINOCULAR_STUDENT_WEIGHTS=$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth
export DINOCULAR_NATIVE_DEPTH_CONTRACT=$CONTRACT_PATH
export DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256=$CONTRACT_SHA256
export DINOCULAR_CACHE_PRODUCER_SHA256=$PRODUCER_SHA256
export DINOCULAR_CACHE_ENVIRONMENT=rope
export ROPE_GT_CACHE_DIR=$CACHE_DIR
export ROPE_GT_VALIDATION_PATH=$VALIDATION_PATH
export ROPE_GT_CONTRACT_PATH=$CONTRACT_PATH
export ROPE_GT_CHECKPOINT_SHA256=$CHECKPOINT_SHA256
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONPATH=$CODE:/opt/dinov2

/usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env DATASET_DIR="$DATASET_DIR" \
    --env DINOV2_REPO="$DINOV2_REPO" \
    --env DINOV2_VITS14_WEIGHTS="$DINOV2_VITS14_WEIGHTS" \
    --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
    --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
    --env DINOCULAR_CACHE_ENVIRONMENT="$DINOCULAR_CACHE_ENVIRONMENT" \
    --env ROPE_GT_CACHE_DIR="$ROPE_GT_CACHE_DIR" \
    --env ROPE_GT_VALIDATION_PATH="$ROPE_GT_VALIDATION_PATH" \
    --env ROPE_GT_CONTRACT_PATH="$ROPE_GT_CONTRACT_PATH" \
    --env ROPE_GT_CHECKPOINT_SHA256="$ROPE_GT_CHECKPOINT_SHA256" \
    --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
    --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
    --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
    --env WANDB_MODE="$WANDB_MODE" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$SIF" python "$RUNFILES/rope_gt_seed1_launch_check.py" \
    --code-root "$CODE" \
    --environment rope \
    --cache-manifest-sha256 "$MANIFEST_SHA256" \
    --validation-sha256 "$VALIDATION_SHA256" \
    --contract-sha256 "$CONTRACT_SHA256" \
    --producer-sha256 "$PRODUCER_SHA256"
