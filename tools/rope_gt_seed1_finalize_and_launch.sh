#!/bin/bash
set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
CODE=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/code/dino_wm_gt_rope_seed1
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RUNFILES=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/runfiles
CACHE_DIR=$PROJECT/data/depth_cache_gt/rope.lmdb
VALIDATION_PATH=$PROJECT/data/depth_cache_validation/rope.gt_hdf5_cam1.json
CONTRACT_PATH=$PROJECT/manifests/native_depth_contract_rope_hdf5_camera1_v1.json
PROVENANCE_PATH=$RUNFILES/gt_depth_provenance.json
CARD=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/cards/p3-rope-dinocular-gt-hdf5-cam1-s1.yaml
RUN_DIR=$PROJECT/outputs/campaign-seed1/native-seed1-20260804c/rope/dinocular_gt_hdf5_cam1/run
CHECKPOINT_SHA256=decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc

test -s "$PROVENANCE_PATH"
python3 - "$PROVENANCE_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("done") is not True:
    raise SystemExit("frozen full-coverage provenance did not pass")
PY

test -d "$CACHE_DIR"
test -s "$CACHE_DIR/manifest.json"
test -s "$CACHE_DIR/data.mdb"
test -s "$VALIDATION_PATH"
test -s "$CONTRACT_PATH"
test ! -e "$RUN_DIR"
test ! -e "$CARD"

MANIFEST_SHA256=$(sha256sum "$CACHE_DIR/manifest.json" | awk '{print $1}')
VALIDATION_SHA256=$(sha256sum "$VALIDATION_PATH" | awk '{print $1}')
CONTRACT_SHA256=$(sha256sum "$CONTRACT_PATH" | awk '{print $1}')
PRODUCER_SHA256=$(python3 - "$VALIDATION_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
result = value["results"]["rope"]
if result.get("state") != "PASS":
    raise SystemExit("Rope cache validation is not PASS")
if result.get("frame_count") != 20000 or result.get("trajectory_count") != 1000:
    raise SystemExit("Rope cache coverage is not 1000 trajectories x 20 frames")
print(result["producer_sha256"])
PY
)

export DINOV2_REPO=$PROJECT/code/dinov2
export DINOV2_VITS14_WEIGHTS=$PROJECT/models/dinov2_vits14_pretrain.pth
export DINOCULAR_STUDENT_WEIGHTS=$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth
export DINOCULAR_NATIVE_DEPTH_CONTRACT=$CONTRACT_PATH
export DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256=$CONTRACT_SHA256
export DINOCULAR_CACHE_PRODUCER_SHA256=$PRODUCER_SHA256
export DINOCULAR_CACHE_ENVIRONMENT=rope

/usr/bin/apptainer exec --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env PYTHONNOUSERSITE=1 \
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$DINOV2_REPO" \
    --env DINOV2_VITS14_WEIGHTS="$DINOV2_VITS14_WEIGHTS" \
    --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
    --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
    --env DINOCULAR_CACHE_ENVIRONMENT="$DINOCULAR_CACHE_ENVIRONMENT" \
    --env PYTHONPATH="$CODE:/opt/dinov2" \
    "$SIF" python "$RUNFILES/make_rope_gt_seed1_card.py" \
    --code-root "$CODE" \
    --cache-dir "$CACHE_DIR" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --validation-path "$VALIDATION_PATH" \
    --validation-sha256 "$VALIDATION_SHA256" \
    --contract-path "$CONTRACT_PATH" \
    --contract-sha256 "$CONTRACT_SHA256" \
    --producer-sha256 "$PRODUCER_SHA256" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --output-card "$CARD"

CARD_FILE_SHA256=$(sha256sum "$CARD" | awk '{print $1}')
CARD_SHA256=$(awk '/^run_card_sha256:/{print $2; exit}' "$CARD")
test "${#CARD_SHA256}" -eq 64
test ! -e "$RUN_DIR"

export DINOV2_REPO DINOV2_VITS14_WEIGHTS DINOCULAR_STUDENT_WEIGHTS
export DINOCULAR_NATIVE_DEPTH_CONTRACT DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256
export DINOCULAR_CACHE_PRODUCER_SHA256 DINOCULAR_CACHE_ENVIRONMENT
python3 "$CODE/tools/submit_p3_chain.py" start \
    --run-dir "$RUN_DIR" \
    --target-steps 53500 \
    --segment-steps 1800 \
    --checkpoint-every-steps 500 \
    --partition sgpu_short \
    --time-limit 07:55:00 \
    --job-name rope-gt-dinocular-s1 \
    --run-card "$CARD" \
    --run-card-file-sha256 "$CARD_FILE_SHA256" \
    --run-card-sha256 "$CARD_SHA256"
