#!/bin/bash
#SBATCH --job-name=dinocular-depth-da3-pusht-merge-validate
#SBATCH --partition=sgpu_medium
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_merge_validate.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/da3_pusht_merge_validate.%j.err

set -euo pipefail

: "${DA3_PLAN_FILE_SHA256:?DA3_PLAN_FILE_SHA256 is required}"
case "$DA3_PLAN_FILE_SHA256" in [0-9a-f][0-9a-f]*) ;; *) exit 2 ;; esac
test "${#DA3_PLAN_FILE_SHA256}" -eq 64

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
PLAN=$PROJECT/data/depth_cache_da3_pusht_shards/plan.json
REPORT=$PROJECT/data/depth_cache_validation/pusht.da3-sharded.$SLURM_JOB_ID.json
EVIDENCE=$PROJECT/data/depth_cache_validation/pusht.da3-sharded.$SLURM_JOB_ID.evidence.json
test -f "$PLAN"
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$DA3_PLAN_FILE_SHA256"
test ! -e "$PROJECT/data/depth_cache/pusht.lmdb"
test ! -e "$REPORT"
test ! -e "$EVIDENCE"

CACHE_ROOT=$PROJECT/cache/da3-pusht-merge/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$(dirname "$REPORT")"
test -w "$TRITON_CACHE_DIR" -a -w "$XDG_CACHE_HOME" -a -w "$HF_HOME"

APPTAINER=(
    /usr/bin/apptainer exec --nv --writable-tmpfs
    --bind "$PROJECT:/workspace"
    --env NUMBA_CACHE_DIR=/tmp/dinocular-numba-$SLURM_JOB_ID
    --env TRITON_CACHE_DIR=/workspace/cache/da3-pusht-merge/$SLURM_JOB_ID/triton
    --env XDG_CACHE_HOME=/workspace/cache/da3-pusht-merge/$SLURM_JOB_ID/xdg
    --env HF_HOME=/workspace/cache/da3-pusht-merge/$SLURM_JOB_ID/hf
    --env TORCH_HOME=/workspace/env/torch
    --env PYTHONPATH=/workspace/env/da3-runtime-site:/workspace/code/dino_wm:/workspace/code/depth-anything-3/src:/workspace/code/depth-anything-3/da3_streaming
    "$SIF"
)

"${APPTAINER[@]}" python /workspace/code/dino_wm/tools/precompute_depth_da3_shards.py merge \
    --root /workspace/data/raw \
    --shards-root /workspace/data/depth_cache_da3_pusht_shards \
    --out /workspace/data/depth_cache \
    --plan /workspace/data/depth_cache_da3_pusht_shards/plan.json

"${APPTAINER[@]}" python /workspace/code/dino_wm/tools/validate_depth_cache.py \
    --root /workspace/data/raw \
    --cache /workspace/data/depth_cache \
    --environments pusht \
    --recompute-trajectory-fraction 0.01 \
    --da3-root /workspace/code/depth-anything-3 \
    --model-dir /workspace/models/da3nested_giant_large_1_1 \
    --json /workspace/data/depth_cache_validation/$(basename "$REPORT")

python3 - "$PROJECT/data/depth_cache/pusht.lmdb/manifest.json" "$REPORT" "$EVIDENCE" "$DA3_PLAN_FILE_SHA256" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
report = Path(sys.argv[2])
output = Path(sys.argv[3])
plan_file_sha256 = sys.argv[4]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            value.update(block)
    return value.hexdigest()

receipt = json.loads(report.read_text(encoding="utf-8"))
manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
if receipt.get("state") != "PASS" or receipt.get("results", {}).get("pusht", {}).get("state") != "PASS":
    raise SystemExit("validator receipt is not PASS")
value = {
    "schema": "dinocular-da3-pusht-promotion-evidence-v1",
    "plan_sha256": manifest_record["merge_provenance"]["plan_sha256"],
    "plan_file_sha256": plan_file_sha256,
    "manifest_path": str(manifest),
    "manifest_sha256": digest(manifest),
    "validation_receipt_path": str(report),
    "validation_receipt_sha256": digest(report),
}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
print(json.dumps(value, sort_keys=True))
PY
