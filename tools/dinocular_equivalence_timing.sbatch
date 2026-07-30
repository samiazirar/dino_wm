#!/bin/bash
#SBATCH --job-name=vera-dinocular-timing
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/vera_timing.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/vera_timing.%x.%j.err

set -euo pipefail

: "${EQUIV_CODE_ROOT:?EQUIV_CODE_ROOT is required}"
: "${EQUIV_RUN_DIR:?EQUIV_RUN_DIR is required}"
: "${EQUIV_LABEL:?EQUIV_LABEL is required}"

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RUNNER_ROOT=$PROJECT/outputs/vera-dinocular-equivalence/runner
WRAPPER=$RUNNER_ROOT/dinocular_equivalence_timing.sbatch
CACHE_ROOT=$PROJECT/cache/vera-timing/$SLURM_JOB_ID
RESULT=$EQUIV_RUN_DIR/timing_result.json
RUNTIME_CARD=$EQUIV_RUN_DIR/timing_runtime_card.yaml
IMMUTABLE_CARD=$EQUIV_RUN_DIR/immutable_run_card.json
mkdir -p "$EQUIV_RUN_DIR" "$CACHE_ROOT/triton" "$CACHE_ROOT/xdg" "$CACHE_ROOT/hf"
test -f "$WRAPPER"

export DINOCULAR_STUDENT_WEIGHTS=$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth
export DINOCULAR_NATIVE_DEPTH_CONTRACT=$PROJECT/manifests/native_depth_contract_v1.json
export DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256=a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b
export DINOCULAR_CACHE_PRODUCER_SHA256=c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc
export DINOCULAR_CACHE_ENVIRONMENT=rope

python3 - "$EQUIV_CODE_ROOT" "$EQUIV_RUN_DIR" "$IMMUTABLE_CARD" "$WRAPPER" "$EQUIV_LABEL" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

code = pathlib.Path(sys.argv[1]).resolve()
run_dir = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3]).resolve()
wrapper = pathlib.Path(sys.argv[4]).resolve()
label = sys.argv[5]
project = pathlib.Path("/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm")

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

commit = subprocess.check_output(
    ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
).strip()
card = {
    "schema": "dino-wm.legacy-run-card.v1",
    "kind": "p2-timing",
    "gate_mode": "timing",
    "run_id": f"vera-rope-{label}",
    "arm": "dinocular",
    "environment": "rope",
    "source_commit": commit,
    "batch_size": 32,
    "frameskip": 1,
    "target_steps": 220,
    "segment_steps": 220,
    "timing": {"fixed_steps": 200, "warmup_steps": 20},
    "run_dir": str(run_dir),
    "container": {
        "path": str(project / "containers/dinocular-wm-p0-20260714.sif"),
        "sha256": "6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae",
    },
    "artifacts": {
        "dinov2": {
            "path": str(project / "models/dinov2_vits14_pretrain.pth"),
            "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        },
        "dinocular_student": {
            "path": str(project / "checkpoints/dinov2_depthembed_dropout_fullpr.pth"),
            "sha256": "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc",
        },
    },
    "depth_inputs": {
        "cache_dir": str(project / "data/depth_cache/rope.lmdb"),
        "cache_manifest_sha256": "5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba",
        "validation_path": str(project / "data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json"),
        "validation_sha256": "3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86",
        "native_contract_path": str(project / "manifests/native_depth_contract_v1.json"),
        "native_contract_sha256": "a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b",
        "producer_sha256": "c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc",
        "checkpoint_sha256": "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc",
    },
    "source_file_sha256": {
        "train.py": sha(code / "train.py"),
        "training_resume.py": sha(code / "training_resume.py"),
        "training_timing.py": sha(code / "training_timing.py"),
        "tools/p3_step_segment.sbatch": sha(wrapper),
    },
}
encoded = json.dumps(card, sort_keys=True, separators=(",", ":")).encode()
card["run_card_sha256"] = hashlib.sha256(encoded).hexdigest()
output.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
print(card["run_card_sha256"])
PY

CARD_SHA256=$(python3 - "$IMMUTABLE_CARD" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["run_card_sha256"])
PY
)

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export STRICT_P2_COMPONENT_TIMING=1
export STRICT_P2_ENV=rope
export STRICT_P2_ARM=dinocular
export STRICT_P2_CONTAINER=$SIF
export STRICT_P2_CONTAINER_SHA256=6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae
export STRICT_P2_SLURM_WRAPPER=$WRAPPER
export STRICT_P2_IMMUTABLE_RUN_CARD=$IMMUTABLE_CARD
export STRICT_P2_IMMUTABLE_RUN_CARD_SHA256=$CARD_SHA256
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export TORCH_HOME=$PROJECT/env/torch

printf '%s  %s\n' \
    "5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba" "$PROJECT/data/depth_cache/rope.lmdb/manifest.json" \
    "3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86" "$PROJECT/data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json" \
    "$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" "$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc" "$DINOCULAR_STUDENT_WEIGHTS" \
    "6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae" "$SIF" \
    | sha256sum --check --strict
nvidia-smi

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    --env STRICT_P2_COMPONENT_TIMING=1 \
    --env STRICT_P2_ENV=rope \
    --env STRICT_P2_ARM=dinocular \
    --env STRICT_P2_CONTAINER="$SIF" \
    --env STRICT_P2_CONTAINER_SHA256="$STRICT_P2_CONTAINER_SHA256" \
    --env STRICT_P2_SLURM_WRAPPER="$WRAPPER" \
    --env STRICT_P2_IMMUTABLE_RUN_CARD="$IMMUTABLE_CARD" \
    --env STRICT_P2_IMMUTABLE_RUN_CARD_SHA256="$CARD_SHA256" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env TORCH_HOME="$TORCH_HOME" \
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$PROJECT/code/dinov2" \
    --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
    --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
    --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
    --env DINOCULAR_CACHE_ENVIRONMENT=rope \
    --env PYTHONPATH="$EQUIV_CODE_ROOT:/opt/dinov2" \
    "$SIF" \
    python "$EQUIV_CODE_ROOT/train.py" \
    env=deformable_env \
    encoder=dinocular \
    training.seed=1 \
    training.batch_size=32 \
    training.predictor_lr=5e-5 \
    training.target_steps=220 \
    training.segment_steps=220 \
    training.resume_from=null \
    training.checkpoint_every_steps=0 \
    training.strict_determinism=true \
    training.test_signal_after_step=null \
    training.timing_output="$RESULT" \
    training.timing_run_card="$RUNTIME_CARD" \
    training.timing_warmup_steps=20 \
    training.timing_measured_steps=200 \
    training.timing_projection_target_steps=53500 \
    env.num_workers=0 \
    frameskip=1 \
    num_hist=1 \
    num_pred=1 \
    has_decoder=false \
    model.train_encoder=false \
    model.train_predictor=true \
    model.train_decoder=false \
    plan_settings.plan_cfg_path=null \
    env.dataset.object_name=rope \
    env.kwargs.object_name=rope \
    +env.dataset.depth_cache_dir="$PROJECT/data/depth_cache/rope.lmdb" \
    +env.dataset.depth_cache_manifest_sha256=5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba \
    +env.dataset.depth_validation_path="$PROJECT/data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json" \
    +env.dataset.depth_validation_sha256=3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86 \
    +env.dataset.native_depth_contract_path="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    +env.dataset.native_depth_contract_sha256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
    +env.dataset.depth_cache_producer_sha256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
    +env.dataset.depth_checkpoint_sha256=decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc \
    ckpt_base_path="$EQUIV_RUN_DIR" \
    hydra.run.dir="$EQUIV_RUN_DIR" \
    hydra.job.chdir=true \
    hydra.output_subdir=null

test -s "$RESULT"
