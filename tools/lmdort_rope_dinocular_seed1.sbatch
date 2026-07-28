#!/bin/bash
#SBATCH --job-name=rope-dinocular-s1
#SBATCH --partition=GPU1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/tmp
#SBATCH --output=/cephfs/users/azirar/projects/dinocular-wm-lmdort-rope-dino-s1/lmdort/logs/dinocular.segment.%j.out
#SBATCH --error=/cephfs/users/azirar/projects/dinocular-wm-lmdort-rope-dino-s1/lmdort/logs/dinocular.segment.%j.err

set -euo pipefail

ROOT=/cephfs/users/azirar/projects/dinocular-wm-lmdort-rope-dino-s1
CANONICAL=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
CODE_ROOT=$CANONICAL/outputs/campaign-seed1/native-seed1-20260728c/code/dino_wm
HOST_CODE_ROOT=$ROOT/outputs/campaign-seed1/native-seed1-20260728c/code/dino_wm
SIF=$ROOT/containers/dinocular-wm-p0-20260714.sif
RUN_DIR=$CANONICAL/outputs/campaign-seed1/native-seed1-20260728c/rope/dinocular/run
HOST_RUN_DIR=$ROOT/outputs/campaign-seed1/native-seed1-20260728c/rope/dinocular/run
CARD=$CANONICAL/outputs/campaign-seed1/native-seed1-20260728c/cards/p3-rope-dinocular-s1.yaml
WRAPPER=$CANONICAL/lmdort/lmdort_rope_dinocular_seed1.sbatch
ARM=dinocular
CACHE_ROOT=$ROOT/cache/p3-resume-dinocular/$SLURM_JOB_ID

export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export TORCH_HOME=$ROOT/env/torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export STRICT_P2_ENV=rope
export STRICT_P2_ARM=$ARM
export STRICT_P2_CONTAINER=$CANONICAL/containers/dinocular-wm-p0-20260714.sif
export STRICT_P2_CONTAINER_SHA256=6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae
export STRICT_P2_SLURM_WRAPPER=$WRAPPER
export STRICT_P2_IMMUTABLE_RUN_CARD=$CARD
export STRICT_P2_IMMUTABLE_RUN_CARD_SHA256=8f41b1cc8cf31ce0cdcbb457742869636f8b9880b7585ff90f2180194105359f
export DINOV2_REPO=$CANONICAL/code/dinov2
export DINOV2_VITS14_WEIGHTS=$CANONICAL/models/dinov2_vits14_pretrain.pth
export DINOCULAR_STUDENT_WEIGHTS=$CANONICAL/checkpoints/dinov2_depthembed_dropout_fullpr.pth
export DINOCULAR_NATIVE_DEPTH_CONTRACT=$CANONICAL/manifests/native_depth_contract_v1.json
export DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256=a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b
export DINOCULAR_CACHE_PRODUCER_SHA256=c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc
export DINOCULAR_CACHE_ENVIRONMENT=rope

mkdir -p \
    "$HOST_RUN_DIR" \
    "$ROOT/lmdort/logs" \
    "$TRITON_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$TORCH_HOME"

source "$HOST_CODE_ROOT/tools/dinocular_container_env.sh"
build_dinocular_container_env "$ARM"

TRAIN_ARGS=(
    python "$CODE_ROOT/train.py"
    training.target_steps=53500
    training.segment_steps=53500
    training.checkpoint_every_steps=500
    training.test_signal_after_step=null
    ckpt_base_path="$RUN_DIR"
    hydra.run.dir="$RUN_DIR"
    hydra.job.chdir=true
    hydra.output_subdir=null
    env=deformable_env
    encoder=dinocular
    training.seed=1
    training.predictor_lr=5e-5
    training.strict_determinism=true
    training.resume_from=auto
    training.batch_size=32
    env.num_workers=0
    frameskip=1
    num_hist=1
    num_pred=1
    has_decoder=false
    model.train_encoder=false
    model.train_predictor=true
    model.train_decoder=false
    plan_settings.plan_cfg_path=null
    env.dataset.object_name=rope
    env.kwargs.object_name=rope
    +env.dataset.depth_cache_dir=$CANONICAL/data/depth_cache/rope.lmdb
    +env.dataset.depth_cache_manifest_sha256=5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba
    +env.dataset.depth_validation_path=$CANONICAL/data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json
    +env.dataset.depth_validation_sha256=3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86
    +env.dataset.native_depth_contract_path=$CANONICAL/manifests/native_depth_contract_v1.json
    +env.dataset.native_depth_contract_sha256=a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b
    +env.dataset.depth_cache_producer_sha256=c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc
    +env.dataset.depth_checkpoint_sha256=decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc
    training.p3_completion_enabled=true
    training.p3_heldout_manifest=$CANONICAL/outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.jsonl
    training.p3_heldout_manifest_sha256=82672d804b198c31258105e294237c2716dd31e34c07ba05fdc9f5002ef0c607
    training.p3_heldout_metadata=$CANONICAL/outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.meta.json
    training.p3_heldout_metadata_sha256=280f11df2ac40011fed3d16b9299d48b550452e30875fd24287d3fbd9248b8c2
    training.p3_data_manifest_sha256=f7e51a2a87b9c197a7c4b5416a765f7a5b73ffb67b8dde1d7614e772cfab194f
    training.p3_split_sha256=2d7a406813f78b6cbdfd0777c579d6a423e548badaebb56281c4dbf48cccf751
)

read_progress_field() {
    python3 - "$HOST_RUN_DIR/progress.json" "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print("")
else:
    value = json.loads(path.read_text(encoding="utf-8"))
    print(value.get(sys.argv[2], ""))
PY
}

run_training() {
    setsid /usr/bin/apptainer exec --nv --writable-tmpfs \
        --bind "$ROOT:$CANONICAL" \
        --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        --env HF_HOME="$HF_HOME" \
        --env TORCH_HOME="$TORCH_HOME" \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --env NVIDIA_TF32_OVERRIDE=0 \
        --env PYTHONHASHSEED=0 \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --env WANDB_MODE=disabled \
        --env STRICT_P2_ENV="$STRICT_P2_ENV" \
        --env STRICT_P2_ARM="$STRICT_P2_ARM" \
        --env STRICT_P2_CONTAINER="$STRICT_P2_CONTAINER" \
        --env STRICT_P2_CONTAINER_SHA256="$STRICT_P2_CONTAINER_SHA256" \
        --env STRICT_P2_SLURM_WRAPPER="$STRICT_P2_SLURM_WRAPPER" \
        --env STRICT_P2_IMMUTABLE_RUN_CARD="$STRICT_P2_IMMUTABLE_RUN_CARD" \
        --env STRICT_P2_IMMUTABLE_RUN_CARD_SHA256="$STRICT_P2_IMMUTABLE_RUN_CARD_SHA256" \
        --env DATASET_DIR="$CANONICAL/data/raw" \
        --env DINOV2_REPO="$DINOV2_REPO" \
        --env DINOV2_VITS14_WEIGHTS="$DINOV2_VITS14_WEIGHTS" \
        --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS" \
        --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
        --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
        --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
        --env DINOCULAR_CACHE_ENVIRONMENT="$DINOCULAR_CACHE_ENVIRONMENT" \
        "${DINOCULAR_CONTAINER_ENV[@]}" \
        --env PYTHONPATH="$CODE_ROOT:/opt/dinov2" \
        "$SIF" "${TRAIN_ARGS[@]}" "$@" &
    TRAIN_PID=$!
}

forward_usr1() {
    kill -USR1 -- "-$TRAIN_PID" 2>/dev/null ||
        kill -USR1 "$TRAIN_PID" 2>/dev/null ||
        true
}
forward_term() {
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null ||
        kill -TERM "$TRAIN_PID" 2>/dev/null ||
        true
}
trap forward_usr1 USR1
trap forward_term TERM

wait_for_training() {
    local before_step before_checkpoint status after_step after_checkpoint
    before_step=$(read_progress_field global_step)
    before_checkpoint=$(read_progress_field checkpoint_sha256)
    set +e
    while :; do
        wait "$TRAIN_PID"
        TRAIN_STATUS=$?
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            continue
        fi
        break
    done
    set -e
    if [ "$TRAIN_STATUS" -eq 0 ]; then
        return
    fi

    status=$(read_progress_field status)
    after_step=$(read_progress_field global_step)
    after_checkpoint=$(read_progress_field checkpoint_sha256)
    if [ "$status" = SIGNAL_CHECKPOINTED ] &&
        [ -n "$after_step" ] &&
        [ "$after_step" -gt "${before_step:-0}" ] &&
        [ -n "$after_checkpoint" ] &&
        [ "$after_checkpoint" != "$before_checkpoint" ]; then
        TRAIN_STATUS=0
        return
    fi
    return "$TRAIN_STATUS"
}

run_training
wait_for_training

PROGRESS_STATUS=$(read_progress_field status)
if [ "$PROGRESS_STATUS" = TARGET_REACHED ]; then
    run_training training.final_acceptance=true
    wait_for_training
fi
