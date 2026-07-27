#!/bin/bash
#SBATCH --job-name=p3-resume
#SBATCH --partition=sgpu_short
#SBATCH --time=07:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p3_resume.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p3_resume.%x.%j.err

set -euo pipefail

: "${P3_CHAIN_MANIFEST:?P3_CHAIN_MANIFEST is required}"
PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
SUBMITTER=$PROJECT/code/dino_wm/tools/submit_p3_chain.py
CACHE_ROOT=$PROJECT/cache/p3-resume/$SLURM_JOB_ID
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export TORCH_HOME=$PROJECT/env/torch

case "$TRITON_CACHE_DIR" in
    "$PROJECT"/cache/p3-resume/"$SLURM_JOB_ID"/triton) ;;
    *) printf 'Refusing unsafe TRITON_CACHE_DIR: %s\n' "$TRITON_CACHE_DIR" >&2; exit 1 ;;
esac
case "$XDG_CACHE_HOME" in
    "$PROJECT"/cache/p3-resume/"$SLURM_JOB_ID"/xdg) ;;
    *) printf 'Refusing unsafe XDG_CACHE_HOME: %s\n' "$XDG_CACHE_HOME" >&2; exit 1 ;;
esac
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
test -w "$TRITON_CACHE_DIR" -a -w "$XDG_CACHE_HOME" -a -w "$HF_HOME"

python3 "$SUBMITTER" verify --manifest "$P3_CHAIN_MANIFEST"

RUN_DIR=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field run_dir)
TARGET_STEPS=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field target_steps)
SEGMENT_STEPS=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field segment_steps)
CHECKPOINT_EVERY=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field checkpoint_every_steps)
RUN_CARD=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field run_card)
RUN_CARD_SHA256=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field run_card_sha256)
RUN_KIND=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field kind)
P2_ENV=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field environment)
P2_ARM=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field arm)
SIF_SHA256=$(python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field container_sha256)
mapfile -t OVERRIDES < <(
    python3 "$SUBMITTER" emit --manifest "$P3_CHAIN_MANIFEST" --field overrides
)

TRAIN_ARGS=(
    python "$PROJECT/code/dino_wm/train.py"
    "training.target_steps=$TARGET_STEPS"
    "training.segment_steps=$SEGMENT_STEPS"
    "training.checkpoint_every_steps=$CHECKPOINT_EVERY"
    "training.test_signal_after_step=null"
    "ckpt_base_path=$RUN_DIR"
    "hydra.run.dir=$RUN_DIR"
    "hydra.job.chdir=true"
    "hydra.output_subdir=null"
)
TRAIN_ARGS+=("${OVERRIDES[@]}")

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export STRICT_P2_ENV=$P2_ENV
export STRICT_P2_ARM=$P2_ARM
export STRICT_P2_CONTAINER=$SIF
export STRICT_P2_CONTAINER_SHA256=$SIF_SHA256
export STRICT_P2_SLURM_WRAPPER=$PROJECT/code/dino_wm/tools/p3_step_segment.sbatch
export STRICT_P2_IMMUTABLE_RUN_CARD=$RUN_CARD
export STRICT_P2_IMMUTABLE_RUN_CARD_SHA256=$RUN_CARD_SHA256

source "$PROJECT/code/dino_wm/tools/dinocular_container_env.sh"
build_dinocular_container_env "$P2_ARM"

nvidia-smi
sha256sum "$SIF"

setsid /usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
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
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$PROJECT/code/dinov2" \
    --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
    "${DINOCULAR_CONTAINER_ENV[@]}" \
    --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
    "$SIF" \
    "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!

forward_usr1() {
    kill -USR1 -- "-$TRAIN_PID" 2>/dev/null || kill -USR1 "$TRAIN_PID" 2>/dev/null || true
}
forward_term() {
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null || true
}
trap forward_usr1 USR1
trap forward_term TERM

set +e
# Re-wait past EINTR: SLURM USR1 wakes `wait` (returns 138) before python finishes
# checkpointing. Proven repair, see /tmp/dinocular_marvin_campaign_relaunch_state.md.
while :; do
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
    kill -0 "$TRAIN_PID" 2>/dev/null || break
done
set -e
if [ "$TRAIN_STATUS" -ne 0 ]; then
    printf 'Training process failed with status %s\n' "$TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi

PROGRESS_STATUS=$(python3 - "$RUN_DIR/progress.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [ "$RUN_KIND" = p3-training ] && [ "$PROGRESS_STATUS" = TARGET_REACHED ]; then
    /usr/bin/apptainer exec --nv --writable-tmpfs \
        --bind "$PROJECT:$PROJECT" \
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
        --env P3_FINAL_ACCEPTANCE_PROCESS=1 \
        --env SLURM_JOB_ID="$SLURM_JOB_ID" \
        --env STRICT_P2_ENV="$STRICT_P2_ENV" \
        --env STRICT_P2_ARM="$STRICT_P2_ARM" \
        --env STRICT_P2_CONTAINER="$STRICT_P2_CONTAINER" \
        --env STRICT_P2_CONTAINER_SHA256="$STRICT_P2_CONTAINER_SHA256" \
        --env STRICT_P2_SLURM_WRAPPER="$STRICT_P2_SLURM_WRAPPER" \
        --env STRICT_P2_IMMUTABLE_RUN_CARD="$STRICT_P2_IMMUTABLE_RUN_CARD" \
        --env STRICT_P2_IMMUTABLE_RUN_CARD_SHA256="$STRICT_P2_IMMUTABLE_RUN_CARD_SHA256" \
        --env DATASET_DIR="$PROJECT/data/raw" \
        --env DINOV2_REPO="$PROJECT/code/dinov2" \
        --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
        "${DINOCULAR_CONTAINER_ENV[@]}" \
        --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
        "$SIF" \
        "${TRAIN_ARGS[@]}" "training.final_acceptance=true"
fi

case "$RUN_KIND" in
    p2-geometry)
        /usr/bin/apptainer exec --nv --writable-tmpfs \
            --bind "$PROJECT:$PROJECT" \
            --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
            --env DATASET_DIR="$PROJECT/data/raw" \
            --env DINOV2_REPO="$PROJECT/code/dinov2" \
            --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
            "${DINOCULAR_CONTAINER_ENV[@]}" \
            "$SIF" python "$PROJECT/code/dino_wm/tools/p2_harness_gate.py" geometry \
            --run-card "$RUN_CARD" --run-dir "$RUN_DIR"
        ;;
    p2-timing)
        /usr/bin/apptainer exec --nv --writable-tmpfs \
            --bind "$PROJECT:$PROJECT" \
            --env PYTHONPATH="$PROJECT/code/dino_wm:/opt/dinov2" \
            "${DINOCULAR_CONTAINER_ENV[@]}" \
            "$SIF" python "$PROJECT/code/dino_wm/tools/p2_harness_gate.py" timing \
            --run-card "$RUN_CARD" --run-dir "$RUN_DIR"
        ;;
    *) ;;
esac

NEXT_JOB=$(
    python3 "$SUBMITTER" continue \
        --manifest "$P3_CHAIN_MANIFEST" \
        --parent-job "$SLURM_JOB_ID"
)
printf 'P3_SEGMENT=PASS job=%s next=%s\n' "$SLURM_JOB_ID" "$NEXT_JOB"
