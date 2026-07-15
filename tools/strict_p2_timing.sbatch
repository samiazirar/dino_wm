#!/bin/bash
#SBATCH --job-name=p2-strict-dinov2
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p2_strict.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/p2_strict.%x.%j.err

set -euo pipefail

: "${P2_ENV:?set P2_ENV to pusht, wall, rope, or granular}"

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
CODE=$PROJECT/code/dino_wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
SIF_SHA256=6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae
WEIGHTS=$PROJECT/models/dinov2_vits14_pretrain.pth
WEIGHTS_SHA256=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
DINOV2_REPO=$PROJECT/code/dinov2
BASE_COMMIT=12374e46f27cb9074afe5e3973b78c5561f05657
WRAPPER=$CODE/tools/strict_p2_timing.sbatch
RESULT_DIR=$PROJECT/results/p2/strict_dinov2_vits14/$P2_ENV
RUN_DIR=$PROJECT/outputs/p2/strict_dinov2_vits14/$P2_ENV/$SLURM_JOB_ID
CACHE_DIR=$PROJECT/cache/p2-strict-$SLURM_JOB_ID
RESULT=$RESULT_DIR/timing.$SLURM_JOB_ID.json
RUN_CARD=$RESULT_DIR/run_card.$SLURM_JOB_ID.yaml

case "$P2_ENV" in
    pusht)
        ENV_CONFIG=pusht
        NUM_HIST=3
        FRAME_SKIP=5
        PROJECTION_TARGET=123858
        EXTRA_OVERRIDE=env.name=pusht
        ;;
    wall)
        ENV_CONFIG=wall
        NUM_HIST=1
        FRAME_SKIP=5
        PROJECTION_TARGET=143910
        EXTRA_OVERRIDE=env.name=wall
        ;;
    rope)
        ENV_CONFIG=deformable_env
        NUM_HIST=1
        FRAME_SKIP=1
        PROJECTION_TARGET=53500
        EXTRA_OVERRIDE=env.dataset.object_name=rope
        ;;
    granular)
        ENV_CONFIG=deformable_env
        NUM_HIST=1
        FRAME_SKIP=1
        PROJECTION_TARGET=53500
        EXTRA_OVERRIDE=env.dataset.object_name=granular
        ;;
    *)
        echo "Unsupported P2_ENV=$P2_ENV" >&2
        exit 2
        ;;
esac

mkdir -p \
    "$RESULT_DIR" \
    "$RUN_DIR" \
    "$CACHE_DIR/triton" \
    "$CACHE_DIR/xdg" \
    "$CACHE_DIR/hf"

printf '%s  %s\n' "$SIF_SHA256" "$SIF" | sha256sum --check --strict
printf '%s  %s\n' "$WEIGHTS_SHA256" "$WEIGHTS" | sha256sum --check --strict

export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export HF_HOME=$CACHE_DIR/hf
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1
export STRICT_P2_ENV=$P2_ENV
export STRICT_P2_BASE_COMMIT=$BASE_COMMIT
export STRICT_P2_CONTAINER=$SIF
export STRICT_P2_CONTAINER_SHA256=$SIF_SHA256
export STRICT_P2_SLURM_WRAPPER=$WRAPPER
export DINOV2_VITS14_WEIGHTS_SHA256=$WEIGHTS_SHA256

nvidia-smi

OVERRIDES=(
    "env=$ENV_CONFIG"
    "encoder=dino_pinned"
    "training.seed=1"
    "training.batch_size=32"
    "training.predictor_lr=5e-5"
    "training.target_steps=220"
    "training.segment_steps=220"
    "training.resume_from=null"
    "training.checkpoint_every_steps=0"
    "training.strict_determinism=true"
    "training.test_signal_after_step=null"
    "training.timing_output=$RESULT"
    "training.timing_run_card=$RUN_CARD"
    "training.timing_warmup_steps=20"
    "training.timing_measured_steps=200"
    "training.timing_projection_target_steps=$PROJECTION_TARGET"
    "env.num_workers=0"
    "frameskip=$FRAME_SKIP"
    "num_hist=$NUM_HIST"
    "num_pred=1"
    "has_decoder=false"
    "model.train_encoder=false"
    "model.train_predictor=true"
    "model.train_decoder=false"
    "plan_settings.plan_cfg_path=null"
    "ckpt_base_path=$PROJECT"
    "hydra.run.dir=$RUN_DIR"
    "$EXTRA_OVERRIDE"
)

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
    --bind "$PROJECT:$PROJECT" \
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE=disabled \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env PYTHONHASHSEED=1 \
    --env STRICT_P2_ENV="$STRICT_P2_ENV" \
    --env STRICT_P2_BASE_COMMIT="$STRICT_P2_BASE_COMMIT" \
    --env STRICT_P2_CONTAINER="$STRICT_P2_CONTAINER" \
    --env STRICT_P2_CONTAINER_SHA256="$STRICT_P2_CONTAINER_SHA256" \
    --env STRICT_P2_SLURM_WRAPPER="$STRICT_P2_SLURM_WRAPPER" \
    --env DINOV2_VITS14_WEIGHTS_SHA256="$DINOV2_VITS14_WEIGHTS_SHA256" \
    --env DATASET_DIR="$PROJECT/data/raw" \
    --env DINOV2_REPO="$DINOV2_REPO" \
    --env DINOV2_VITS14_WEIGHTS="$WEIGHTS" \
    --env PYTHONPATH="$CODE:/opt/dinov2" \
    "$SIF" \
    python "$CODE/train.py" "${OVERRIDES[@]}"

test -s "$RESULT"
test -s "$RUN_CARD"
printf 'STRICT_P2_JOB=PASS env=%s job=%s result=%s\n' \
    "$P2_ENV" "$SLURM_JOB_ID" "$RESULT"
