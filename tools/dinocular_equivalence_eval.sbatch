#!/bin/bash
#SBATCH --job-name=vera-dinocular-equivalence
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/vera_equivalence.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/vera_equivalence.%x.%j.err

set -euo pipefail

: "${EQUIV_CODE_ROOT:?EQUIV_CODE_ROOT is required}"
: "${EQUIV_RUNNER_ROOT:?EQUIV_RUNNER_ROOT is required}"
: "${EQUIV_RUN_DIR:?EQUIV_RUN_DIR is required}"

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
CACHE_ROOT=$PROJECT/cache/vera-equivalence/$SLURM_JOB_ID
RESULT=$EQUIV_RUN_DIR/equivalence.json
mkdir -p "$EQUIV_RUN_DIR" "$CACHE_ROOT/triton" "$CACHE_ROOT/xdg" "$CACHE_ROOT/hf"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export TORCH_HOME=$PROJECT/env/torch
export DINOCULAR_STUDENT_WEIGHTS=$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth
export DINOCULAR_NATIVE_DEPTH_CONTRACT=$PROJECT/manifests/native_depth_contract_v1.json
export DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256=a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b
export DINOCULAR_CACHE_PRODUCER_SHA256=c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc
export DINOCULAR_CACHE_ENVIRONMENT=rope
export EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256=5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba
export EQUIVALENCE_DEPTH_VALIDATION_SHA256=3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86
export DINOCULAR_EQUIVALENCE_OUTPUT=$RESULT

printf '%s  %s\n' \
    "$EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256" "$PROJECT/data/depth_cache/rope.lmdb/manifest.json" \
    "$EQUIVALENCE_DEPTH_VALIDATION_SHA256" "$PROJECT/data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json" \
    "$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" "$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc" "$DINOCULAR_STUDENT_WEIGHTS" \
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
    --env EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256="$EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256" \
    --env EQUIVALENCE_DEPTH_VALIDATION_SHA256="$EQUIVALENCE_DEPTH_VALIDATION_SHA256" \
    --env DINOCULAR_EQUIVALENCE_OUTPUT="$RESULT" \
    --env PYTHONPATH="$EQUIV_CODE_ROOT:/opt/dinov2" \
    "$SIF" \
    python "$EQUIV_RUNNER_ROOT/evaluate_dinocular_equivalence.py" \
    env=deformable_env \
    encoder=dinocular \
    training.seed=1 \
    training.batch_size=32 \
    training.predictor_lr=5e-5 \
    training.target_steps=1 \
    training.segment_steps=1 \
    training.resume_from=null \
    training.checkpoint_every_steps=0 \
    training.strict_determinism=true \
    training.test_signal_after_step=null \
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
    +env.dataset.depth_cache_manifest_sha256="$EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256" \
    +env.dataset.depth_validation_path="$PROJECT/data/depth_cache_validation/rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json" \
    +env.dataset.depth_validation_sha256="$EQUIVALENCE_DEPTH_VALIDATION_SHA256" \
    +env.dataset.native_depth_contract_path="$DINOCULAR_NATIVE_DEPTH_CONTRACT" \
    +env.dataset.native_depth_contract_sha256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256" \
    +env.dataset.depth_cache_producer_sha256="$DINOCULAR_CACHE_PRODUCER_SHA256" \
    +env.dataset.depth_checkpoint_sha256=decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc \
    ckpt_base_path="$EQUIV_RUN_DIR" \
    hydra.run.dir="$EQUIV_RUN_DIR" \
    hydra.job.chdir=true \
    hydra.output_subdir=null

test -s "$RESULT"
