#!/bin/bash
#SBATCH --job-name=gaia-zero-depth-equivalence
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/gaia_zero_depth_equivalence.%x.%j.out
#SBATCH --error=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/logs/gaia_zero_depth_equivalence.%x.%j.err

set -euo pipefail

PROJECT=/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm
SOURCE_CODE=$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/code/dino_wm
SOURCE_COMMIT=22aa276bbc0ae9e76d603e4f87cb360d684622bd
TOOL=$PROJECT/staging/gaia-zero-depth-functional-equivalence-luna/measure_zero_depth_functional_equivalence.py
SIF=$PROJECT/containers/dinocular-wm-p0-20260714.sif
RESULT=$PROJECT/outputs/gaia-zero-depth-functional-equivalence/zero_depth_functional_equivalence.json
RESULT_DIR=$(dirname "$RESULT")
CLEAN_ROOT=/tmp/gaia-zero-depth-functional-equivalence-${SLURM_JOB_ID}/code
CACHE_ROOT=/tmp/gaia-zero-depth-functional-equivalence-${SLURM_JOB_ID}/cache

mkdir -p "$CLEAN_ROOT" "$CACHE_ROOT/triton" "$CACHE_ROOT/xdg" "$CACHE_ROOT/hf" "$RESULT_DIR"

git -C "$SOURCE_CODE" archive "$SOURCE_COMMIT" | tar -x -C "$CLEAN_ROOT"
test -s "$TOOL"
printf '%s  %s\n' \
  "4f17cc3dc0313b9c365303abecf42eb83578738f66f16c7940edaf1a1b69f70a" "$TOOL" \
  "6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae" "$SIF" \
  | sha256sum --check --strict
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

CARD_ROOT_1=$PROJECT/outputs/campaign-seed1/native-seed1-20260728c/cards
CARD_ROOT_23=$PROJECT/outputs/campaign-seeds23/seeds23-20260728a/cards

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  --bind "$PROJECT:$PROJECT" \
  --bind "$CLEAN_ROOT:$CLEAN_ROOT" \
  --env CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
  --env NVIDIA_TF32_OVERRIDE="$NVIDIA_TF32_OVERRIDE" \
  --env PYTHONHASHSEED="$PYTHONHASHSEED" \
  --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  --env WANDB_MODE="$WANDB_MODE" \
  --env TRITON_CACHE_DIR="/tmp/gaia-zero-depth-functional-equivalence-${SLURM_JOB_ID}/cache/triton" \
  --env XDG_CACHE_HOME="/tmp/gaia-zero-depth-functional-equivalence-${SLURM_JOB_ID}/cache/xdg" \
  --env HF_HOME="/tmp/gaia-zero-depth-functional-equivalence-${SLURM_JOB_ID}/cache/hf" \
  --env TORCH_HOME="$PROJECT/env/torch" \
  --env DATASET_DIR="$PROJECT/data/raw" \
  --env DINOV2_REPO="$PROJECT/code/dinov2" \
  --env DINOV2_VITS14_WEIGHTS="$PROJECT/models/dinov2_vits14_pretrain.pth" \
  --env DINOCULAR_STUDENT_WEIGHTS="$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth" \
  --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$PROJECT/manifests/native_depth_contract_v1.json" \
  --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="a558b4fee9eb380a40de138e372a12f0796ef9af6b5eef9901f6e7d3853b3d4b" \
  --env DINOCULAR_CACHE_PRODUCER_SHA256="c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc" \
  --env DINOCULAR_CACHE_ENVIRONMENT=rope \
  --env PYTHONPATH="$CLEAN_ROOT:/opt/dinov2" \
  "$SIF" python "$TOOL" \
  --implementation-root "$CLEAN_ROOT" \
  --dataset-dir "$PROJECT/data/raw" \
  --output "$RESULT" \
  --card "$CARD_ROOT_1/p3-wall-dinocular_zerodepth-s1.yaml" \
  --card "$CARD_ROOT_1/p3-rope-dinocular_zerodepth-s1.yaml" \
  --card "$CARD_ROOT_1/p3-granular-dinocular_zerodepth-s1.yaml" \
  --card "$CARD_ROOT_23/p3-wall-dinocular_zerodepth-s2.yaml" \
  --card "$CARD_ROOT_23/p3-rope-dinocular_zerodepth-s2.yaml" \
  --card "$CARD_ROOT_23/p3-granular-dinocular_zerodepth-s2.yaml" \
  --card "$CARD_ROOT_23/p3-wall-dinocular_zerodepth-s3.yaml" \
  --card "$CARD_ROOT_23/p3-rope-dinocular_zerodepth-s3.yaml" \
  --card "$CARD_ROOT_23/p3-granular-dinocular_zerodepth-s3.yaml"

test -s "$RESULT"
