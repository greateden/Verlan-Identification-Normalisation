#!/usr/bin/env bash
#SBATCH --job-name=verlan-e2e
#SBATCH --partition=aoraki_gpu_A100_80GB
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Fail fast if account not provided via command line or environment wrapper.
if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
  echo "[WARN] No --account provided. Submit with: sbatch --account=YOUR_ACCOUNT scripts/aoraki/train_e2e.slurm" >&2
fi

# Threading hygiene
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Use project-space caches to avoid $HOME quotas
AORAKI_BASE="${AORAKI_BASE:-/projects/sciences/computing/liyi5784}"
AORAKI_CACHE_DIR="${AORAKI_CACHE_DIR:-$AORAKI_BASE/.cache}"
export HF_HOME="$AORAKI_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" logs

# Load conda and activate project-local env (see scripts/aoraki/create_env.sh)
MINIFORGE_DIR="$AORAKI_BASE/miniforge3"
if [[ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
  echo "[ERROR] Miniforge not found at $MINIFORGE_DIR. Run: bash scripts/aoraki/create_env.sh on the login node." >&2
  exit 1
fi
export PYTHONNOUSERSITE=1

ENV_PREFIX="${ENV_PREFIX:-$AORAKI_BASE/.conda/aoraki-verlan-e2e}"
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "[ERROR] Conda env not found at $ENV_PREFIX. Run: bash scripts/aoraki/create_env.sh" >&2
  exit 2
fi
conda activate "$ENV_PREFIX"

# Show GPU info for provenance
echo "---- GPU info ----"
nvidia-smi || true
echo "------------------"

# Reasonable defaults; override at sbatch submission with e.g. --export
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_LEN="${MAX_LEN:-128}"
LR="${LR:-2e-5}"

set -x
python -m src.detect.detect_train_lr_e2e \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LEN" \
  --lr "$LR"
set +x

echo "[OK] Training finished. Artifacts under models/detect/latest/lr_e2e"
