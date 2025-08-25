#!/usr/bin/env bash
#SBATCH --job-name=verlan-embed
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Detect GPU resources
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

echo "Detected GPU: $GPU_NAME (${GPU_MEM} MiB)"

if (( GPU_MEM <= 16000 )); then
  echo "Trashy GPU found, next time!"
  exit 0
fi

if (( GPU_MEM >= 80000 )); then
  QUANT=16
  BATCH=32
elif (( GPU_MEM >= 40000 )); then
  QUANT=8
  BATCH=16
else
  QUANT=4
  BATCH=8
fi

echo "Using $QUANT-bit quantisation with batch size $BATCH"
python src/detect_train_nn.py --batch "$BATCH" --quant "$QUANT"
