#!/usr/bin/env bash
# Wrapper that picks the best available GPU partition and tails logs.

set -euo pipefail

mkdir -p logs

PARTS=(aoraki_gpu_H100 aoraki_gpu_A100_80GB aoraki_gpu_A100_40GB aoraki_gpu_L40S aoraki_gpu_L40 aoraki_gpu_A6000 aoraki_gpu_RTX3090 aoraki_gpu_L4)

for P in "${PARTS[@]}"; do
  idle=$(sinfo -h -p "$P" -t idle | awk '{print $4}' | paste -sd+ - | bc)
  if [[ "${idle:-0}" -ge 1 ]]; then
    TARGET=$P
    break
  fi
done

: ${TARGET:=aoraki_gpu}

echo "→ Submitting to fastest idle partition: $TARGET"
JOBID=$(sbatch --parsable --partition="$TARGET" --gpus-per-node=1 scripts/train_detect.slurm)
LOGFILE="logs/verlan-embed-${JOBID}.out"
echo "Tailing $LOGFILE"
while [ ! -f "$LOGFILE" ]; do sleep 1; done
tail -n0 -f "$LOGFILE"
