#!/usr/bin/env bash
# Submit E2E training to Aoraki with a suitable GPU and tail logs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="$REPO_ROOT/logs"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[ERROR] sbatch not found. Run on Aoraki login node." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

pushd "$REPO_ROOT" >/dev/null
trap 'popd >/dev/null' EXIT

# Allow user to force a partition via AORAKI_PARTITION; else pick best idle.
if [[ -n "${AORAKI_PARTITION:-}" ]]; then
  TARGET="$AORAKI_PARTITION"
else
  # Prefer faster/bigger first, based on RTIS docs.
  CANDIDATES=(
    aoraki_gpu_L40
    aoraki_gpu_H100
    aoraki_gpu_A100_80GB
    aoraki_gpu_A100_40GB
    aoraki_gpu
  )
  TARGET="aoraki_gpu_L40"
  for P in "${CANDIDATES[@]}"; do
    # Count idle nodes in partition P
    idle=$(sinfo -h -p "$P" -t idle | awk '{print $4}' | paste -sd+ - 2>/dev/null | bc 2>/dev/null || echo 0)
    if [[ "${idle:-0}" -ge 1 ]]; then
      TARGET="$P"; break
    fi
  done
fi

echo "→ Submitting to partition: $TARGET"

# Allow tuning via environment; e.g. EPOCHS=3 BATCH_SIZE=8 MAX_LEN=128 LR=2e-5
JOBID=$(sbatch \
  --parsable \
  --partition="$TARGET" \
  --chdir="$REPO_ROOT" \
  --export=ALL \
  scripts/aoraki/train_e2e.slurm)

echo "Submitted as job $JOBID"
LOGOUT="$LOG_DIR/verlan-e2e-${JOBID}.out"
LOGERR="$LOG_DIR/verlan-e2e-${JOBID}.err"
echo "Tailing $LOGOUT and $LOGERR"
while [[ ! -f "$LOGOUT" || ! -f "$LOGERR" ]]; do sleep 1; done
tail -n0 -f "$LOGOUT" "$LOGERR"
