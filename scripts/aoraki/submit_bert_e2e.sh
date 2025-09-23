#!/usr/bin/env bash
# Submit BERT-head E2E training to Aoraki and tail logs.

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

if [[ -n "${AORAKI_PARTITION:-}" ]]; then
  TARGET="$AORAKI_PARTITION"
else
  CANDIDATES=(
    aoraki_gpu_H100
    aoraki_gpu_A100_80GB
    aoraki_gpu_A100_40GB
    aoraki_gpu_L40
    aoraki_gpu
  )
  TARGET="aoraki_gpu"
  for P in "${CANDIDATES[@]}"; do
    idle=$(sinfo -h -p "$P" -t idle | awk '{print $4}' | paste -sd+ - 2>/dev/null | bc 2>/dev/null || echo 0)
    if [[ "${idle:-0}" -ge 1 ]]; then
      TARGET="$P"
      break
    fi
  done
fi

echo "→ Submitting to partition: $TARGET"

JOBID=$(sbatch \
  --parsable \
  --partition="$TARGET" \
  --export=ALL \
  scripts/aoraki/train_bert_e2e.slurm)

echo "Submitted as job $JOBID"
LOGOUT="$LOG_DIR/verlan-bert-e2e-${JOBID}.out"
LOGERR="$LOG_DIR/verlan-bert-e2e-${JOBID}.err"
echo "Tailing $LOGOUT and $LOGERR"
while [[ ! -f "$LOGOUT" || ! -f "$LOGERR" ]]; do
  sleep 1
done
tail -n0 -f "$LOGOUT" "$LOGERR"
