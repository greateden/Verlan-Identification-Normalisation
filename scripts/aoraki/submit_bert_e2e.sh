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
  TARGET="aoraki_gpu_L40"
fi

echo "→ Submitting to partition: $TARGET"

JOBID=$(sbatch \
  --parsable \
  --partition="$TARGET" \
  --chdir="$REPO_ROOT" \
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
