#!/usr/bin/env bash
#SBATCH --job-name=verlan-bert-e2e
#SBATCH --partition=aoraki_gpu_L40
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
fi
cd "$REPO_ROOT"

if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
  echo "[WARN] No --account provided. Submit with: sbatch --account=YOUR_ACCOUNT scripts/aoraki/train_bert_e2e.slurm" >&2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
: "${MKL_INTERFACE_LAYER:=GNU,LP64}"
export MKL_INTERFACE_LAYER

DEFAULT_PROJECT_BASE=$(cd "$REPO_ROOT/.." && pwd)
AORAKI_BASE="${AORAKI_BASE:-$DEFAULT_PROJECT_BASE}"
AORAKI_CACHE_DIR="${AORAKI_CACHE_DIR:-$AORAKI_BASE/.cache}"
export HF_HOME="$AORAKI_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
: "${HF_HUB_ENABLE_HF_TRANSFER:=0}"
export HF_HUB_ENABLE_HF_TRANSFER
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" logs

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

echo "---- GPU info ----"
nvidia-smi || true
echo "------------------"

BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_LEN="${MAX_LEN:-128}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TRIALS="${TRIALS:-20}"
SEED_START="${SEED_START:-1}"

OUT_BASE="models/detect/latest/bert_head/e2e"
mkdir -p "$OUT_BASE"

set -x
for ((i=0; i<TRIALS; i++)); do
  SEED=$((SEED_START + i))
  RUN_ID="seed-${SEED}"
  python -m src.detect.detect_train_bert_e2e \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LEN" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --run_id "$RUN_ID"
done
set +x

python - <<'PY'
import json, csv
from pathlib import Path
base = Path('models/detect/latest/bert_head/e2e')
rows = []
for p in sorted(base.glob('seed-*/meta.json')):
    try:
        meta = json.loads(p.read_text())
        m = meta.get('metrics', {})
        test_conf = m.get('test_confusion@0.5', {})
        slang = m.get('slang_test_set@0.5', {})
        rows.append({
            'run_dir': str(p.parent),
            'seed': meta.get('seed'),
            'run_id': meta.get('run_id'),
            'val_best_f1@0.5': m.get('val_best_f1@0.5'),
            'test_f1@0.5': m.get('test_f1@0.5'),
            'test_acc@0.5': m.get('test_acc@0.5'),
            'test_fp@0.5': test_conf.get('fp'),
            'test_fn@0.5': test_conf.get('fn'),
            'slang_fp@0.5': slang.get('fp'),
            'slang_tn@0.5': slang.get('tn'),
            'slang_accuracy@0.5': slang.get('accuracy'),
        })
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")

rows.sort(key=lambda r: (r.get('seed') is None, r.get('seed')))

out_json = base / 'trials_summary.json'
out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

out_csv = base / 'trials_summary.csv'
with out_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed','run_id','run_dir','val_best_f1@0.5','test_f1@0.5','test_acc@0.5','test_fp@0.5','test_fn@0.5','slang_tn@0.5','slang_fp@0.5','slang_accuracy@0.5'])
    w.writeheader()
    w.writerows(rows)

print(f"[OK] Wrote {out_json} and {out_csv}")
PY

echo "[OK] Trials finished. Artifacts under models/detect/latest/bert_head/e2e; summaries saved."
