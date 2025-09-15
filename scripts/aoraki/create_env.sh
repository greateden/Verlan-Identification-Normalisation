#!/usr/bin/env bash
# Create a project-local conda env suitable for E2E training on Aoraki.

set -euo pipefail

if [[ ! -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  cat >&2 <<'EOF'
[ERROR] Miniforge not found at ~/miniforge3
Install it on Aoraki login node:
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -u
Then re-run this script.
EOF
  exit 1
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"

ENV_PREFIX="$PWD/.conda/aoraki-verlan-e2e"
ENV_YML="envs/aoraki-verlan-e2e.yml"

mkdir -p "$(dirname "$ENV_PREFIX")"

echo "Creating env at: $ENV_PREFIX"
if command -v mamba >/dev/null 2>&1; then
  conda activate base
  mamba env create -p "$ENV_PREFIX" -f "$ENV_YML"
else
  conda activate base
  conda env create -p "$ENV_PREFIX" -f "$ENV_YML"
fi

echo "[OK] Env created. Activate with: conda activate $ENV_PREFIX"

