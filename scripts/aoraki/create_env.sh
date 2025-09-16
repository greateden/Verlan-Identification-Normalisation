#!/usr/bin/env bash
# Create a project-local conda env suitable for E2E training on Aoraki.

set -euo pipefail

# Where to install Miniforge and the env (defaults to project space, not $HOME)
AORAKI_BASE="${AORAKI_BASE:-/projects/sciences/computing/liyi5784}"
MINIFORGE_DIR="$AORAKI_BASE/miniforge3"

mkdir -p "$AORAKI_BASE"

# Ensure Miniforge exists under project space; install there if missing.
if [[ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
  echo "[INFO] Miniforge not found at $MINIFORGE_DIR — installing under project space..."
  INSTALLER="$AORAKI_BASE/Miniforge3-Linux-x86_64.sh"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  else
    echo "[ERROR] Neither wget nor curl is available to download Miniforge." >&2
    exit 1
  fi
  bash "$INSTALLER" -b -u -p "$MINIFORGE_DIR"
  rm -f "$INSTALLER"
  echo "[OK] Miniforge installed at $MINIFORGE_DIR"
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# Place the env under project space as well
ENV_PREFIX="$AORAKI_BASE/.conda/aoraki-verlan-e2e"
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
