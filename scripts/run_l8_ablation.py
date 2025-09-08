#!/usr/bin/env python
"""CLI entrypoint for the L8 ablation experiment.

Usage:
  python scripts/run_l8_ablation.py [args]

This imports the implementation from src/experiments/run_l8_ablation.py
so you can run it from a clean `scripts/` location.
"""
from pathlib import Path
import sys

# Ensure `src` is importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experiments.run_l8_ablation import main  # type: ignore

if __name__ == "__main__":
    raise SystemExit(main())

