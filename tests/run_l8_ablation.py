#!/usr/bin/env python
"""Compatibility wrapper for the L8 ablation script.

The implementation now lives in `src/experiments/run_l8_ablation.py`.
This wrapper keeps the original entrypoint working:
  python tests/run_l8_ablation.py [args]
"""
from pathlib import Path
import sys

# Ensure `src` is importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experiments.run_l8_ablation import main  # type: ignore

if __name__ == "__main__":
    sys.exit(main())
