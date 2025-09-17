#!/usr/bin/env python3
"""Run detect_train_lr.py 20 times and capture output to a timestamped log."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

RUN_COUNT = 20
TARGET_SCRIPT = Path("src/detect/detect_train_lr.py")

def main() -> int:
    if not TARGET_SCRIPT.is_file():
        sys.stderr.write(f"Could not find {TARGET_SCRIPT}\n")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{TARGET_SCRIPT.stem}_{timestamp}.out"
    log_path = Path(log_name)

    with log_path.open("w", encoding="utf-8") as log_file:
        header = f"Running {TARGET_SCRIPT} {RUN_COUNT} times\nSaved output to {log_path}\n"
        sys.stdout.write(header)
        log_file.write(header)
        log_file.flush()

        for run_index in range(1, RUN_COUNT + 1):
            seed = run_index
            run_header = f"\n--- Run {run_index}/{RUN_COUNT} (seed={seed}) ---\n"
            sys.stdout.write(run_header)
            log_file.write(run_header)
            log_file.flush()

            seed_script = (
                "import random\n"
                "import numpy as np\n"
                "import torch\n"
                "from src.detect import detect_train_lr as module\n"
                f"seed = {seed}\n"
                "module.SEED = seed\n"
                "random.seed(seed)\n"
                "np.random.seed(seed)\n"
                "torch.manual_seed(seed)\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.manual_seed_all(seed)\n"
                "module.main()\n"
            )

            command = [sys.executable, "-c", seed_script]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return_code = process.wait()

            status_line = f"Run {run_index} exited with status {return_code}\n"
            sys.stdout.write(status_line)
            log_file.write(status_line)
            log_file.flush()

            if return_code != 0:
                error_line = "Aborting remaining runs due to non-zero exit status.\n"
                sys.stdout.write(error_line)
                log_file.write(error_line)
                return return_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
