#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate slang test-set confusion matrices across experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.detect.data_utils import PROJECT_ROOT
from src.detect.eval_utils import confusion_from_arrays


EXPERIMENT_PATTERNS: Dict[str, List[Path]] = {
    "lr_frozen": [Path("lr_frozen")],
    "lr_e2e": [Path("lr_e2e")],
    "bert_frozen": [Path("bert_head") / "frozen"],
    "bert_e2e": [Path("bert_head") / "e2e"],
    "mistral_zeroshot": [Path("mistral_zeroshot")],
}


def locate_prediction_files(root: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for name, subpaths in EXPERIMENT_PATTERNS.items():
        collected: List[Path] = []
        for sub in subpaths:
            base = root / sub
            if not base.exists():
                continue
            collected.extend(sorted(base.glob("seed-*/slang_predictions.csv")))
        if collected:
            files[name] = collected
    return files


def load_labels(df: pd.DataFrame) -> np.ndarray:
    if "label" in df.columns:
        return df["label"].astype(int).to_numpy()
    if "verlan_label" in df.columns:
        return df["verlan_label"].astype(int).to_numpy()
    raise ValueError("Could not find label column in predictions dataframe.")


def summarize_runs(file_map: Dict[str, List[Path]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for exp, files in file_map.items():
        for csv_path in files:
            df = pd.read_csv(csv_path)
            y_true = load_labels(df)
            y_pred = df["pred_label"].astype(int).to_numpy()
            conf = confusion_from_arrays(y_true, y_pred)
            accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
            if len(np.unique(y_true)) > 1:
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
            else:
                f1 = 0.0
            rows.append({
                "experiment": exp,
                "run_dir": str(csv_path.parent),
                "num_samples": int(len(y_true)),
                "tn": conf["tn"],
                "fp": conf["fp"],
                "fn": conf["fn"],
                "tp": conf["tp"],
                "accuracy": accuracy,
                "f1": f1,
            })
    return rows


def aggregate_by_experiment(run_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    agg = defaultdict(lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "num_samples": 0, "runs": 0})
    for row in run_rows:
        exp = row["experiment"]
        agg_row = agg[exp]
        agg_row["tn"] += int(row["tn"])
        agg_row["fp"] += int(row["fp"])
        agg_row["fn"] += int(row["fn"])
        agg_row["tp"] += int(row["tp"])
        agg_row["num_samples"] += int(row["num_samples"])
        agg_row["runs"] += 1
    summary = []
    for exp, counts in agg.items():
        tn = counts["tn"]
        fp = counts["fp"]
        fn = counts["fn"]
        tp = counts["tp"]
        total = counts["num_samples"] if counts["num_samples"] else tn + fp + fn + tp
        accuracy = (tn + tp) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        summary.append({
            "experiment": exp,
            "runs": counts["runs"],
            "num_samples": total,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": accuracy,
            "f1": f1,
        })
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble confusion matrices on the slang-only test set.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "detect" / "latest",
        help="Directory containing experiment subfolders with slang_predictions.csv files.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "slang_confusion",
        help="Destination for CSV/JSON summaries.",
    )
    args = ap.parse_args()

    file_map = locate_prediction_files(args.input_root)
    if not file_map:
        raise SystemExit(f"No slang prediction files found under {args.input_root}")

    run_rows = summarize_runs(file_map)
    agg_rows = aggregate_by_experiment(run_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_run_csv = args.output_dir / "slang_confusion_by_run.csv"
    by_exp_csv = args.output_dir / "slang_confusion_by_experiment.csv"
    by_run_json = args.output_dir / "slang_confusion_by_run.json"
    by_exp_json = args.output_dir / "slang_confusion_by_experiment.json"

    pd.DataFrame(run_rows).to_csv(by_run_csv, index=False)
    pd.DataFrame(agg_rows).to_csv(by_exp_csv, index=False)

    by_run_json.write_text(json.dumps(run_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    by_exp_json.write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote per-run summary to {by_run_csv}")
    print(f"[OK] Wrote per-experiment summary to {by_exp_csv}")


if __name__ == "__main__":
    main()
