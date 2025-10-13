#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return TN/FP/FN/TP counts even if a class is absent."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (1, 1):
        tn = int(cm[0, 0])
        fp = fn = tp = 0
    else:
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def save_predictions(
    base_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    out_path: Path,
    extra: Mapping[str, Sequence] | None = None,
) -> None:
    """Persist probabilities/predictions alongside the source dataframe."""
    if len(base_df) != len(probs) or len(probs) != len(preds):
        raise ValueError("Length mismatch when saving predictions.")
    out_df = base_df.copy()
    out_df["prob_1"] = probs
    out_df["pred_label"] = preds
    if extra:
        for key, values in extra.items():
            if len(values) != len(base_df):
                raise ValueError(f"Length mismatch for column '{key}'.")
            out_df[key] = values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


__all__ = ["confusion_from_arrays", "save_predictions"]
