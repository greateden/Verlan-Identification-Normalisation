#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for loading Verlan detection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_label_column(sent_df: pd.DataFrame, lex_df: pd.DataFrame) -> pd.DataFrame:
    if "label" in sent_df.columns:
        return sent_df
    vset = set(lex_df["verlan_form"].dropna().astype(str).str.lower().tolist())

    def has_verlan(text: str) -> int:
        toks = str(text).lower().split()
        return int(any(t in vset for t in toks))

    sent_df = sent_df.copy()
    sent_df["label"] = sent_df["text"].apply(has_verlan)
    return sent_df


def load_verlan_dataset(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    missing = [p for p in (sent_path, gaz_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Could not locate required raw data files:\n"
            + "\n".join(f" - {p}" for p in missing)
        )

    df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    df = _ensure_label_column(df, lex)

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_slang_test_set() -> pd.DataFrame:
    slang_path = PROCESSED_DIR / "slang_test_set.csv"
    if not slang_path.exists():
        raise FileNotFoundError(f"Missing slang test set: {slang_path}")
    df = pd.read_csv(slang_path)
    if "verlan_label" not in df.columns:
        raise ValueError("Expected column 'verlan_label' in slang test set.")
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df["verlan_label"].astype(int)
    return df


__all__ = [
    "PROJECT_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "load_verlan_dataset",
    "load_slang_test_set",
]
