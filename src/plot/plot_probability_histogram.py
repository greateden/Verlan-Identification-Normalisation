
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot probability histogram(s) for Verlan detection results.

This script reads a CSV with at least two columns: a probability column
in [0,1] (probability of being Verlan), and a label/target column (binary or
categorical). It draws overlapping histograms for the two classes.

Examples
--------
Basic usage with auto-detected columns:
    

Specify explicit column names:
    python src/plot/plot_probability_histogram.py \
        --csv file.csv --prob-col proba --label-col label

Dependencies
------------
- python>=3.9, pandas>=2.0, numpy>=1.25, matplotlib>=3.7
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def autodetect_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    prob_like = [c for c in df.columns if any(k in c.lower() for k in [
        "prob", "proba", "score", "logit", "p_verlan", "verlan_prob"
    ])]

    cand_probs = []
    for c in (prob_like or list(df.columns)):
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.9:
            finite = s[np.isfinite(s)]
            if len(finite) and finite.min() >= -0.05 and finite.max() <= 1.05:
                cand_probs.append(c)

    prob_col = None
    for key in ["p_verlan", "verlan_prob", "prob_verlan", "proba_verlan",
                "probability_verlan", "score_verlan"]:
        for c in cand_probs:
            if key in c.lower():
                prob_col = c
                break
        if prob_col:
            break
    if prob_col is None and cand_probs:
        prob_col = cand_probs[0]

    label_candidates = [c for c in df.columns if any(k in c.lower() for k in [
        "label", "target", "gold", "y_true", "true", "is_verlan", "class"
    ])]
    label_col = label_candidates[0] if label_candidates else None

    if label_col is None:
        for c in df.columns:
            if df[c].dtype == "bool":
                label_col = c
                break
    if label_col is None:
        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.number) and df[c].nunique(dropna=True) <= 5:
                label_col = c
                break

    return prob_col, label_col

def to_label_name(v) -> str:
    s = str(v).strip().lower()
    if s in {"1", "true", "verlan", "v"}:
        return "Verlan"
    if s in {"0", "false", "standard", "std", "french", "non-verlan", "nonverlan", "nv"}:
        return "Standard French"
    try:
        x = float(s)
        return "Verlan" if x >= 0.5 else "Standard French"
    except:
        return s.title()

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Draw overlapping histograms of Verlan vs Standard probabilities.")
    ap.add_argument("--csv", "-i", required=True, help="Path to CSV with probabilities and labels")
    ap.add_argument("--out", "-o", default="prob_dist.png", help="Output image path (PNG, SVG, etc.)")
    ap.add_argument("--prob-col", default=None, help="Name of probability column (0..1)")
    ap.add_argument("--label-col", default=None, help="Name of label/target column")
    ap.add_argument("--bins", type=int, default=20, help="Number of histogram bins (default: 20)")
    ap.add_argument("--title", default="Probability Distribution for Verlan vs Standard French",
                    help="Custom plot title")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)

    prob_col = args.prob_col
    label_col = args.label_col
    if prob_col is None or label_col is None:
        auto_prob, auto_label = autodetect_cols(df)
        prob_col = prob_col or auto_prob
        label_col = label_col or auto_label

    if prob_col is None or label_col is None:
        raise SystemExit("Could not detect columns. Use --prob-col and --label-col.")

    prob = pd.to_numeric(df[prob_col], errors="coerce").clip(0, 1)
    labels = df[label_col].map(to_label_name)

    mask = prob.notna() & labels.notna()
    prob = prob[mask]
    labels = labels[mask]

    if not ((labels == "Verlan").any() and (labels == "Standard French").any()):
        top2 = labels.value_counts().index[:2].tolist()
        verlan_name = top2[0]
        std_name = top2[1] if len(top2) > 1 else "Other"
    else:
        verlan_name, std_name = "Verlan", "Standard French"

    verlan_probs = prob[labels == verlan_name]
    std_probs = prob[labels == std_name]

    bins = np.linspace(0, 1, args.bins + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(verlan_probs, bins=bins, alpha=0.6, label=verlan_name, color="tab:red")
    ax.hist(std_probs,    bins=bins, alpha=0.6, label=std_name,    color="tab:blue")

    ax.set_title(args.title, fontsize=20, pad=12)
    ax.set_xlabel("Probability of being Verlan")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
