# -*- coding: utf-8 -*-
"""
Evaluate thresholds on validation split for the current detector
- Encoder: Salesforce/SFR-Embedding-Mistral (4-bit, BF16)
- Classifier: verlan-detector/lr_head.joblib
- Rebuilds train/val/test with SEED=42 to match training split
- Supports optional score calibration (temperature, Platt, isotonic)
- Saves CSV with metrics for each threshold, and prints two recommended thresholds:
  (a) best score by selected metric; (b) max recall with precision >= target

Outputs:
  verlan-detector/threshold_eval.csv

Usage:
  python EvaluateThreshold.py
  python EvaluateThreshold.py --tmin 0.05 --tmax 0.95 --tstep 0.02 --prec_target 0.85
"""

import os, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import joblib
from calibration import (
    temperature_scale,
    platt_scale,
    isotonic_calibration,
)

MODEL_ID   = "Salesforce/SFR-Embedding-Mistral"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR    = PROJECT_ROOT / "data" / "raw"
MODEL_DIR  = PROJECT_ROOT / "verlan-detector"
HEAD_PATH  = MODEL_DIR / "lr_head.joblib"
SEED       = 42
DEF_BATCH  = 64
DEF_MAXLEN = 512

os.makedirs(MODEL_DIR, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def load_data():
    """
    Build the same splits as training. If 'label' doesn't exist,
    auto-label with exact lexicon match to reduce noise.
    """
    sent_path = RAW_DIR / "Sentences.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    sent_df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    if "label" not in sent_df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def is_verlan_sentence(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        sent_df["label"] = sent_df["text"].apply(is_verlan_sentence)

    train_df, test_df = train_test_split(
        sent_df, test_size=0.15, stratify=sent_df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def load_encoder():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = AutoModel.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        )
    except Exception:
        enc = AutoModel.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    enc.eval()
    return tok, enc

@torch.inference_mode()
def embed_texts(texts, tok, enc, max_len=DEF_MAXLEN, batch_size=DEF_BATCH):
    device = next(enc.parameters()).device
    out = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [str(x) for x in texts[i:i+batch_size]]
        inputs = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hs = enc(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
            denom = mask.sum(1).clamp(min=1)
            pooled = (hs * mask).sum(1) / denom
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.detach().float().cpu().numpy())
        print(f"Embedded {min(i+batch_size, n)}/{n}")
    return np.vstack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.05)
    ap.add_argument("--prec_target", type=float, default=0.80)
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--max_len", type=int, default=DEF_MAXLEN)
    ap.add_argument(
        "--calibration",
        choices=["none", "temperature", "platt", "isotonic"],
        default="none",
        help="Optional score calibration method.",
    )
    ap.add_argument(
        "--opt_metric",
        choices=["f1", "youden"],
        default="f1",
        help="Metric for selecting the best threshold.",
    )
    args = ap.parse_args()

    train_df, val_df, _ = load_data()
    print(f"VAL size: {len(val_df)}  (pos={val_df['label'].sum()}, neg={len(val_df)-val_df['label'].sum()})")

    tok, enc = load_encoder()
    clf = joblib.load(HEAD_PATH)

    X_val = embed_texts(val_df["text"].tolist(), tok, enc, args.max_len, args.batch_size)
    y_val = val_df["label"].to_numpy()
    p1 = clf.predict_proba(X_val)[:, 1]

    # Optional calibration on validation scores
    if args.calibration == "temperature":
        p1, temp = temperature_scale(p1, y_val)
        print(f"Applied temperature scaling with T={temp:.4f}")
    elif args.calibration == "platt":
        p1 = platt_scale(p1, y_val)
        print("Applied Platt scaling")
    elif args.calibration == "isotonic":
        p1 = isotonic_calibration(p1, y_val)
        print("Applied isotonic calibration")

    # Global, threshold-free scores
    try:
        roc = roc_auc_score(y_val, p1)
    except Exception:
        roc = float("nan")
    ap_score = average_precision_score(y_val, p1)
    print(f"ROC-AUC = {roc:.4f}   PR-AUC(AP) = {ap_score:.4f}")

    # Threshold sweep
    thresholds = np.arange(args.tmin, args.tmax + 1e-9, args.tstep)
    rows = []
    for th in thresholds:
        pred = (p1 >= th).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, pred, average=None, labels=[0, 1]
        )
        p_pos, r_pos, f_pos = float(prec[1]), float(rec[1]), float(f1[1])
        acc = (pred == y_val).mean()
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        rows.append(
            {
                "threshold": th,
                "precision_1": p_pos,
                "recall_1": r_pos,
                "f1_1": f_pos,
                "youden": tpr - fpr,
                "accuracy": acc,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = "verlan-detector/threshold_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Recommendations
    metric_col = "f1_1" if args.opt_metric == "f1" else "youden"
    best_row = df.loc[df[metric_col].idxmax()]

    cand = df[df["precision_1"] >= args.prec_target]
    if len(cand):
        best_prec_row = cand.loc[cand["recall_1"].idxmax()]
    else:
        # If no threshold reaches the target precision, choose the one with the highest precision
        best_prec_row = df.loc[df["precision_1"].idxmax()]

    print("\nRecommended thresholds:")
    print(
        f"- Best {args.opt_metric}: threshold={best_row['threshold']:.2f}  "
        f"P={best_row['precision_1']:.3f} R={best_row['recall_1']:.3f} "
        f"F1={best_row['f1_1']:.3f} Y={best_row['youden']:.3f}  "
        f"Acc={best_row['accuracy']:.3f}  TP/FP/TN/FN={int(best_row['tp'])}/{int(best_row['fp'])}/{int(best_row['tn'])}/{int(best_row['fn'])}"
    )
    print(
        f"- Max R with P≥{args.prec_target:.2f}: threshold={best_prec_row['threshold']:.2f}  "
        f"P={best_prec_row['precision_1']:.3f} R={best_prec_row['recall_1']:.3f} F1={best_prec_row['f1_1']:.3f}  "
        f"Acc={best_prec_row['accuracy']:.3f}  TP/FP/TN/FN={int(best_prec_row['tp'])}/{int(best_prec_row['fp'])}/{int(best_prec_row['tn'])}/{int(best_prec_row['fn'])}"
    )
    print("\nTip: Use detect_infer.py with --threshold <value above>.")

if __name__ == "__main__":
    main()
