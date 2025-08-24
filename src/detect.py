# -*- coding: utf-8 -*-
"""
Sentence-level Verlan detector
- Encoder: Salesforce/SFR-Embedding-Mistral (4-bit inference, BF16 compute)
- Head: CPU LogisticRegression (class_weight balanced)
- Pooling: mean over valid tokens (attention mask), then L2-normalize
- Works on NVIDIA A4000 (16GB)

Recommended dependency versions:
- torch>=2.2
- transformers>=4.41
- bitsandbytes>=0.43
- scikit-learn>=1.3
- pandas, numpy, joblib, openpyxl
"""

import os, random, joblib, argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ------------------------ Tunable parameters ------------------------
MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42

# Default batch size and length (can be overridden via CLI)
DEF_BATCH = 32
DEF_MAXLEN = 512

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_data():
    print("Loading data …")
    """
    Load datasets from data/raw regardless of current working directory.
    """
    sent_path = RAW_DIR / "Sentences.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"

    # Friendly existence check and debug output (comment ratio >30%)
    missing = [p for p in [sent_path, gaz_path] if not p.exists()]
    if missing:
        msg = (
            "❌ Could not find the following required files:\n"
            + "\n".join(f" - {p}" for p in missing)
            + f"\n\nCurrent working directory (cwd) = {Path.cwd()}\n"
            f"PROJECT_ROOT resolved by detect.py = {PROJECT_ROOT}\n"
            "Please check the paths or run `python -m src.detect` from the project root."
        )
        raise FileNotFoundError(msg)

    # Actual file reading
    sent_df = pd.read_excel(sent_path)          # Sentences.xlsx
    lex  = pd.read_excel(gaz_path)           # GazetteerEntries.xlsx
    if "label" not in sent_df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        sent_df["label"] = sent_df["text"].apply(has_verlan)
    train_df, test_df = train_test_split(
        sent_df, test_size=0.15, stratify=sent_df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    print(f"Splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def load_encoder():
    # 4-bit inference + BF16 compute; use flash-attn2 if installed, otherwise fall back to SDPA
    bnb_cfg = BitsAndBytesConfig(
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
        model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    return tok, model

@torch.inference_mode()
def embed_texts(texts: pd.Series, tok, model, batch_size=DEF_BATCH, max_len=DEF_MAXLEN):
    """Mean-pool with attention mask, then L2-normalize. Entire pipeline runs on GPU."""
    device = next(model.parameters()).device
    embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        chunk = texts.iloc[i:i+batch_size].astype(str).tolist()
        enc = tok(
            chunk, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        ).to(device)

        # BF16 automatic mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**enc)
            H = out.last_hidden_state                         # [B, T, D] (GPU)
            mask = enc["attention_mask"].unsqueeze(-1)        # [B, T, 1] (GPU)
            mask = mask.to(H.dtype)
            denom = mask.sum(dim=1).clamp(min=1)              # [B, 1]
            pooled = (H * mask).sum(dim=1) / denom            # [B, D]
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        embs.append(pooled.detach().float().cpu().numpy())
        print(f"Embedded {i + len(chunk)}/{total} texts")
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    args = ap.parse_args()

    train_df, val_df, test_df = load_data()
    tok, model = load_encoder()

    print("Embedding train set …")
    X_train = embed_texts(train_df["text"], tok, model, args.batch_size, args.max_length)
    y_train = train_df["label"].values

    print("Embedding val set …")
    X_val = embed_texts(val_df["text"], tok, model, args.batch_size, args.max_length)
    y_val = val_df["label"].values

    print("Training classifier …")
    # class_weight balanced with higher max_iter; lbfgs is stable for medium-sized samples with continuous features
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        verbose=0,
    )
    clf.fit(X_train, y_train)

    print("Val results:")
    yv_pred = clf.predict(X_val)
    print(classification_report(y_val, yv_pred, digits=3))

    print("Embedding test set …")
    X_test = embed_texts(test_df["text"], tok, model, args.batch_size, args.max_length)
    y_test = test_df["label"].values
    yp = clf.predict(X_test)
    print("Test F1:", f1_score(y_test, yp))

    os.makedirs("verlan-detector", exist_ok=True)
    joblib.dump(clf, "verlan-detector/lr_head.joblib")
    print("Detect model saved.")

if __name__ == "__main__":
    main()
