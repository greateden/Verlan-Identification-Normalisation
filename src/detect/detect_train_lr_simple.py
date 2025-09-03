# -*- coding: utf-8 -*-
"""
Simplified Verlan sentence detector (no mean pooling, no L2 norm, no calibration).
Embeddings: CLS token from Salesforce/SFR-Embedding-Mistral (4-bit).
Classifier: LogisticRegression (balanced).
Decision threshold fixed at 0.5.
"""

import os, argparse, random, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42
DEF_BATCH = 32
DEF_MAXLEN = 512

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_data():
    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    for p in (sent_path, gaz_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    sent_df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    if "label" not in sent_df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower())
        def has_verlan(s):
            return int(any(t in vset for t in str(s).lower().split()))
        sent_df["label"] = sent_df["text"].apply(has_verlan)
    train_df, test_df = train_test_split(
        sent_df, test_size=0.15, stratify=sent_df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def load_encoder():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
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
def embed_texts(texts, tok, model, batch_size=DEF_BATCH, max_len=DEF_MAXLEN):
    """
    Return CLS token embeddings only (no mean pooling, no normalization).
    """
    device = next(model.parameters()).device
    out_list = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts.iloc[i:i+batch_size].astype(str).tolist()
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hs = model(**enc).last_hidden_state  # [B, T, D]
            cls = hs[:, 0, :].float().cpu().numpy()  # CLS token
        out_list.append(cls)
        print(f"Embedded {i + len(batch)}/{total}")
    return np.vstack(out_list)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "models" / "detect" / "latest"))
    args = ap.parse_args()

    train_df, val_df, test_df = load_data()
    tok, model = load_encoder()

    X_train = embed_texts(train_df["text"], tok, model, args.batch_size, args.max_length)
    y_train = train_df["label"].values

    X_val = embed_texts(val_df["text"], tok, model, args.batch_size, args.max_length)
    y_val = val_df["label"].values

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        verbose=0,
    )
    clf.fit(X_train, y_train)

    # Fixed threshold = 0.5
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)
    print("Validation (threshold=0.5):")
    print(classification_report(y_val, val_pred, digits=3))
    try:
        print("Val ROC AUC:", roc_auc_score(y_val, val_probs))
    except Exception:
        pass

    X_test = embed_texts(test_df["text"], tok, model, args.batch_size, args.max_length)
    y_test = test_df["label"].values
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= 0.5).astype(int)
    print("Test (threshold=0.5):")
    print(classification_report(y_test, test_pred, digits=3))
    try:
        print("Test ROC AUC:", roc_auc_score(y_test, test_probs))
    except Exception:
        pass
    print("Test F1:", f1_score(y_test, test_pred))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "lr_head.joblib")
    print(f"Saved classifier to {out_dir/'lr_head.joblib'}")

if __name__ == "__main__":
    main()