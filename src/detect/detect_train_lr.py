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

import argparse
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from src.detect.data_utils import (
    PROJECT_ROOT,
    load_slang_test_set,
    load_verlan_dataset,
)
from src.detect.eval_utils import confusion_from_arrays, save_predictions

# ------------------------ Tunable parameters ------------------------
MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42

# Default batch size and length (can be overridden via CLI)
DEF_BATCH = 32
DEF_MAXLEN = 512

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    model.eval() # freeze
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
    ap = argparse.ArgumentParser(description="Frozen encoder + logistic regression baseline")
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH, help="Batch size for embedding extraction")
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN, help="Maximum input token length")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed for data splits and embedding order")
    ap.add_argument("--run_id", type=str, default="", help="Optional identifier for the run output directory")
    ap.add_argument("--out_dir", type=str, default="", help="Optional custom output directory (absolute or relative to project root)")
    args = ap.parse_args()

    set_seed(args.seed)

    train_df, val_df, test_df = load_verlan_dataset(args.seed)
    slang_df = load_slang_test_set()
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
    # Use explicit 0.5 threshold for clarity
    yv_prob = clf.predict_proba(X_val)[:, 1]
    yv_pred = (yv_prob >= 0.5).astype(int)
    val_f1 = float(f1_score(y_val, yv_pred, zero_division=0))
    val_acc = float(accuracy_score(y_val, yv_pred))
    print(classification_report(y_val, yv_pred, digits=3))
    print("Val Accuracy@0.5:", val_acc)

    print("Embedding test set …")
    X_test = embed_texts(test_df["text"], tok, model, args.batch_size, args.max_length)
    y_test = test_df["label"].values
    yp_prob = clf.predict_proba(X_test)[:, 1]
    yp = (yp_prob >= 0.5).astype(int)
    test_f1 = float(f1_score(y_test, yp, zero_division=0))
    test_acc = float(accuracy_score(y_test, yp))
    test_conf = confusion_from_arrays(y_test, yp)
    print("Test F1@0.5:", test_f1)
    print("Test Accuracy@0.5:", test_acc)

    print("Embedding slang test set …")
    X_slang = embed_texts(slang_df["text"], tok, model, args.batch_size, args.max_length)
    y_slang = slang_df["label"].astype(int).values
    slang_prob = clf.predict_proba(X_slang)[:, 1]
    slang_pred = (slang_prob >= 0.5).astype(int)
    slang_conf = confusion_from_arrays(y_slang, slang_pred)
    slang_acc = float(accuracy_score(y_slang, slang_pred)) if len(y_slang) else 0.0
    slang_f1 = float(f1_score(y_slang, slang_pred, zero_division=0)) if len(np.unique(y_slang)) > 1 else 0.0
    print("Slang test Accuracy@0.5:", slang_acc)
    print("Slang test F1@0.5:", slang_f1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        tag = (args.run_id.strip() or f"seed-{args.seed}")
        out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "lr_frozen" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    head_path = out_dir / "lr_head.joblib"
    test_pred_path = out_dir / "test_predictions.csv"
    slang_pred_path = out_dir / "slang_predictions.csv"

    joblib.dump(clf, head_path)
    save_predictions(test_df, yp_prob, yp, test_pred_path)
    save_predictions(slang_df, slang_prob, slang_pred, slang_pred_path)

    meta = {
        "model_id": MODEL_ID,
        "encoder_frozen": True,
        "head": "LogisticRegression",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "run_id": args.run_id,
        "out_dir": str(out_dir.relative_to(PROJECT_ROOT) if str(out_dir).startswith(str(PROJECT_ROOT)) else out_dir),
        "head_params": {
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "metrics": {
            "val_f1@0.5": val_f1,
            "val_acc@0.5": val_acc,
            "test_f1@0.5": test_f1,
            "test_acc@0.5": test_acc,
            "test_confusion@0.5": test_conf,
            "slang_test_set@0.5": {
                **slang_conf,
                "accuracy": slang_acc,
                "f1": slang_f1,
            },
        },
        "artifacts": {
            "classifier": str(head_path.relative_to(PROJECT_ROOT) if str(head_path).startswith(str(PROJECT_ROOT)) else head_path),
            "test_predictions": str(test_pred_path.relative_to(PROJECT_ROOT) if str(test_pred_path).startswith(str(PROJECT_ROOT)) else test_pred_path),
            "slang_predictions": str(slang_pred_path.relative_to(PROJECT_ROOT) if str(slang_pred_path).startswith(str(PROJECT_ROOT)) else slang_pred_path),
        },
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detect model saved to {head_path}")

if __name__ == "__main__":
    main()
