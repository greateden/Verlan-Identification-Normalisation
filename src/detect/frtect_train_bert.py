# -*- coding: utf-8 -*-
"""
CamemBERT fine-tuning for sentence-level Verlan detection.

Pipeline alignment:
- Stratified split: Train 72.25%, Val 12.75%, Test 15%
- Ingestion + normalization: Unicode NFC, strip control chars, keep accents
- Tokenize once via CamemBERT tokenizer
- Branch 1 (diagnostic): UMAP on frozen encoder embeddings (optional)
- Branch 2 (training): Fine-tune CamemBERT + 1-logit head, sigmoid, t=0.5

Outputs (under models/detect):
- models/detect/YYYY-MM-DD/camembert/  (HuggingFace save_pretrained)
- models/detect/latest/camembert/      (updated copy of best model)
- metrics JSON alongside the model (val/test metrics)
- optional diagnostic plot at docs/results/frtect_umap.png

Usage examples:
  python -m src.detect.frtect_train_bert --epochs 3 --batch 16 --max_len 128 --umap
  python -m src.detect.frtect_train_bert --no_train --umap --umap_points 600
"""

import os
import random
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    CamembertModel,
    get_linear_schedule_with_warmup,
)

import unicodedata as ud


# ------------------------ Constants and defaults ------------------------
SEED = 42
CAM_ID = "camembert-base"
DEF_BATCH = 16
DEF_MAXLEN = 128
DEF_EPOCHS = 3
DEF_LR = 2e-5
DEF_WD = 0.01
DEF_WARMUP = 0.06  # ratio of total steps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_BASE = PROJECT_ROOT / "models" / "detect"
LATEST_DIR = MODEL_BASE / "latest"
DIAG_FPATH = PROJECT_ROOT / "docs" / "results" / "frtect_umap.png"


# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------ Utilities ------------------------
def normalize_nfc_strip_control(s: str) -> str:
    """Normalize to Unicode NFC and strip control/format characters.

    - Keeps accents (no ASCII folding)
    - Removes general categories Cc (control) and Cf (format)
    """
    if not isinstance(s, str):
        s = str(s)
    s = ud.normalize("NFC", s)
    return "".join(ch for ch in s if not ud.category(ch) in {"Cc", "Cf"})


def ensure_labels(df: pd.DataFrame, gaz_path: Path) -> pd.DataFrame:
    """Ensure a binary 'label' column exists; if missing, infer by exact lexicon match."""
    if "label" in df.columns:
        return df
    gaz = pd.read_excel(gaz_path)
    vset = set(gaz["verlan_form"].dropna().astype(str).str.lower().tolist())
    def has_verlan(s: str) -> int:
        toks = str(s).lower().split()
        return int(any(t in vset for t in toks))
    out = df.copy()
    out["label"] = out["text"].apply(has_verlan)
    return out


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Sentences.xlsx (+GazetteerEntries.xlsx for optional label) and build splits.

    Train 72.25%, Val 12.75%, Test 15% (stratified, SEED=42).
    """
    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    if not sent_path.exists() or not gaz_path.exists():
        raise FileNotFoundError(
            f"Missing input files under {RAW_DIR}. Expected Sentences.xlsx and GazetteerEntries.xlsx."
        )
    df = pd.read_excel(sent_path)
    df = ensure_labels(df, gaz_path)
    # Normalize text
    df["text"] = df["text"].apply(normalize_nfc_strip_control)
    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    print(
        f"Splits: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(pos counts: {train_df['label'].sum()}/{val_df['label'].sum()}/{test_df['label'].sum()})"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ------------------------ Dataset ------------------------
class VerlanDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CamembertTokenizer, max_len: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        s = self.texts[idx]
        y = float(self.labels[idx])
        enc = self.tok(
            s,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor([y], dtype=torch.float32),
        }


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ------------------------ Diagnostics: UMAP on frozen encoder ------------------------
def mean_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hs.dtype)
    denom = mask.sum(1).clamp(min=1)
    return (hs * mask).sum(1) / denom


@torch.inference_mode()
def compute_umap_plot(df: pd.DataFrame, tok: CamembertTokenizer, out_path: Path, max_len: int = 128, sample: int = 400):
    try:
        from umap import UMAP
    except Exception:
        print("[UMAP] umap-learn not installed. Skipping diagnostic plot. Install via: pip install umap-learn")
        return
    import matplotlib.pyplot as plt

    # Freeze encoder for diagnostics
    enc = CamembertModel.from_pretrained(CAM_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device).eval()

    df_ = df.sample(n=min(sample, len(df)), random_state=SEED)
    texts = df_["text"].astype(str).tolist()
    labels = df_["label"].to_numpy()

    embs = []
    bs = 64
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        encd = tok(
            batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(device)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hs = enc(**encd).last_hidden_state
                pooled = mean_pool(hs, encd["attention_mask"]).float()
        else:
            hs = enc(**encd).last_hidden_state
            pooled = mean_pool(hs, encd["attention_mask"]).float()
        embs.append(pooled.cpu().numpy())
    X = np.vstack(embs)

    reducer = UMAP(n_components=2, random_state=SEED)
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(7.2, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="coolwarm", s=16, edgecolors="none", alpha=0.85)
    plt.title("CamemBERT (frozen) – UMAP of sentence embeddings")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()
    print(f"[UMAP] Saved: {out_path}")


# ------------------------ Training ------------------------
@dataclass
class TrainConfig:
    batch: int = DEF_BATCH
    epochs: int = DEF_EPOCHS
    lr: float = DEF_LR
    weight_decay: float = DEF_WD
    warmup_ratio: float = DEF_WARMUP
    max_len: int = DEF_MAXLEN
    patience: int = 2


def evaluate_model(model: CamembertForSequenceClassification, loader: DataLoader, device: str):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            p = torch.sigmoid(out.logits).cpu().numpy().ravel()
            y = labels.cpu().numpy().ravel()
            probs.append(p); ys.append(y)
    p1 = np.concatenate(probs)
    y_true = np.concatenate(ys).astype(int)
    preds = (p1 >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, labels=[0, 1])
    try:
        roc = roc_auc_score(y_true, p1)
    except Exception:
        roc = float("nan")
    ap = average_precision_score(y_true, p1)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
        "accuracy": float((preds == y_true).mean()),
        "roc_auc": float(roc),
        "ap": float(ap),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train(cfg: TrainConfig, do_umap: bool = False, umap_points: int = 400, no_train: bool = False):
    # Data
    train_df, val_df, test_df = load_splits()
    tok = CamembertTokenizer.from_pretrained(CAM_ID)

    if do_umap:
        compute_umap_plot(train_df, tok, DIAG_FPATH, max_len=cfg.max_len, sample=umap_points)

    if no_train:
        return  # Only diagnostics

    # Datasets/loaders
    train_ds = VerlanDataset(train_df, tok, cfg.max_len)
    val_ds = VerlanDataset(val_df, tok, cfg.max_len)
    test_ds = VerlanDataset(test_df, tok, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, collate_fn=collate)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CamembertForSequenceClassification.from_pretrained(
        CAM_ID, num_labels=1, problem_type="single_label_classification"
    ).to(device)
    model.train()

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    warmup = int(cfg.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    # Loss
    bce = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = bce(out.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step(); sched.step()

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}: VAL {val_metrics}")

        if val_metrics["f1_1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1_1"]
            best_state = {"model": model.state_dict(), "val": val_metrics}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Final test metrics
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"TEST {test_metrics}")

    # Save
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = MODEL_BASE / today / "camembert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": best_state["val"] if best_state else {}, "test": test_metrics}, f, ensure_ascii=False, indent=2)
    print(f"Saved model + tokenizer to: {out_dir}")

    # Update latest copy
    latest_sub = LATEST_DIR / "camembert"
    latest_sub.mkdir(parents=True, exist_ok=True)
    # Copy by re-saving to latest (avoid shutil dir merge semantics)
    model.save_pretrained(latest_sub)
    tok.save_pretrained(latest_sub)
    with open(latest_sub / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": best_state["val"] if best_state else {}, "test": test_metrics}, f, ensure_ascii=False, indent=2)
    print(f"Updated latest at: {latest_sub}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--weight_decay", type=float, default=DEF_WD)
    ap.add_argument("--warmup_ratio", type=float, default=DEF_WARMUP)
    ap.add_argument("--max_len", type=int, default=DEF_MAXLEN)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--umap", action="store_true", help="Run diagnostic UMAP on frozen encoder")
    ap.add_argument("--umap_points", type=int, default=400)
    ap.add_argument("--no_train", action="store_true", help="Skip training; only run diagnostics if requested")
    args = ap.parse_args()

    cfg = TrainConfig(
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_len=args.max_len,
        patience=args.patience,
    )
    train(cfg, do_umap=args.umap, umap_points=args.umap_points, no_train=args.no_train)


if __name__ == "__main__":
    main()
