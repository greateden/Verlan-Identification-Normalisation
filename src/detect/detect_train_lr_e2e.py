#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment B: End-to-end fine-tuning with mean pooling + linear head

Purpose
- Compare to the frozen-encoder + CPU LogisticRegression baseline (Experiment A).
- If end-to-end fine-tuning significantly outperforms the frozen baseline, it suggests
  Mistral did not already encode verlan knowledge linearly; if performance is similar,
  it suggests pretraining already captured the relevant structure.

Model
- Encoder: Salesforce/SFR-Embedding-Mistral
- Head:    Linear( hidden_size -> 1 ) trained jointly with the encoder
- Pooling: Mean over valid tokens (attention mask), then L2-normalize
- Loss:    BCEWithLogitsLoss with optional positive-class weighting

Notes
- Full fine-tuning of a 7B encoder is memory-intensive. This script matches the
  baseline's quantization settings (4-bit NF4 + BF16 compute, device_map=auto).
  For practical fine-tuning, LoRA/QLoRA is recommended (not included here).

Usage (examples)
  # Default run (GPU if available)
  python -m src.detect.detect_train_lr_e2e --epochs 3 --batch_size 32 --max_length 512 --lr 2e-5

Dependencies
- torch>=2.2, transformers>=4.41, bitsandbytes>=0.43, scikit-learn, pandas, numpy, openpyxl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

# Default batch size and length (aligned with LR baseline)
DEF_BATCH = 32
DEF_MAXLEN = 512

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_encoder():
    """Match LR baseline: 4-bit NF4 + BF16 compute, device_map=auto, flash-attn hint."""
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
        enc = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        enc = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    return tok, enc


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = [str(t) for t in texts]
        self.labels = [int(x) for x in labels]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i: int):
        return {"text": self.texts[i], "label": float(self.labels[i])}


def make_collate(tokenizer: AutoTokenizer, max_length: int):
    def collate(samples: List[Dict[str, object]]):
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].long(),
            "attention_mask": enc["attention_mask"].long(),
            "label": torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
        }
    return collate


class E2ELinear(nn.Module):
    """Encoder + mean pooling + L2 + linear head → 1 logit"""
    def __init__(self, encoder: AutoModel):
        super().__init__()
        self.encoder = encoder
        hidden = int(getattr(self.encoder.config, "hidden_size", 1024))
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = out.last_hidden_state  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).to(H.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (H * mask).sum(dim=1) / denom
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        logits = self.head(pooled)  # [B, 1]
        return logits


@torch.no_grad()
def evaluate(model: E2ELinear, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    probs, gold = [], []
    start_device = next(model.encoder.parameters()).device
    for batch in loader:
        ids = batch["input_ids"].to(start_device)
        attn = batch["attention_mask"].to(start_device)
        yb = batch["label"]
        with torch.no_grad():
            logits = model(ids, attn)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.append(p)
        gold.append(yb.numpy().ravel())
    probs = np.concatenate(probs) if probs else np.zeros((0,))
    gold = np.concatenate(gold) if gold else np.zeros((0,))
    res = {"f1@0.5": 0.0, "acc@0.5": 0.0}
    if len(np.unique(gold)) > 1:
        res["f1@0.5"] = float(f1_score(gold, (probs >= 0.5).astype(int), zero_division=0))
        res["acc@0.5"] = float(accuracy_score(gold, (probs >= 0.5).astype(int)))
    # also return raw to enable threshold scan
    res["_probs"] = probs
    res["_gold"] = gold
    return res


def scan_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    # Deprecated: not used when evaluating strictly at threshold 0.5
    ts = np.linspace(0, 1, 501)
    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in ts]
    idx = int(np.argmax(f1s))
    return float(ts[idx]), float(f1s[idx])


def main():
    ap = argparse.ArgumentParser(description="Experiment B: end-to-end fine-tuning (mean-pool + linear head)")
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed for splits and training")
    ap.add_argument("--run_id", type=str, default="", help="Optional run identifier for output directory")
    ap.add_argument("--out_dir", type=str, default="", help="Optional custom output directory (absolute or relative to project root)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Re-seed with provided seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    train_df, val_df, test_df = load_verlan_dataset(args.seed)
    slang_df = load_slang_test_set()
    tok, enc = load_encoder()

    model = E2ELinear(enc)
    start_device = next(model.encoder.parameters()).device
    model.head.to(start_device)
    model.train()  # unfrozen encoder

    # Data
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].astype(int).tolist())
    val_ds = TextDataset(val_df["text"].tolist(), val_df["label"].astype(int).tolist())
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label"].astype(int).tolist())

    collate = make_collate(tok, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size * 2), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size * 2), shuffle=False, collate_fn=collate)

    # Class weighting (pos_weight) for BCE if imbalanced
    y_np = train_df["label"].astype(int).values
    n_pos = max(1, int((y_np == 1).sum()))
    n_neg = max(1, int((y_np == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    pos_weight = pos_weight.to(start_device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state = None
    best_t = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"]
            attn = batch["attention_mask"]
            yb = batch["label"]
            # Align inputs/labels to encoder's starting device (device_map="auto" compatible)
            ids = ids.to(start_device)
            attn = attn.to(start_device)
            yb = yb.to(start_device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(start_device.type == "cuda")):
                logits = model(ids, attn)
                loss = bce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch} train_loss={total_loss/ max(1,len(train_loader)):.4f}")

        # Validation @ fixed 0.5 threshold (mirrors LR baseline)
        val = evaluate(model, val_loader, device)
        yv_pred = (val["_probs"] >= 0.5).astype(int)
        print(classification_report(val["_gold"], yv_pred, digits=3))
        print(f"Val F1@0.5: {val['f1@0.5']:.3f}")
        print(f"Val Accuracy@0.5: {val['acc@0.5']:.3f}")
        f1_star = float(val["f1@0.5"]) if len(np.unique(val["_gold"])) > 1 else -1.0
        if f1_star > best_f1:
            best_f1 = f1_star
            # Save minimal state
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                "head": model.head.state_dict(),
            }

    # Test @ fixed 0.5 threshold
    test = evaluate(model, test_loader, device)
    test_probs = test["_probs"]
    test_gold = test["_gold"]
    test_preds = (test_probs >= 0.5).astype(int)
    test_f1 = float(f1_score(test_gold, test_preds, zero_division=0)) if len(np.unique(test_gold)) > 1 else 0.0
    test_acc_05 = float(accuracy_score(test_gold, test_preds)) if len(np.unique(test_gold)) > 1 else 0.0
    test_conf = confusion_from_arrays(test_gold, test_preds)
    print(f"Test F1@0.5: {test_f1:.3f}")
    print(f"Test Accuracy@0.5: {test_acc_05:.3f}")

    slang_ds = TextDataset(slang_df["text"].tolist(), slang_df["label"].astype(int).tolist())
    slang_loader = DataLoader(slang_ds, batch_size=max(1, args.batch_size * 2), shuffle=False, collate_fn=collate)
    slang_eval = evaluate(model, slang_loader, device)
    slang_probs = slang_eval["_probs"]
    slang_gold = slang_eval["_gold"]
    slang_preds = (slang_probs >= 0.5).astype(int)
    slang_conf = confusion_from_arrays(slang_gold, slang_preds)
    slang_acc = float(accuracy_score(slang_gold, slang_preds)) if len(slang_gold) else 0.0
    slang_f1 = float(f1_score(slang_gold, slang_preds, zero_division=0)) if len(np.unique(slang_gold)) > 1 else 0.0
    print(f"Slang test Accuracy@0.5: {slang_acc:.3f}")
    print(f"Slang test F1@0.5: {slang_f1:.3f}")

    # Persist model + metadata
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        tag = (args.run_id.strip() or f"seed-{args.seed}")
        out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "lr_e2e" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    meta_path = out_dir / "meta.json"
    test_pred_path = out_dir / "test_predictions.csv"
    slang_pred_path = out_dir / "slang_predictions.csv"
    if best_state is not None:
        # Replace runtime params with the best snapshot
        torch.save(best_state, ckpt_path)
    else:
        # Save final state if best not captured (shouldn't happen)
        torch.save({
            "encoder": model.encoder.state_dict(),
            "head": model.head.state_dict(),
        }, ckpt_path)
    save_predictions(test_df, test_probs, test_preds, test_pred_path)
    save_predictions(slang_df, slang_probs, slang_preds, slang_pred_path)
    meta = {
        "model_id": MODEL_ID,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "run_id": args.run_id,
        "out_dir": str(out_dir.relative_to(PROJECT_ROOT) if str(out_dir).startswith(str(PROJECT_ROOT)) else out_dir),
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        },
        "metrics": {
            "val_best_f1@0.5": best_f1,
            "test_f1@0.5": test_f1,
            "test_acc@0.5": test_acc_05,
            "test_confusion@0.5": test_conf,
            "slang_test_set@0.5": {
                **slang_conf,
                "accuracy": slang_acc,
                "f1": slang_f1,
            },
        },
        "artifacts": {
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT) if str(ckpt_path).startswith(str(PROJECT_ROOT)) else ckpt_path),
            "test_predictions": str(test_pred_path.relative_to(PROJECT_ROOT) if str(test_pred_path).startswith(str(PROJECT_ROOT)) else test_pred_path),
            "slang_predictions": str(slang_pred_path.relative_to(PROJECT_ROOT) if str(slang_pred_path).startswith(str(PROJECT_ROOT)) else slang_pred_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved E2E model to: {ckpt_path}")


if __name__ == "__main__":
    main()
