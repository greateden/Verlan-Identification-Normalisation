#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment D: End-to-end fine-tuning with a BERT-style classification head

This mirrors the LR fine-tuning experiment but swaps the linear head for the
same BERT-style classifier used in the frozen baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
HEAD_DROPOUT = 0.1

# Default batch size and length (aligned with other experiments)
DEF_BATCH = 32
DEF_MAXLEN = 512

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import random

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_encoder():
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
    def __len__(self) -> int:
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

class BertStyleHead(nn.Module):
    """Dropout -> Linear -> Tanh -> Dropout -> Linear."""
    def __init__(self, hidden_size: int, dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(hidden_size, 1)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1)
    pooled = (last_hidden * mask).sum(dim=1) / denom
    return nn.functional.normalize(pooled, p=2, dim=1)

class MistralBertClassifier(nn.Module):
    def __init__(self, encoder: AutoModel):
        super().__init__()
        self.encoder = encoder
        hidden = int(getattr(self.encoder.config, "hidden_size", 1024))
        self.head = BertStyleHead(hidden)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        logits = self.head(pooled)
        return logits

@torch.no_grad()
def evaluate(model: MistralBertClassifier, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    probs, gold = [], []
    start_device = next(model.encoder.parameters()).device
    for batch in loader:
        ids = batch["input_ids"].to(start_device)
        attn = batch["attention_mask"].to(start_device)
        with torch.no_grad():
            logits = model(ids, attn)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.append(p)
        gold.append(batch["label"].numpy().ravel())
    probs = np.concatenate(probs) if probs else np.zeros((0,))
    gold = np.concatenate(gold) if gold else np.zeros((0,))
    res = {"f1@0.5": 0.0, "acc@0.5": 0.0, "_probs": probs, "_gold": gold}
    if len(np.unique(gold)) > 1:
        res["f1@0.5"] = float(f1_score(gold, (probs >= 0.5).astype(int), zero_division=0))
        res["acc@0.5"] = float(accuracy_score(gold, (probs >= 0.5).astype(int)))
    return res

def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end fine-tuning with BERT-style head")
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    device = "cuda"

    set_seed(args.seed)

    train_df, val_df, test_df = load_verlan_dataset(args.seed)
    slang_df = load_slang_test_set()
    tok, enc = load_encoder()

    model = MistralBertClassifier(enc)
    start_device = next(model.encoder.parameters()).device
    model.head.to(start_device)
    model.train()

    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].astype(int).tolist())
    val_ds = TextDataset(val_df["text"].tolist(), val_df["label"].astype(int).tolist())
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label"].astype(int).tolist())

    collate = make_collate(tok, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size * 2), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size * 2), shuffle=False, collate_fn=collate)

    y_np = train_df["label"].astype(int).values
    n_pos = max(1, int((y_np == 1).sum()))
    n_neg = max(1, int((y_np == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    pos_weight = pos_weight.to(start_device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(start_device)
            attn = batch["attention_mask"].to(start_device)
            yb = batch["label"].to(start_device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(start_device.type == "cuda")):
                logits = model(ids, attn)
                loss = bce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch} train_loss={total_loss / max(1, len(train_loader)):.4f}")

        val = evaluate(model, val_loader)
        yv_pred = (val["_probs"] >= 0.5).astype(int)
        print(classification_report(val["_gold"], yv_pred, digits=3))
        print(f"Val F1@0.5: {val['f1@0.5']:.3f}")
        print(f"Val Accuracy@0.5: {val['acc@0.5']:.3f}")
        if val["f1@0.5"] > best_f1:
            best_f1 = val["f1@0.5"]
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                "head": model.head.state_dict(),
            }

    test = evaluate(model, test_loader)
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
    slang_eval = evaluate(model, slang_loader)
    slang_probs = slang_eval["_probs"]
    slang_gold = slang_eval["_gold"]
    slang_preds = (slang_probs >= 0.5).astype(int)
    slang_conf = confusion_from_arrays(slang_gold, slang_preds)
    slang_acc = float(accuracy_score(slang_gold, slang_preds)) if len(slang_gold) else 0.0
    slang_f1 = float(f1_score(slang_gold, slang_preds, zero_division=0)) if len(np.unique(slang_gold)) > 1 else 0.0
    print(f"Slang test Accuracy@0.5: {slang_acc:.3f}")
    print(f"Slang test F1@0.5: {slang_f1:.3f}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        tag = (args.run_id.strip() or f"seed-{args.seed}")
        out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "bert_head" / "e2e" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "model.pt"
    meta_path = out_dir / "meta.json"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
    else:
        torch.save({
            "encoder": model.encoder.state_dict(),
            "head": model.head.state_dict(),
        }, ckpt_path)
    test_pred_path = out_dir / "test_predictions.csv"
    slang_pred_path = out_dir / "slang_predictions.csv"
    save_predictions(test_df, test_probs, test_preds, test_pred_path)
    save_predictions(slang_df, slang_probs, slang_preds, slang_pred_path)
    meta = {
        "model_id": MODEL_ID,
        "encoder_frozen": False,
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
        }
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model to: {ckpt_path}")

if __name__ == "__main__":
    main()
