#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment C: Frozen Mistral encoder + BERT-style classification head

This mirrors the logistic-regression baseline but swaps the CPU LR head for a
small BERT-style classifier implemented in PyTorch. The Mistral encoder remains
frozen while only the classifier head updates.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# ------------------------ Tunable parameters ------------------------
MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42
HEAD_DROPOUT = 0.1

# Default batch size and length (can be overridden via CLI)
DEF_BATCH = 32
DEF_MAXLEN = 512

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data …")
    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"

    missing = [p for p in [sent_path, gaz_path] if not p.exists()]
    if missing:
        msg = (
            "Could not find required files:\n"
            + "\n".join(f" - {p}" for p in missing)
            + f"\nCWD = {Path.cwd()}\nPROJECT_ROOT = {PROJECT_ROOT}"
        )
        raise FileNotFoundError(msg)

    df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    if "label" not in df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        df["label"] = df["text"].apply(has_verlan)

    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=seed
    )
    print(f"Splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

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
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
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

def forward_encoder(encoder: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    start_device = next(encoder.parameters()).device
    input_ids = input_ids.to(start_device)
    attention_mask = attention_mask.to(start_device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(start_device.type == "cuda")):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
    return pooled.detach()

def evaluate(head: BertStyleHead, encoder: AutoModel, loader: DataLoader) -> Dict[str, float]:
    head.eval()
    probs, gold = [], []
    head_device = next(head.parameters()).device
    for batch in loader:
        with torch.no_grad():
            pooled = forward_encoder(encoder, batch["input_ids"], batch["attention_mask"])
        pooled = pooled.to(head_device)
        logits = head(pooled)
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
    ap = argparse.ArgumentParser(description="Frozen Mistral encoder with BERT-style head")
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    device = "cuda"

    set_seed(args.seed)
    train_df, val_df, test_df = load_data(args.seed)
    tok, enc = load_encoder()

    hidden = int(getattr(enc.config, "hidden_size", 1024))
    head = BertStyleHead(hidden_size=hidden)
    head.to(device)

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
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                pooled = forward_encoder(enc, batch["input_ids"], batch["attention_mask"])
            logits = head(pooled.to(device))
            loss = bce(logits, batch["label"].to(device))
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} head_loss={avg_loss:.4f}")

        val = evaluate(head, enc, val_loader)
        yv_pred = (val["_probs"] >= 0.5).astype(int)
        print(classification_report(val["_gold"], yv_pred, digits=3))
        print(f"Val F1@0.5: {val['f1@0.5']:.3f}")
        print(f"Val Accuracy@0.5: {val['acc@0.5']:.3f}")
        if val["f1@0.5"] > best_f1:
            best_f1 = val["f1@0.5"]
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    test = evaluate(head, enc, test_loader)
    test_preds = (test["_probs"] >= 0.5).astype(int)
    test_f1 = float(f1_score(test["_gold"], test_preds, zero_division=0)) if len(np.unique(test["_gold"])) > 1 else 0.0
    test_acc_05 = float(accuracy_score(test["_gold"], test_preds)) if len(np.unique(test["_gold"])) > 1 else 0.0
    print(f"Test F1@0.5: {test_f1:.3f}")
    print(f"Test Accuracy@0.5: {test_acc_05:.3f}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        tag = (args.run_id.strip() or f"seed-{args.seed}")
        out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "bert_head" / "frozen" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "head.pt"
    meta_path = out_dir / "meta.json"
    state = best_state if best_state is not None else head.state_dict()
    torch.save(state, ckpt_path)
    meta = {
        "model_id": MODEL_ID,
        "encoder_frozen": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "head_lr": args.head_lr,
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
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved frozen head to: {ckpt_path}")

if __name__ == "__main__":
    main()
