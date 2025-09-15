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
- Full fine-tuning of a 7B encoder is memory-intensive. This script exposes
  flags to try BF16 and bitsandbytes 4-bit loading. For real fine-tuning on
  limited VRAM, LoRA/QLoRA is recommended (not included here to keep deps light).

Usage (examples)
  # Default small-batch run (GPU if available)
  python -m src.detect.detect_train_lr_e2e --epochs 3 --batch_size 8 --max_length 128 --lr 2e-5

  # Try 4-bit loading to reduce VRAM (training effectiveness may be limited without PEFT)
  python -m src.detect.detect_train_lr_e2e --epochs 3 --batch_size 8 --max_length 128 --lr 2e-5 --load_in_4bit

Dependencies
- torch>=2.2, transformers>=4.41, bitsandbytes>=0.43 (optional), scikit-learn, pandas, numpy, openpyxl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:  # optional
    from transformers import BitsAndBytesConfig  # type: ignore
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


# ------------------------ Tunable parameters ------------------------
MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42

# Default batch size and length
DEF_BATCH = 8
DEF_MAXLEN = 128

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ------------------------ Stability/efficiency settings ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Stratified 85/15, then 15% of the 85% as val -> 72.25/12.75/15
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    print(f"Splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_encoder(load_in_4bit: bool, device_map_auto: bool, use_flash_attn: bool, device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {}
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if load_in_4bit:
        if not _HAS_BNB:
            print("[WARN] bitsandbytes not available; falling back to standard load.")
            load_in_4bit = False
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_cfg
            if device_map_auto:
                load_kwargs["device_map"] = "auto"

    try:
        enc = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            **load_kwargs,
        )
    except Exception:
        # Retry without flash-attn hint
        load_kwargs.pop("attn_implementation", None)
        enc = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            **load_kwargs,
        )

    if not device_map_auto:
        enc = enc.to(device)

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
    for batch in loader:
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        yb = batch["label"]
        if device == "cuda":
            ids = ids.cuda(non_blocking=True)
            attn = attn.cuda(non_blocking=True)
        logits = model(ids, attn)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.append(p)
        gold.append(yb.numpy().ravel())
    probs = np.concatenate(probs) if probs else np.zeros((0,))
    gold = np.concatenate(gold) if gold else np.zeros((0,))
    res = {"ap": 0.0, "auc": 0.0, "f1@0.5": 0.0}
    if len(np.unique(gold)) > 1:
        res["ap"] = float(average_precision_score(gold, probs))
        res["auc"] = float(roc_auc_score(gold, probs))
        res["f1@0.5"] = float(f1_score(gold, (probs >= 0.5).astype(int), zero_division=0))
    # also return raw to enable threshold scan
    res["_probs"] = probs
    res["_gold"] = gold
    return res


def scan_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
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
    ap.add_argument("--load_in_4bit", action="store_true", help="Load encoder in 4-bit (bnb). Caution: for training, LoRA is usually preferred.")
    ap.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' (may shard across CPU/GPU)")
    ap.add_argument("--use_flash_attn", action="store_true", help="Hint FlashAttention2 if available")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_df, val_df, test_df = load_data()
    tok, enc = build_encoder(args.load_in_4bit, args.device_map_auto, args.use_flash_attn, device)

    model = E2ELinear(enc)
    if device == "cuda" and not args.device_map_auto:
        model = model.cuda()

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
    if device == "cuda":
        pos_weight = pos_weight.cuda()
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
            if device == "cuda":
                ids = ids.cuda(non_blocking=True)
                attn = attn.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(ids, attn)
                loss = bce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch} train_loss={total_loss/ max(1,len(train_loader)):.4f}")

        # Validation
        val = evaluate(model, val_loader, device)
        t_star, f1_star = scan_best_threshold(val["_gold"], val["_probs"])
        print(f"Val: AUC={val['auc']:.3f} AP={val['ap']:.3f} F1@0.5={val['f1@0.5']:.3f} | best t*={t_star:.3f} F1={f1_star:.3f}")
        if f1_star > best_f1:
            best_f1 = f1_star
            best_t = t_star
            # Save minimal state
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                "head": model.head.state_dict(),
            }

    # Test with best threshold
    test = evaluate(model, test_loader, device)
    test_preds = (test["_probs"] >= best_t).astype(int)
    test_f1 = float(f1_score(test["_gold"], test_preds, zero_division=0)) if len(np.unique(test["_gold"])) > 1 else 0.0
    print(f"Test: AUC={test['auc']:.3f} AP={test['ap']:.3f} F1@t*={test_f1:.3f} (t*={best_t:.3f})")

    # Persist model + metadata
    out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "lr_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    meta_path = out_dir / "meta.json"
    if best_state is not None:
        # Replace runtime params with the best snapshot
        torch.save(best_state, ckpt_path)
    else:
        # Save final state if best not captured (shouldn't happen)
        torch.save({
            "encoder": model.encoder.state_dict(),
            "head": model.head.state_dict(),
        }, ckpt_path)
    meta = {
        "model_id": MODEL_ID,
        "threshold": best_t,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "load_in_4bit": bool(args.load_in_4bit),
        "device_map_auto": bool(args.device_map_auto),
        "use_flash_attn": bool(args.use_flash_attn),
        "metrics": {
            "val_best_f1": best_f1,
            "test_auc": test.get("auc", 0.0),
            "test_ap": test.get("ap", 0.0),
            "test_f1_at_t": test_f1,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved E2E model to: {ckpt_path}")


if __name__ == "__main__":
    main()

