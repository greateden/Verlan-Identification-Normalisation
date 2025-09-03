#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence-level Verlan detector (mistral_mistral)
- Tokenizer:  Salesforce/SFR-Embedding-Mistral
- Encoder:    Salesforce/SFR-Embedding-Mistral
- Head:       Linear (1 logit) on mean-pooled sentence vector
- Activation: Sigmoid, fixed threshold t = 0.5
- Splits:     Fixed random split (no stratification): 72.25 / 12.75 / 15 (train/val/test)

This script mirrors the style/IO of our other detect_* scripts and implements the
two-branch pipeline described in the flowchart:
  • Branch 1 (Diagnostic): Frozen encoder → mean-pool → UMAP (dataset separability check)
  • Branch 2 (Classifier): Fine-tune encoder + linear head → evaluate on fixed test split

Recommended deps (minimum):
- torch>=2.2
- transformers>=4.41
- scikit-learn>=1.3
- umap-learn>=0.5
- matplotlib, pandas, numpy, openpyxl

Usage (examples)
---------------
# 1) Plot UMAP on the full dataset (all splits combined), no training
python detect_train_mistral_mistral.py --freeze_for_umap --split all --num_texts 0 --no_show

# 2) Train for 3 epochs, save under models/detect/<YYYY-MM-DD>/mistral_mistral/
python detect_train_mistral_mistral.py --epochs 3 --out_name mistral_mistral

Notes
-----
• We keep max_length modest (128) by default to fit on 16GB GPUs; raise if you have headroom.
• We deterministically split with random_state=42, *without* stratification (per flowchart).
• Comments kept verbose (≥30%) to document all moving parts clearly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------- Environment knobs to reduce tokenizer thread storms & CUDA OOM ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")

# ---------- Project-relative paths (match existing scripts) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # expect src/detect/<this_file>.py
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR  = PROJECT_ROOT / "docs" / "results"
MODELS_DIR   = PROJECT_ROOT / "models" / "detect"
DEFAULT_UMAP_PNG = RESULTS_DIR / "mistral_mistral_umap.png"


def _lazy_imports():
    """Import heavy libs on demand so that --help is fast and CI doesn't choke."""
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
        import umap  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Missing a dependency? {e}")


# ------------------------------ CLI ---------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SFR-Embedding-Mistral → (UMAP | fine-tuned linear head) pipeline",
    )
    p.add_argument("--mistral_tokenizer", default="Salesforce/SFR-Embedding-Mistral",
                   help="HF repo id/path for tokenizer (defaults to SFR-Embedding-Mistral)")
    p.add_argument("--mistral_model", default="Salesforce/SFR-Embedding-Mistral",
                   help="HF repo id/path for encoder (defaults to SFR-Embedding-Mistral)")
    p.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenization")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--device", default=None, help="Torch device, e.g., cuda or cpu (auto by default)")

    # UMAP controls (Branch 1)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                   help="Which split to visualize for UMAP (default: all)")
    p.add_argument("--num_texts", type=int, default=0,
                   help="Optional subsample size for UMAP; 0 means use all")
    p.add_argument("--save", default=str(DEFAULT_UMAP_PNG),
                   help="Path to save UMAP PNG (Branch 1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits/UMAP")
    p.add_argument("--no_show", action="store_true", help="Don't show UMAP window (headless nodes)")
    # UMAP memory / efficiency controls
    p.add_argument("--umap_dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                   help="Precision for UMAP embedding extraction (default: float16 for memory saving)")
    p.add_argument("--umap_batch_size", type=int, default=32,
                   help="Batch size for embedding extraction during UMAP (auto-reduces on OOM)")
    p.add_argument("--umap_device_map_auto", action="store_true",
                   help="Use accelerate/transformers device_map='auto' when loading the encoder for UMAP (helps fit on limited VRAM)")

    # Training controls (Branch 2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2, help="Early-stop patience on val loss")
    p.add_argument("--out_name", default="mistral_mistral", help="Subdir under models/detect/<YYYY-MM-DD>/")

    # Mode toggles
    p.add_argument("--freeze_for_umap", action="store_true",
                   help="Do only the Diagnostic UMAP (no training).")
    return p


# -------------------------- Data loading -----------------------------
def _read_raw_tables() -> Tuple["pandas.DataFrame", "pandas.DataFrame"]:
    """Read sentence dataset + gazetteer from PROJECT_ROOT/data/raw.

    We require Sentences_balanced.xlsx (no fallback)
    to stay compatible with other scripts dropped in this repo.
    """
    import pandas as pd

    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path  = RAW_DIR / "GazetteerEntries.xlsx"

    missing = [p for p in [sent_path, gaz_path] if not p.exists()]
    if missing:
        msg = (
            "❌ Missing required files:\n"
            + "\n".join(f" - {p}" for p in missing)
            + f"\n\nResolved PROJECT_ROOT = {PROJECT_ROOT}\n"
            "Please place the Excel files under data/raw/."
        )
        raise FileNotFoundError(msg)

    df  = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)

    # If no label column, derive a weak label by lexicon hit (token-level heuristic).
    if "label" not in df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        df["label"] = df["text"].apply(has_verlan)

    # Keep only required columns
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df, lex


def load_splits(seed: int = 42):
    """Return (train_df, val_df, test_df) with fixed random split (NO stratification)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df, _ = _read_raw_tables()
    # First carve out 15% as test (no stratification)
    train_full, test_df = train_test_split(df, test_size=0.15, random_state=seed, shuffle=True)
    # From the remaining 85%, carve 15% for validation => 12.75% overall val
    train_df, val_df = train_test_split(train_full, test_size=0.15, random_state=seed, shuffle=True)
    # Sanity check proportions (best-effort; won't be exact on tiny datasets)
    # print("Split sizes:", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def select_texts_labels(split: str, num_texts: int, seed: int) -> Tuple[List[str], List[int]]:
    """Get (texts, labels) from the chosen split, with optional subsampling for UMAP."""
    import pandas as pd

    train_df, val_df, test_df = load_splits(seed)
    if split == "train":
        df = train_df
    elif split == "val":
        df = val_df
    elif split == "test":
        df = test_df
    else:
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if num_texts and num_texts > 0:
        df = df.sample(n=min(num_texts, len(df)), random_state=seed)

    texts = normalize_texts(df["text"].astype(str).tolist())
    labels = df["label"].astype(int).tolist()
    return texts, labels


# ------------------------ Text normalization -------------------------
def normalize_texts(texts: Iterable[str]) -> List[str]:
    """Basic normalization: Unicode NFC; strip control chars (keep accents)."""
    import unicodedata
    out = []
    for s in texts:
        s2 = unicodedata.normalize("NFC", str(s))
        s2 = "".join(ch for ch in s2 if (unicodedata.category(ch) != "Cc") or ch in "\t\n\r")
        out.append(s2)
    return out


# ----------------------- Tokenize & Embeddings -----------------------
def mean_pool(last_hidden_state, attention_mask):
    """Mean-pool over valid tokens according to attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1)                           # [B, 1]
    return summed / counts


def encode_umap_embeddings(
    texts: List[str],
    tok_id: str,
    enc_id: str,
    max_length: int,
    device: str,
    dtype: str = "float16",
    batch_size: int = 32,
    device_map_auto: bool = False,
) -> np.ndarray:
    """Tokenize with SFR-Embedding-Mistral, pass through a *frozen* encoder, and return sentence vectors.

    Strategies applied to mitigate CUDA OOM:
      1. Load model in reduced precision (float16/bfloat16) when on CUDA.
      2. Optionally leverage device_map='auto' (HF accelerate) for model sharding.
      3. Adaptive batch size: on runtime OOM, halve batch size and retry.
      4. Final fallback: run entirely on CPU if GPU load fails.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(tok_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Select torch dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    def _try_load(target_device: str, use_auto: bool):
        load_kwargs = {"torch_dtype": torch_dtype} if torch_dtype != torch.float32 else {}
        if use_auto:
            # device_map auto ignores explicit .to(device) call
            return AutoModel.from_pretrained(enc_id, device_map="auto", **load_kwargs)
        m = AutoModel.from_pretrained(enc_id, **load_kwargs)
        return m.to(target_device)

    enc = None
    tried = []
    # Attempt order: (requested device, maybe auto map) -> (requested device, no auto) -> CPU
    for (dev, auto_map) in [
        (device, device_map_auto),
        (device, False) if device_map_auto else (device, False),
        ("cpu", False),
    ]:
        key = f"dev={dev},auto={auto_map}"
        if key in tried:
            continue
        tried.append(key)
        try:
            enc = _try_load(dev, auto_map)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ OOM while loading model with {key}; trying next fallback…")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Failed loading model with {key}: {e}")
    if enc is None:
        raise RuntimeError("Could not load model on any device (GPU/CPU)")

    enc.eval()

    # Hidden size (robust even if sharded)
    hidden_size = int(getattr(enc.config, "hidden_size", 1024))
    all_vecs: List[np.ndarray] = []

    # When using auto device_map, inputs can stay on CPU; HF dispatch handles device placement.
    run_on_cpu_inputs = isinstance(getattr(enc, "hf_device_map", None), dict)
    current_bs = max(1, batch_size)
    i = 0
    while i < len(texts):
        batch = texts[i:i+current_bs]
        try:
            enc_in = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc_in["input_ids"]
            attn      = enc_in["attention_mask"]
            if not run_on_cpu_inputs:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=(torch_dtype!=torch.float32 and (not run_on_cpu_inputs) and device.startswith("cuda"))):
                    out = enc(input_ids=input_ids, attention_mask=attn, return_dict=True)
                pooled = mean_pool(out.last_hidden_state, attn).detach().cpu().numpy()
            all_vecs.append(pooled)
            i += current_bs
            del enc_in, input_ids, attn, out, pooled
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size
            if current_bs == 1:
                print("❌ OOM even at batch_size=1; aborting embedding extraction. Consider --umap_dtype float16 and/or CPU mode.")
                break
            new_bs = max(1, current_bs // 2)
            print(f"⚠️ OOM at batch_size={current_bs}; retrying with batch_size={new_bs} …")
            current_bs = new_bs
            torch.cuda.empty_cache()
        except RuntimeError as e:
            # Some mixed precision issues on older GPUs; retry in float32 if needed
            if "attempting to use dtype half" in str(e).lower() and torch_dtype != torch.float32:
                print("⚠️ Mixed precision not supported on this GPU for this op; switching to float32 and continuing…")
                torch_dtype = torch.float32
                continue
            raise

    if not all_vecs:
        return np.zeros((0, hidden_size), dtype=np.float32)
    return np.vstack(all_vecs)


def plot_umap(embeddings: np.ndarray, labels: Optional[List[int]], save_path: str, seed: int, show: bool):
    """2D UMAP scatter with class coloring to visually check separability."""
    import umap
    import matplotlib.pyplot as plt

    if embeddings.shape[0] == 0:
        print("⚠️ No embeddings to plot.")
        return

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=seed)
    X2 = reducer.fit_transform(embeddings)

    plt.figure(figsize=(6.4, 5.0))
    if labels is None:
        plt.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.7)
    else:
        labels = np.asarray(labels)
        mask1 = labels == 1
        mask0 = labels == 0
        plt.scatter(X2[mask0, 0], X2[mask0, 1], s=8, alpha=0.7, label="Standard")
        plt.scatter(X2[mask1, 0], X2[mask1, 1], s=8, alpha=0.7, label="Verlan")
        plt.legend()
    plt.title("UMAP (SFR-Embedding-Mistral)")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


# -------------------- Fine-tuned classifier branch -------------------
class MistralBinary:
    """Thin wrapper: SFR-Embedding-Mistral + linear 1-logit head with mean pooling."""
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel

        self.device = device
        self.encoder = AutoModel.from_pretrained(model_id).to(device)
        hidden = int(self.encoder.config.hidden_size)

        import torch.nn as nn
        self.head = nn.Linear(hidden, 1).to(device)

    def parameters(self):
        # Expose trainable params (encoder + head) to optimizer
        for p in self.encoder.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    def zero_grad(self):
        import torch
        self.encoder.zero_grad(set_to_none=True)
        self.head.zero_grad(set_to_none=True)

    def forward_logits(self, input_ids, attention_mask):
        """Forward pass to logits (pre-sigmoid)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


@dataclass
class TrainConfig:
    mistral_tokenizer: str
    mistral_model: str
    max_length: int
    threshold: float = 0.5
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 2


def _tokenizer_for_id(tok_id: str):
    """Prepare tokenizer with a pad token if missing (common for decoder LLMs)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def train_classifier(
    texts: List[str], labels: List[int], cfg: TrainConfig, device: str, out_subdir: Path
) -> Dict[str, float]:
    """Train + early stop on val loss, then evaluate on the fixed test split.

    Returns a metrics dict and writes the best checkpoint to out_subdir/_tmp/model.pt.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    class TokDataset(Dataset):
        """On-the-fly tokenization dataset to keep RAM low."""
        def __init__(self, X: List[str], y: List[int], tok, max_length: int):
            self.X = X; self.y = y; self.tok = tok; self.max_length = max_length
        def __len__(self): return len(self.X)
        def __getitem__(self, i: int):
            enc = self.tok(self.X[i], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0).long(),
                "attention_mask": enc["attention_mask"].squeeze(0).long(),
                "label": torch.tensor([float(self.y[i])], dtype=torch.float32),
            }

    # Tokenizer (shared across splits)
    tok = _tokenizer_for_id(cfg.mistral_tokenizer)

    # Create model
    model = MistralBinary(cfg.mistral_model, device)

    # Deterministic fixed random split (NO stratification)
    y_arr = np.asarray(labels)
    X_train, X_tmp, y_train, y_tmp = train_test_split(texts, y_arr, test_size=0.15, random_state=42, shuffle=True)
    X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, shuffle=True)
    X_test,  y_test = X_tmp, y_tmp

    # Datasets & loaders
    train_ds = TokDataset(X_train, y_train.tolist(), tok, cfg.max_length)
    val_ds   = TokDataset(X_val,   y_val.tolist(),   tok, cfg.max_length)
    test_ds  = TokDataset(X_test,  y_test.tolist(),  tok, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size)

    # Optimizer & loss
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Early-stopping on val loss
    best_val = float("inf")
    epochs_no_improve = 0
    tmp_dir = out_subdir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.encoder.train(); model.head.train()
        train_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            yb   = batch["label"].to(device)
            logits = model.forward_logits(ids, attn)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())

            # free batch tensors to keep memory tidy
            del ids, attn, yb, logits, loss
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Validation pass
        model.encoder.eval(); model.head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                yb   = batch["label"].to(device)
                logits = model.forward_logits(ids, attn)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item())
                del ids, attn, yb, logits, loss
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Check for improvement
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save(
                {"encoder_state": model.encoder.state_dict(),
                 "head_state": model.head.state_dict()},
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping after {epoch} epochs (no improve ≥ {cfg.patience})")
                break

    # Load best state for test evaluation
    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location=device)
        model.encoder.load_state_dict(data["encoder_state"])
        model.head.load_state_dict(data["head_state"])
    model.encoder.eval(); model.head.eval()

    # Test metrics
    with torch.no_grad():
        probs, gold = [], []
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            yb   = batch["label"].to(device)
            logits = model.forward_logits(ids, attn)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(p); gold.append(yb.cpu().numpy().ravel())
            del ids, attn, yb, logits
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        probs = np.concatenate(probs) if probs else np.zeros((0,))
        gold  = np.concatenate(gold)  if gold  else np.zeros((0,))

    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    preds = (probs >= cfg.threshold).astype(int)
    f1  = float(f1_score(gold, preds, zero_division=0)) if len(np.unique(gold)) > 1 else 0.0
    ap  = float(average_precision_score(gold, probs))   if len(np.unique(gold)) > 1 else 0.0
    auc = float(roc_auc_score(gold, probs))             if len(np.unique(gold)) > 1 else 0.0

    # Save a minimal config for the benchmark script to reload
    (out_subdir / "config.json").write_text(
        __import__("json").dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"test_f1@0.5": f1, "test_ap": ap, "test_auc": auc, "model_dir": str(out_subdir)}


def finalize_save(out_subdir: Path) -> Path:
    """Move _tmp snapshot to dated directory and update models/detect/latest symlink."""
    date = datetime.now().strftime("%Y-%m-%d")
    date_root = MODELS_DIR / date
    final_dir = date_root / out_subdir.name
    date_root.mkdir(parents=True, exist_ok=True)

    # Move tmp dir into final_dir
    tmp_src = out_subdir / "_tmp"
    if tmp_src.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        # copy files (model.pt) and remove tmp
        for p in tmp_src.iterdir():
            shutil.move(str(p), str(final_dir / p.name))
        shutil.rmtree(tmp_src, ignore_errors=True)

    # Move config.json next to model.pt
    cfg_src = out_subdir / "config.json"
    if cfg_src.exists():
        shutil.copy2(str(cfg_src), str(final_dir / "config.json"))

    # Update latest → <YYYY-MM-DD> symlink if supported
    latest = MODELS_DIR / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(date_root)
    except Exception:
        # On Windows (or restricted FS) symlinks may fail; ignore.
        pass
    return final_dir


# ------------------------------- Main --------------------------------
def main(argv: List[str] | None = None) -> int:
    _lazy_imports()
    import torch

    args = build_argparser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Branch 1: UMAP only (frozen encoder)
    if args.freeze_for_umap:
        texts, labels = select_texts_labels(args.split, args.num_texts, args.seed)
        print(f"UMAP on {len(texts)} texts (split={args.split}) …")
        vecs = encode_umap_embeddings(texts, args.mistral_tokenizer, args.mistral_model, args.max_length, device)
        plot_umap(vecs, labels, save_path=args.save, seed=args.seed, show=(not args.no_show))
        print(f"UMAP saved to: {args.save}")
        return 0

    # Branch 2: Train + test evaluation
    full_texts, full_labels = select_texts_labels("all", 0, args.seed)
    cfg = TrainConfig(
        mistral_tokenizer=args.mistral_tokenizer,
        mistral_model=args.mistral_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    print("Training SFR-Embedding-Mistral + linear head (no LR) …")
    work_dir = MODELS_DIR / "_work" / args.out_name
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_classifier(full_texts, full_labels, cfg, device, work_dir)

    # Finalize and print summary
    final_dir = finalize_save(work_dir)
    print("Saved best model under:", final_dir)
    print(f"Test AP={metrics['test_ap']:.3f} | AUC={metrics['test_auc']:.3f} | F1@0.5={metrics['test_f1@0.5']:.3f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
