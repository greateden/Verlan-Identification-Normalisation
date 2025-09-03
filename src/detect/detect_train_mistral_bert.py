#!/usr/bin/env python3
"""
Mistral tokenizer -> CamemBERT encoder

Implements the requested two-branch pipeline using a single tokenization pass:

- Branch 1 (Diagnostic): Frozen CamemBERT encoder. We compute sentence embeddings
  and plot a UMAP to visually check dataset separability.

- Branch 2 (Classifier): Fine-tune CamemBERT with a simple linear head (1 logit)
  for sentence-level binary classification (verlan vs standard). Sigmoid + fixed
  threshold t=0.5 for now. No LR is used — this is an embedded model.

Notes
- Tokenizer/encoder mismatch: We intentionally tokenize with Mistral 7B’s tokenizer,
  then remap token ids into CamemBERT’s vocab range via modulo to avoid index errors.
  This is hacky; results should be interpreted with care. It mirrors the current
  UMAP behavior the project already uses.
- Normalization: Basic Unicode NFC normalization and control-strip are applied
  before tokenization; accents are preserved.

Data & saving
- Splits: stratified 72.25% train / 12.75% val / 15% test.
- Saves UMAP under docs/results by default (configurable via --save).
- Saves the fine-tuned model under models/detect/<date>/camembert_mistral/ and
  updates models/detect/latest/camembert_mistral/.

Examples
  # Train end-to-end, make UMAP over all splits
  python -m src.detect.detect_train_mistral_bert \
    --split all --save docs/results/mistral_bert_umap.png \
    --epochs 3 --lr 2e-5 --batch_size 16

  # Faster sanity run on train only, subsample 1000 texts
  python -m src.detect.detect_train_mistral_bert \
    --split train --num_texts 1000 --epochs 1 --no_show

Dependencies
  pip install transformers torch umap-learn matplotlib pandas scikit-learn openpyxl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Align environment knobs with other detection scripts
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6",
)

# Repository paths consistent with other detect scripts
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SAVE = PROJECT_ROOT / "docs" / "results" / "mistral_bert_umap.png"
MODELS_DIR = PROJECT_ROOT / "models" / "detect"


def _lazy_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
        import umap  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        print(
            "Missing dependencies. Please install: transformers torch umap-learn matplotlib",
            file=sys.stderr,
        )
        raise


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mistral-tokenizer -> CamemBERT pipeline with UMAP + fine-tuned classifier",
    )
    p.add_argument(
        "--mistral_tokenizer",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HF repo or path for Mistral tokenizer (only tokenizer is used)",
    )
    p.add_argument(
        "--camembert_model",
        default="camembert-base",
        help="HF model id for CamemBERT encoder (e.g., camembert-base)",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to visualise (matches other detect scripts)",
    )
    p.add_argument(
        "--num_texts",
        type=int,
        default=0,
        help="Optional subsample size per chosen split; 0 means use all",
    )
    p.add_argument(
        "--save",
        default=str(DEFAULT_SAVE),
        help="Where to save the UMAP plot (default under docs/results)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP and numpy",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cpu. Default: cuda if available else cpu",
    )
    # Training options (Branch 2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2, help="Early stop patience (val loss)")
    p.add_argument("--freeze_for_umap", action="store_true", help="Do not fine-tune; only UMAP")
    p.add_argument("--out_name", default="camembert_mistral", help="Subdir name under models/detect/<date>/")
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Do not show plot window (useful on headless boxes)",
    )
    return p

def load_splits(seed: int):
    """Load full dataset and return (train_df, val_df, test_df) with 72.25/12.75/15 split.

    Mirrors the logic used in other detect scripts to ensure consistency.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

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
        df, test_size=0.15, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def select_texts_labels(split: str, num_texts: int, seed: int) -> Tuple[List[str], List[int]]:
    """Select texts/labels from the requested split, optionally subsampling.

    split: one of {train, val, test, all}
    num_texts: 0 means all available; otherwise sample that many from the chosen set
    """
    import pandas as pd

    train_df, val_df, test_df = load_splits(seed)
    if split == "train":
        df = train_df
    elif split == "val":
        df = val_df
    elif split == "test":
        df = test_df
    else:  # all
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if num_texts and num_texts > 0:
        n = min(num_texts, len(df))
        df = df.sample(n=n, random_state=seed)

    # Basic normalization: NFC + strip control chars, keep accents
    import unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFC", str(s))
        # strip Cc controls except tab/newline/carriage return
        return "".join(ch for ch in s if (unicodedata.category(ch) != "Cc") or ch in "\t\n\r")
    texts = [ _norm(x) for x in df["text"].astype(str).tolist() ]
    labels = df["label"].astype(int).tolist()
    return texts, labels


def unsafe_remap_ids_to_camembert(
    ids, attention_mask, cam_vocab_size: int, cam_pad_id: int
):
    """Map arbitrary token ids into BERT's vocab range.

    - Modulo maps all ids safely into range [0, bert_vocab_size).
    - For padded positions (attention_mask==0), explicitly set to BERT's pad id.
    """
    import torch

    remapped = ids % int(cam_vocab_size)
    if attention_mask is not None:
        remapped = remapped.clone()
        remapped = remapped.masked_fill(attention_mask == 0, int(cam_pad_id))
    return remapped


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def embed_with_mistral_tokenizer_and_camembert(
    texts: List[str],
    mistral_tokenizer_id: str,
    camembert_model_id: str,
    max_length: int,
    device: str,
    batch_size: int,
):
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load only Mistral tokenizer; DO NOT load the Mistral model.
    tok = AutoTokenizer.from_pretrained(mistral_tokenizer_id)
    # Ensure a pad token exists so we can use padding=True
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token  # common practice for decoder LMs
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    # Right padding is conventional when mixing with encoders like BERT
    tok.padding_side = "right"

    # Load CamemBERT model for encoding.
    cam = AutoModel.from_pretrained(camembert_model_id)
    cam.eval()
    cam.to(device)

    # Remap ids into CamemBERT vocab range to avoid index errors.
    cam_vocab_size = int(cam.config.vocab_size)
    cam_pad_id = int(getattr(cam.config, "pad_token_id", 1) or 1)
    all_embs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_enc = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = unsafe_remap_ids_to_camembert(
                batch_enc["input_ids"], batch_enc.get("attention_mask"), cam_vocab_size, cam_pad_id
            )
            attn = batch_enc.get("attention_mask")

            input_ids = input_ids.to(device)
            attn = attn.to(device) if attn is not None else None

            outputs = cam(input_ids=input_ids, attention_mask=attn, return_dict=True)
            # Prefer mean pooling over pooler_output for generality.
            emb = mean_pool(outputs.last_hidden_state, attn)
            all_embs.append(emb.detach().cpu().numpy())

            # free per-batch tensors
            del batch_enc, input_ids, attn, outputs, emb
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return np.vstack(all_embs) if all_embs else np.zeros((0, cam.config.hidden_size), dtype=np.float32)


def plot_umap(embeddings: np.ndarray, labels: Optional[List[int]], save_path: str, seed: int, show: bool):
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(7, 6))
    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=25, alpha=0.8)
    else:
        labels = np.asarray(labels)
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(coords[idx, 0], coords[idx, 1], s=25, alpha=0.85, label=f"class {int(cls)}")
        plt.legend(frameon=False)
    plt.title("UMAP (Mistral tokenizer → CamemBERT encoder)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


class CamemBertBinary:
    """Thin wrapper around CamemBERT + linear 1-logit head with mean pooling.

    We keep the encoder in a separate attribute for save/load clarity.
    """

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel
        self.device = device
        self.encoder = AutoModel.from_pretrained(model_id).to(device)
        hidden = int(self.encoder.config.hidden_size)
        import torch.nn as nn
        self.head = nn.Linear(hidden, 1).to(device)

    def parameters(self):
        for p in self.encoder.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    def zero_grad(self):
        import torch
        self.encoder.zero_grad(set_to_none=True)
        self.head.zero_grad(set_to_none=True)

    def train(self):
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def forward_logits(self, input_ids, attention_mask):
        import torch
        with torch.set_grad_enabled(self.encoder.training or self.head.training):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = mean_pool(out.last_hidden_state, attention_mask)
            logits = self.head(pooled)
        return logits

    @property
    def config(self) -> Dict:
        return {"hidden_size": int(self.encoder.config.hidden_size)}

    def save(self, out_dir: Path, extra: Dict[str, object]):
        import torch, json
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "head_state": self.head.state_dict(),
            "extra": extra,
        }, out_dir / "model.pt")
        (out_dir / "config.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")

    def load_state(self, ckpt_path: Path):
        import torch
        data = torch.load(ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(data["encoder_state"])
        self.head.load_state_dict(data["head_state"])


def train_classifier(
    texts: List[str],
    labels: List[int],
    tok_id: str,
    camembert_id: str,
    max_length: int,
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    out_subdir: Path,
):
    """Fine-tune CamemBERT + linear head with BCEWithLogitsLoss.

    Returns: path to saved model directory and final metrics dict for train/val/test.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    import numpy as np
    from datetime import datetime
    import json, shutil

    class TokDataset(Dataset):
        def __init__(self, X: List[str], y: List[int], tok, max_len: int, cam_vocab: int, cam_pad: int):
            self.X = X
            self.y = y
            self.tok = tok
            self.max_len = max_len
            self.cam_vocab = cam_vocab
            self.cam_pad = cam_pad
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i: int):
            enc = self.tok(
                self.X[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
            ids = unsafe_remap_ids_to_camembert(enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), self.cam_vocab, self.cam_pad)
            return {
                "input_ids": ids.long(),
                "attention_mask": enc["attention_mask"].squeeze(0).long(),
                "label": torch.tensor([float(self.y[i])], dtype=torch.float32),
            }

    # Prepare tokenizer
    tok = AutoTokenizer.from_pretrained(tok_id)
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "right"

    # Create model (CamemBERT + head)
    model = CamemBertBinary(camembert_id, device)
    cam_vocab = int(model.encoder.config.vocab_size)
    cam_pad = int(getattr(model.encoder.config, "pad_token_id", 1) or 1)

    # Train/val/test split (stratified) - mirror earlier logic exactly
    y_arr = np.asarray(labels)
    X_train, X_tmp, y_train, y_tmp = train_test_split(texts, y_arr, test_size=0.15, stratify=y_arr, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    X_test, y_test = X_tmp, y_tmp

    train_ds = TokDataset(X_train, y_train.tolist(), tok, max_length, cam_vocab, cam_pad)
    val_ds = TokDataset(X_val, y_val.tolist(), tok, max_length, cam_vocab, cam_pad)
    test_ds = TokDataset(X_test, y_test.tolist(), tok, max_length, cam_vocab, cam_pad)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Class imbalance handling via pos_weight
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    patience_left = patience
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            yb = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(ids, attn)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            probs = []
            y_all = []
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                yb = batch["label"].to(device)
                logits = model.forward_logits(ids, attn)
                val_losses.append(float(criterion(logits, yb).detach().cpu().item()))
                probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
                y_all.append(yb.cpu().numpy().ravel())
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch}: train_loss={total_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            patience_left = patience
            # Save best snapshot to temp dir
            tmp_dir = out_subdir / "_tmp"
            model.save(tmp_dir, extra={
                "mistral_tokenizer": tok_id,
                "camembert_model": camembert_id,
                "max_length": max_length,
                "threshold": 0.5,
            })
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Load best and evaluate on test
    best_ckpt = out_subdir / "_tmp" / "model.pt"
    if best_ckpt.exists():
        model.load_state(best_ckpt)
    model.eval()
    with torch.no_grad():
        test_probs, test_y = [], []
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            yb = batch["label"].to(device)
            logits = model.forward_logits(ids, attn)
            test_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            test_y.append(yb.cpu().numpy().ravel())
        test_probs = np.concatenate(test_probs) if test_probs else np.zeros((0,))
        test_y = np.concatenate(test_y) if test_y else np.zeros((0,))
        test_preds = (test_probs >= 0.5).astype(int)
        test_f1 = float(f1_score(test_y, test_preds, zero_division=0))
        test_ap = float(average_precision_score(test_y, test_probs)) if len(np.unique(test_y)) > 1 else 0.0
        test_auc = float(roc_auc_score(test_y, test_probs)) if len(np.unique(test_y)) > 1 else 0.0

    # Finalize save directory
    # - Save trained model under models/detect/<YYYY-MM-DD>/<out_name>/
    # - Update models/detect/latest -> models/detect/<YYYY-MM-DD> (symlink if possible; else copy)
    date = datetime.now().strftime("%Y-%m-%d")
    date_root = MODELS_DIR / date
    date_dir = date_root / out_subdir.name
    date_root.mkdir(parents=True, exist_ok=True)

    # Move tmp snapshot to date_dir
    tmp_src = out_subdir / "_tmp"
    if tmp_src.exists():
        # write metrics & config
        (tmp_src / "metrics.json").write_text(json.dumps({
            "test_f1@0.5": test_f1,
            "test_ap": test_ap,
            "test_auc": test_auc,
        }, indent=2), encoding="utf-8")
        # replace dated directory contents
        if date_dir.exists():
            shutil.rmtree(date_dir)
        shutil.copytree(tmp_src, date_dir)

        # Update 'latest' to point at the new date root without writing through an old symlink
        latest_root = MODELS_DIR / "latest"
        try:
            if latest_root.is_symlink():
                latest_root.unlink()
            elif latest_root.exists():
                # Remove existing directory/file to replace with symlink
                if latest_root.is_dir():
                    shutil.rmtree(latest_root)
                else:
                    latest_root.unlink()
            # Point latest -> models/detect/<YYYY-MM-DD>
            latest_root.symlink_to(date_root, target_is_directory=True)
        except Exception:
            # Fallback when symlinks are not available: copy the whole dated tree
            if latest_root.exists():
                shutil.rmtree(latest_root)
            shutil.copytree(date_root, latest_root)
    return {
        "model_dir": str(date_dir),
        "test_f1@0.5": test_f1,
        "test_ap": test_ap,
        "test_auc": test_auc,
    }


def main(argv: Optional[List[str]] = None) -> int:
    _lazy_imports()
    import numpy as np
    import torch

    args = build_argparser().parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data (split:", args.split, ") …")
    texts, labels = select_texts_labels(args.split, args.num_texts, args.seed)

    print("Tokenizing with Mistral tokenizer:", args.mistral_tokenizer)
    print("Encoder:", args.camembert_model)
    print("Device:", args.device)
    embeddings = embed_with_mistral_tokenizer_and_camembert(
        texts=texts,
        mistral_tokenizer_id=args.mistral_tokenizer,
        camembert_model_id=args.camembert_model,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
    )
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    print(f"Embeddings shape: {embeddings.shape}. Computing UMAP and saving to {args.save}...")
    plot_umap(embeddings, labels, args.save, seed=args.seed, show=not args.no_show)
    if args.freeze_for_umap:
        print("UMAP only (freeze_for_umap). Skipping fine-tuning.")
        print("Done.")
        return 0

    print("Training CamemBERT + linear head (no LR) …")
    out_tmp = MODELS_DIR / "_work" / args.out_name
    out_tmp.mkdir(parents=True, exist_ok=True)
    metrics = train_classifier(
        texts=texts,
        labels=labels,
        tok_id=args.mistral_tokenizer,
        camembert_id=args.camembert_model,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        out_subdir=out_tmp,
    )
    print("Saved best model under:", metrics["model_dir"])
    print(
        f"Test AP={metrics['test_ap']:.3f} | AUC={metrics['test_auc']:.3f} | F1@0.5={metrics['test_f1@0.5']:.3f}"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
