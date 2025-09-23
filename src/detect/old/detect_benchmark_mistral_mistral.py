#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark/Inference for the mistral_mistral detector.

Loads the fine-tuned model saved by detect_train_mistral_mistral.py and either:
 - Benchmarks on the canonical *random (non-stratified)* test split (72.25/12.75/15), or
 - Runs inference on an input file and saves predictions.

example

python -m src.detect.detect_benchmark_mistral_mistral \
    --model_dir models/mistral_mistral --mode infer \
    --infile data/processed/verlan_test_set_invented.csv \
    --outfile data/predictions/invented_shuffled_pred.csv


Tokenization uses the SFR-Embedding-Mistral tokenizer specified in config.json,
matching the training pipeline. No id remapping is needed here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

# Paths aligned with training script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def _lazy_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
        import pandas  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Missing a dependency? {e}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark/Infer for mistral_mistral detector")
    p.add_argument("--model_dir", required=True, help="Directory with model.pt and config.json")
    p.add_argument("--device", default=None, help="cuda or cpu (auto by default)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mode", choices=["benchmark", "infer"], default="benchmark")
    p.add_argument("--infile", default=None, help=".txt/.csv/.xlsx with a 'text' column for infer mode")
    p.add_argument("--outfile", default=None,
                   help=("Output CSV path (infer mode). If omitted, auto-generates under "
                         "data/predictions/<YYYY-MM-DD>/<dataset>_<YYYY-MM-DD>_<HHMMSS>_pred.csv"))
    p.add_argument("--out_csv", default=None, help="Optional CSV path to save benchmark test predictions")
    return p


def _read_raw_tables():
    import pandas as pd
    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path  = RAW_DIR / "GazetteerEntries.xlsx"
    if (not sent_path.exists()) or (not gaz_path.exists()):
        raise FileNotFoundError("Required files not found under data/raw/.")
    df  = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    if "label" not in df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        df["label"] = df["text"].apply(has_verlan)
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def load_splits(seed: int = 42):
    """Fixed random split without stratification (train/val/test = 72.25/12.75/15)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = _read_raw_tables()
    tr_full, te = train_test_split(df, test_size=0.15, random_state=seed, shuffle=True)
    tr, va = train_test_split(tr_full, test_size=0.15, random_state=seed, shuffle=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def normalize_texts(texts):
    import unicodedata
    out = []
    for s in texts:
        s2 = unicodedata.normalize("NFC", str(s))
        s2 = "".join(ch for ch in s2 if (unicodedata.category(ch) != "Cc") or ch in "\t\n\r")
        out.append(s2)
    return out


class MistralBinary:
    """Must mirror the architecture used at training: encoder + linear(1)."""
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel
        self.device = device
        self.encoder = AutoModel.from_pretrained(model_id).to(device)
        hidden = int(self.encoder.config.hidden_size)
        import torch.nn as nn
        self.head = nn.Linear(hidden, 1).to(device)

    def load_state(self, ckpt_path: Path):
        import torch
        data = torch.load(ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(data["encoder_state"])
        self.head.load_state_dict(data["head_state"])

    def eval(self):
        self.encoder.eval(); self.head.eval()

    def logits(self, input_ids, attention_mask):
        import torch
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            mask = attention_mask.unsqueeze(-1).type_as(out.last_hidden_state)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(pooled)


def _tokenizer_for_id(tok_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _embed_and_predict(texts: List[str], model: MistralBinary, tok_id: str, max_length: int, batch_size: int, device: str):
    import torch
    tok = _tokenizer_for_id(tok_id)
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        logits = model.logits(ids, attn)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        probs_all.append(probs)
        del enc, ids, attn, logits
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.concatenate(probs_all) if probs_all else np.zeros((0,))


def _read_texts(infile: Path) -> List[str]:
    import pandas as pd
    if infile.suffix.lower() == ".txt":
        return [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
    if infile.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if infile.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(infile, sep=sep)
    elif infile.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(infile)
    else:
        try:
            df = pd.read_csv(infile)
        except Exception:
            return [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "text" in df.columns:
        return df["text"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def main(argv: List[str] | None = None) -> int:
    _lazy_imports()
    import torch
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

    args = build_argparser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load config + weights
    model_dir = Path(args.model_dir)
    cfg_path  = model_dir / "config.json"
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Validate config keys and provide actionable guidance if mismatched
    if "mistral_model" not in cfg:
        # If this looks like a CamemBERT+Mistral config, point to the right script
        if "camembert_model" in cfg:
            raise SystemExit(
                "This config looks like CamemBERT+Mistral (has 'camembert_model').\n"
                "Use: python -m src.detect.detect_benchmark_mistral_bert --model_dir <camembert_mistral_dir> ..."
            )
        missing = [k for k in ("mistral_model", "mistral_tokenizer", "max_length") if k not in cfg]
        raise SystemExit(
            "config.json missing required keys: " + ", ".join(missing) +
            ". Expected a model saved by detect_train_mistral_mistral.py"
        )

    model = MistralBinary(cfg["mistral_model"], device)
    model.load_state(ckpt_path)
    model.eval()

    if args.mode == "infer":
        if not args.infile:
            raise SystemExit("--infile is required for infer mode")
        infile = Path(args.infile)
        texts = normalize_texts(_read_texts(infile))
        probs = _embed_and_predict(texts, model, cfg["mistral_tokenizer"], cfg["max_length"], args.batch_size, device)
        preds = (probs >= float(cfg.get("threshold", 0.5))).astype(int)

        # Build output path
        outfile = Path(args.outfile) if args.outfile else None
        if outfile is None:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            ts = now.strftime("%H%M%S")
            out_dir = PROJECT_ROOT / "data" / "predictions" / today
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = infile.stem.replace(" ", "_")
            outfile = out_dir / f"mistral_mistral_{stem}_{today}_{ts}_pred.csv"

        # Save alongside original content (when possible)
        import pandas as pd
        df_in = None
        if infile.suffix.lower() in {".csv", ".tsv"}:
            sep = "," if infile.suffix.lower() == ".csv" else "\t"
            df_in = pd.read_csv(infile, sep=sep)
        elif infile.suffix.lower() in {".xlsx", ".xls"}:
            df_in = pd.read_excel(infile)
        if df_in is None:
            # If not a table, create one
            df_out = pd.DataFrame({"text": texts, "prob": probs, "pred": preds})
        else:
            df_out = df_in.copy()
            df_out["prob"] = probs
            df_out["pred"] = preds
        df_out.to_csv(outfile, index=False)
        print(f"Saved predictions to: {outfile}")
        return 0

    # Benchmark mode on canonical test split
    train_df, val_df, test_df = load_splits(seed=42)
    texts  = normalize_texts(test_df["text"].astype(str).tolist())
    labels = test_df["label"].astype(int).to_numpy()
    probs  = _embed_and_predict(texts, model, cfg["mistral_tokenizer"], cfg["max_length"], args.batch_size, device)
    preds  = (probs >= float(cfg.get("threshold", 0.5))).astype(int)

    ap  = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    auc = roc_auc_score(labels, probs)           if len(np.unique(labels)) > 1 else 0.0
    f1  = f1_score(labels, preds, zero_division=0) if len(np.unique(labels)) > 1 else 0.0
    print(f"[TEST] AUC={auc:.3f} | AP={ap:.3f} | F1@0.5={f1:.3f}")

    # Optional CSV dump for analysis
    out_csv = args.out_csv
    if out_csv is None:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        ts = now.strftime("%H%M%S")
        out_dir = PROJECT_ROOT / "data" / "predictions" / today
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = str(out_dir / f"mistral_mistral_test_{today}_{ts}_pred.csv")
    import pandas as pd
    pd.DataFrame({"text": texts, "prob": probs, "pred": preds, "label": labels}).to_csv(out_csv, index=False)
    print(f"Saved test predictions to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
