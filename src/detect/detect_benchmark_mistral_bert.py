#!/usr/bin/env python3
"""
Benchmark/Inference for the Mistral-tokenizer → CamemBERT+Linear detector.

Loads the fine-tuned model saved by src/detect/detect_train_mistral_bert.py and either:
 - Benchmarks on the canonical test split (72.25/12.75/15), or
 - Runs inference on an input file and saves predictions.

Tokenization uses the Mistral tokenizer specified in the saved config, with ids
remapped into CamemBERT's vocab range, matching the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "detect"


def _lazy_imports():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel  # noqa: F401
        import pandas as pd  # noqa: F401
    except Exception:
        raise


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark/Infer CamemBERT+Linear (Mistral tokenizer)")
    p.add_argument(
        "--model_dir",
        default=str(MODELS_DIR / "latest" / "camembert_mistral"),
        help="Directory containing model.pt and config.json",
    )
    p.add_argument("--device", default=None, help="cuda or cpu (auto by default)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mode", choices=["benchmark", "infer"], default="benchmark")
    p.add_argument("--infile", default=None, help=".txt/.csv/.xlsx with 'text' column for infer mode")
    p.add_argument(
        "--outfile",
        default=None,
        help=(
            "Output CSV path (infer mode). If omitted, auto-generates under "
            "data/predictions/<YYYY-MM-DD>/<dataset>_<YYYY-MM-DD>_<HHMMSS>_pred.csv"
        ),
    )
    p.add_argument("--out_csv", default=None, help="Optional CSV path to save benchmark test predictions")
    return p


def load_splits(seed: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    sent_path = RAW_DIR / "Sentences_balanced.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    if "label" not in df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        df["label"] = df["text"].apply(has_verlan)
    tr, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=seed)
    tr, va = train_test_split(tr, test_size=0.15, stratify=tr["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def normalize_texts(texts: List[str]) -> List[str]:
    import unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFC", str(s))
        return "".join(ch for ch in s if (unicodedata.category(ch) != "Cc") or ch in "\t\n\r")
    return [_norm(t) for t in texts]


def unsafe_remap_ids_to_camembert(ids, attention_mask, cam_vocab_size: int, cam_pad_id: int):
    import torch
    remapped = ids % int(cam_vocab_size)
    if attention_mask is not None:
        remapped = remapped.clone()
        remapped = remapped.masked_fill(attention_mask == 0, int(cam_pad_id))
    return remapped


class CamemBertBinary:
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
        from torch import no_grad
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            mask = attention_mask.unsqueeze(-1).type_as(out.last_hidden_state)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(pooled)


def _embed_and_predict(texts: List[str], model, mistral_tok_id: str, max_length: int, batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mistral_tok_id)
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "right"
    cam_vocab = int(model.encoder.config.vocab_size)
    cam_pad = int(getattr(model.encoder.config, "pad_token_id", 1) or 1)
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        ids = unsafe_remap_ids_to_camembert(enc["input_ids"], enc.get("attention_mask"), cam_vocab, cam_pad)
        ids = ids.to(device)
        attn = enc["attention_mask"].to(device)
        logits = model.logits(ids, attn)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        probs_all.append(probs)
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


def _read_dataframe(infile: Path):
    import pandas as pd
    if infile.suffix.lower() == ".txt":
        texts = [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame({"text": texts})
    if infile.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if infile.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(infile, sep=sep)
    elif infile.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(infile)
    else:
        try:
            df = pd.read_csv(infile)
        except Exception:
            texts = [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
            return pd.DataFrame({"text": texts})
    # Ensure a 'text' column exists; if absent, duplicate first column as 'text'
    if "text" not in df.columns:
        df = df.copy()
        df["text"] = df.iloc[:, 0].astype(str)
    else:
        df = df.copy()
        df["text"] = df["text"].astype(str)
    return df


def main(argv: List[str] | None = None) -> int:
    _lazy_imports()
    import torch
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

    args = build_argparser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model = CamemBertBinary(cfg["camembert_model"], device)
    model.load_state(ckpt_path)
    model.eval()

    if args.mode == "infer":
        if not args.infile:
            raise SystemExit("--infile is required for infer mode")
        infile = Path(args.infile)
        # Read full dataframe to preserve original columns
        df_in = _read_dataframe(infile)
        texts = normalize_texts(df_in["text"].astype(str).tolist())
        probs = _embed_and_predict(texts, model, cfg["mistral_tokenizer"], cfg["max_length"], args.batch_size, device)
        preds = (probs >= float(cfg.get("threshold", 0.5))).astype(int)

        # Prepare output path if not specified
        if args.outfile is None:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            ts = datetime.now().strftime("%H%M%S")
            base = infile.stem
            out_dir = PROJECT_ROOT / "data" / "predictions" / today
            out_dir.mkdir(parents=True, exist_ok=True)
            outfile = out_dir / f"{base}_{today}_{ts}_pred.csv"
        else:
            outfile = Path(args.outfile)
            outfile.parent.mkdir(parents=True, exist_ok=True)

        # Copy original columns and append prob/pred
        df_out = df_in.copy()
        df_out["prob"] = probs
        df_out["pred"] = preds
        df_out.to_csv(outfile, index=False)
        print(f"Saved predictions to: {outfile}")
        return 0

    # Benchmark mode: evaluate on canonical test split
    train_df, val_df, test_df = load_splits(seed=42)
    texts = normalize_texts(test_df["text"].astype(str).tolist())
    labels = test_df["label"].astype(int).to_numpy()
    probs = _embed_and_predict(texts, model, cfg["mistral_tokenizer"], cfg["max_length"], args.batch_size, device)
    preds = (probs >= float(cfg.get("threshold", 0.5))).astype(int)
    ap = average_precision_score(labels, probs)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, zero_division=0)
    print(f"[TEST] AUC={auc:.3f} | AP={ap:.3f} | F1@0.5={f1:.3f}")

    # Optionally save CSV for plotting and analysis
    out_csv = args.out_csv
    if out_csv is None:
        from datetime import datetime
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        ts = now.strftime("%H%M%S")
        out_dir = PROJECT_ROOT / "data" / "predictions" / today
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = str(out_dir / f"camembert_mistral_test_{today}_{ts}_pred.csv")
    import pandas as pd
    pd.DataFrame({
        "text": texts,
        "prob": probs,
        "pred": preds,
        "label": labels,
    }).to_csv(out_csv, index=False)
    print(f"Saved test predictions to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
