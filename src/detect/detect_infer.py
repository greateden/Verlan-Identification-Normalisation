"""
Detection inference utilities and CLI.

Pipeline:
- sfr_mistral: Salesforce/SFR-Embedding-Mistral encoder + LR head at models/detect/latest/

Also exposes helper functions used by other modules/tests:
- tokenize_basic(text) -> List[str]
- load_verlan_set(path) -> Set[str]
- has_fuzzy_verlan(tokens, vset) -> bool

Usage examples:
  python -m src.detect.detect_infer \
    --infile data/raw/mixed_shuffled.txt \
    --outfile data/predictions/mixed_shuffled_pred.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ------------------------ Basic helpers exposed to other modules ------------------------
def tokenize_basic(text: str) -> List[str]:
    return [t for t in str(text).lower().strip().split() if t]


def load_verlan_set(gaz_path: Path | str | os.PathLike) -> Set[str]:
    import pandas as pd

    gaz_path = Path(gaz_path)
    df = pd.read_excel(gaz_path)
    if "verlan_form" not in df.columns:
        return set()
    return set(df["verlan_form"].dropna().astype(str).str.lower().tolist())


def has_fuzzy_verlan(tokens: Iterable[str], vset: Set[str]) -> bool:
    """Lightweight heuristic: token is in lexicon or its reverse is.

    This deliberately stays simple to avoid heavy deps during rule scoring.
    """
    for t in tokens:
        t = str(t).lower()
        if t in vset or t[::-1] in vset:
            return True
    return False


# ------------------------ Embedding backends ------------------------
def _embed_sfr_mistral(texts: List[str], max_len: int, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
    bnb = BitsAndBytesConfig(
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
            MODEL_ID, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        )
    except Exception:
        enc = AutoModel.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    enc.eval()
    device = next(enc.parameters()).device

    out = []
    n = len(texts)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            batch = [str(x) for x in texts[i : i + batch_size]]
            inputs = tok(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hs = enc(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
                denom = mask.sum(1).clamp(min=1)
                pooled = (hs * mask).sum(1) / denom
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, enc.config.hidden_size), dtype=np.float32)


# Note: The experimental Mistral-tokenizer -> BERT-encoder path has been removed.


# ------------------------ I/O helpers ------------------------
def _read_texts(infile: Path) -> List[str]:
    import pandas as pd

    if infile.suffix.lower() == ".txt":
        return [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
    # CSV/XLSX support
    if infile.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if infile.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(infile, sep=sep)
    elif infile.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(infile)
    else:
        # Fallback: try csv
        try:
            df = pd.read_csv(infile)
        except Exception:
            # treat as txt
            return [line.strip() for line in infile.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "text" in df.columns:
        return df["text"].astype(str).tolist()
    # fallback to first column
    return df.iloc[:, 0].astype(str).tolist()


def _write_predictions(outfile: Path, texts: List[str], probs: np.ndarray, preds: np.ndarray) -> None:
    import pandas as pd

    df = pd.DataFrame({"text": texts, "prob": probs, "pred": preds})
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)


def _read_threshold(config_path: Path | None, default: float = 0.50) -> float:
    if config_path is None or not config_path.exists():
        return default
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return float(data.get("threshold", default))
    except Exception:
        # very small YAML: threshold: 0.50
        try:
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("threshold"):
                    _, val = line.split(":", 1)
                    return float(val.strip())
        except Exception:
            pass
        return default


# ------------------------ CLI ------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verlan detection inference")
    p.add_argument("--infile", required=True, help="Input .txt/.csv/.xlsx with a 'text' column or one per line")
    p.add_argument("--outfile", required=True, help="Output CSV with columns: text, prob, pred")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "detect.yaml"), help="YAML config with 'threshold'")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)

    # model head locations
    p.add_argument("--head_dir", default=None, help="Directory containing lr_head.joblib; defaults per pipeline")
    p.add_argument("--threshold", type=float, default=None, help="Override threshold from config")
    return p


def main(argv: List[str] | None = None) -> int:
    import joblib

    args = build_argparser().parse_args(argv)
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    config_path = Path(args.config) if args.config else None

    threshold = float(args.threshold) if args.threshold is not None else _read_threshold(config_path)

    # pick head dir
    if args.head_dir is not None:
        head_dir = Path(args.head_dir)
    else:
        head_dir = PROJECT_ROOT / "models" / "detect" / "latest"
    head_path = head_dir / "lr_head.joblib"
    if not head_path.exists():
        raise FileNotFoundError(f"Head not found: {head_path}")
    clf = joblib.load(head_path)

    texts = _read_texts(infile)
    X = _embed_sfr_mistral(texts, max_len=args.max_length, batch_size=args.batch_size)

    # predict probabilities and threshold
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    _write_predictions(outfile, texts, probs, preds)
    print(f"Saved predictions to: {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
