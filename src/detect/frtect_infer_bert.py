# -*- coding: utf-8 -*-
"""
CamemBERT fine-tuned detector inference (mirrors detect_infer CLI)

- Loads model/tokenizer from models/detect/latest/camembert by default
- Supports single text or batch file (.txt/.csv/.xlsx)
- Optional lexicon gate using GazetteerEntries.xlsx (exact + fuzzy within 1 edit)
- Outputs pred_raw, gate_allow, pred, proba for auditability

Usage examples:
  # Single:
  python -m src.detect.frtect_infer_bert --text "il a fumé un bédo avec ses rebeus"

  # Batch TXT (one per line):
  python -m src.detect.frtect_infer_bert --infile data/raw/mixed_shuffled.txt --outfile data/predictions/mixed_pred.csv

  # Batch XLSX/CSV (reads 'text' column by default):
  python -m src.detect.frtect_infer_bert --infile Sentences.xlsx --xlsx

  # With config (uses threshold, gate, batch_size from configs/detect.yaml if present):
  python -m src.detect.frtect_infer_bert --infile samples.txt --config configs/detect.yaml
"""

import os, sys, argparse, re
from pathlib import Path
import warnings

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from transformers import CamembertTokenizer, CamembertForSequenceClassification
from unidecode import unidecode

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "detect" / "latest" / "camembert"

# Runtime stability/speed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch is not None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------- Lexicon gate (same logic as detect_infer) ----------------
def load_verlan_set(xlsx_path: Path = RAW_DIR / "GazetteerEntries.xlsx") -> set:
    if pd is None:
        return set()
    if not xlsx_path.exists():
        print(f"[gate] WARNING: {xlsx_path} not found. Gate may block more positives.", file=sys.stderr)
        return set()
    df = pd.read_excel(xlsx_path)
    if "verlan_form" not in df.columns:
        return set()
    vset = set(df["verlan_form"].dropna().astype(str).str.lower().tolist())
    return vset


def tokenize_basic(s: str):
    s = unidecode(str(s).lower())
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", s)


def one_edit_apart(a: str, b: str) -> bool:
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = diff = 0
    while i < la and j < lb and diff <= 1:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1
            if la == lb:
                i += 1; j += 1
            else:
                j += 1
    diff += (lb - j) + (la - i)
    return diff <= 1


def has_fuzzy_verlan(tokens, vset: set, max_edit: int = 1) -> bool:
    if not tokens or not vset:
        return False
    if any(t in vset for t in tokens):
        return True
    cand_by_len = {}
    for w in vset:
        cand_by_len.setdefault(len(w), []).append(w)
    for t in tokens:
        if len(t) < 3:
            continue
        for L in (len(t) - 1, len(t), len(t) + 1):
            for w in cand_by_len.get(L, []):
                if one_edit_apart(t, w):
                    return True
    return False


VSET = load_verlan_set()


# ---------------- Core inference ----------------
@torch.inference_mode()
def predict_proba(texts, model_dir: Path, max_len=128, batch_size=64, device: str | None = None):
    tok = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    probs = []
    n = len(texts)
    bs = max(1, int(batch_size))
    for i in range(0, n, bs):
        batch = [str(x) for x in texts[i : i + bs]]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        p = torch.sigmoid(logits).float().cpu().numpy().ravel()
        probs.append(p)
    p1 = np.concatenate(probs)
    return p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="Single input sentence")
    ap.add_argument("--infile", default=None, help="Batch file: .txt / .csv / .xlsx")
    ap.add_argument("--sheet", default=None, help="Sheet name for .xlsx (optional)")
    ap.add_argument("--column", default="text", help="Text column name for CSV/XLSX (default: 'text')")
    ap.add_argument("--outfile", default=None, help="Output file (.txt/.csv/.xlsx)")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold (default 0.5)")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size (default 64)")
    ap.add_argument("--xlsx", action="store_true", help="Force Excel I/O")
    ap.add_argument("--gate_debug", action="store_true", help="Print raw pred and gate decision in single-sentence mode")
    ap.add_argument("--no_gate", action="store_true", help="Disable lexicon gate")
    ap.add_argument("--config", default=None, help="Optional YAML with threshold/batch_size/use_lexicon_gate")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="Path to fine-tuned CamemBERT model dir")
    args = ap.parse_args()

    # Config (optional, reuses fields from configs/detect.yaml if supplied)
    cfg = {}
    if args.config and yaml is not None and Path(args.config).exists():
        with open(args.config) as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}

    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 64))
    use_gate = bool(cfg.get("use_lexicon_gate", True)) and not args.no_gate

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Single sentence path ----
    if args.text:
        p1 = predict_proba([args.text], model_dir, max_len=args.max_len, batch_size=batch_size)
        pred_raw = int(p1[0] >= threshold)
        toks = tokenize_basic(args.text)
        allow = has_fuzzy_verlan(toks, VSET) if use_gate else True
        pred_final = int((pred_raw == 1) and allow)
        if args.gate_debug:
            print(f"[debug] pred_raw={pred_raw}  gate_allow={int(bool(allow))}  tokens={toks}")
        print(f"pred={pred_final}  proba={p1[0]:.4f}")
        sys.exit(0)

    # ---- Batch path ----
    if not args.infile:
        print("Please provide --text or --infile", file=sys.stderr)
        sys.exit(1)

    infile = args.infile
    root, ext = os.path.splitext(infile)

    # Read input table/lines
    if args.xlsx or ext.lower() == ".xlsx":
        df = pd.read_excel(infile, sheet_name=args.sheet) if args.sheet else pd.read_excel(infile)
    elif ext.lower() == ".csv":
        df = pd.read_csv(infile)
    elif ext.lower() == ".txt":
        with open(infile, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]
        df = pd.DataFrame({args.column: lines})
    else:
        print("Unsupported input format (use .txt / .csv / .xlsx)", file=sys.stderr)
        sys.exit(1)

    if args.column not in df.columns:
        print(f"Missing column: {args.column}", file=sys.stderr)
        sys.exit(1)

    texts = df[args.column].astype(str).tolist()
    p1 = predict_proba(texts, model_dir, max_len=args.max_len, batch_size=batch_size)
    pred_raw = (p1 >= threshold).astype(int)

    # Apply lexicon gate per line
    gate_allow = []
    pred_final = []
    for s, r in zip(texts, pred_raw):
        toks = tokenize_basic(s)
        allow = has_fuzzy_verlan(toks, VSET) if use_gate else True
        gate_allow.append(int(bool(allow)))
        pred_final.append(int(r == 1 and allow))

    df_out = df.copy()
    df_out["pred_raw"] = pred_raw
    df_out["gate_allow"] = gate_allow
    df_out["pred"] = pred_final
    df_out["proba"] = p1

    # Output
    if args.outfile:
        out = args.outfile
    else:
        out = f"{root}_pred{ext if ext.lower() in ['.csv', '.xlsx'] else '.csv'}"

    if out.lower().endswith(".xlsx"):
        df_out.to_excel(out, index=False)
    elif out.lower().endswith(".csv"):
        df_out.to_csv(out, index=False)
    elif out.lower().endswith(".txt"):
        with open(out, "w", encoding="utf-8") as f:
            for pr, ga, pf, q in zip(pred_raw, gate_allow, pred_final, p1):
                f.write(f"{pr}\t{ga}\t{pf}\t{q:.6f}\n")
    else:
        out = f"{root}_pred.csv"
        df_out.to_csv(out, index=False)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

