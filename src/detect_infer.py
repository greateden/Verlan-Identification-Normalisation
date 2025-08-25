# -*- coding: utf-8 -*-
"""
Verlan sentence detector inference (with optional lexicon gate)
- Encoder: Salesforce/SFR-Embedding-Mistral (4-bit, BF16)
- Classifier: models/detect/latest/lr_head.joblib
- Hybrid decision: (proba >= threshold) AND (lexicon/fuzzy match) -> 1, else 0
- Batch outputs include gate_allow and pred_raw for auditing.

Usage examples:
  # Single:
  python detect_infer.py --text "il a fumé un bédo avec ses rebeus" --config configs/detect.yaml

  # Batch TXT (one per line):
  python detect_infer.py --infile samples.txt --outfile preds.csv --config configs/detect.yaml

  # Batch XLSX/CSV (reads 'text' column by default):
  python detect_infer.py --infile Sentences.xlsx --xlsx --config configs/detect.yaml
"""

import os, sys, argparse, re
from pathlib import Path
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
from unidecode import unidecode
import warnings

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
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
try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoTokenizer = AutoModel = BitsAndBytesConfig = None
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="The `use_auth_token` argument is deprecated")

MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models" / "detect" / "latest"
HEAD_PATH = MODEL_DIR / "lr_head.joblib"

# --- Runtime knobs for stability/speed on A4000 ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if torch is not None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ---------------- Lexicon gate utilities ----------------
def load_verlan_set(xlsx_path: Path = RAW_DIR / "GazetteerEntries.xlsx") -> set:
    """
    Load verlan surface forms into a lowercase set.
    If the gazetteer is missing, return an empty set (gate will become stricter).
    """
    if not xlsx_path.exists():
        print(f"[gate] WARNING: {xlsx_path} not found. Gate may block more positives.", file=sys.stderr)
        return set()
    df = pd.read_excel(xlsx_path)
    if "verlan_form" not in df.columns:
        return set()
    vset = set(df["verlan_form"].dropna().astype(str).str.lower().tolist())
    return vset

def tokenize_basic(s: str):
    """
    Very light tokenizer: lowercase, de-accent using unidecode, keep alnum and inner apostrophes.
    Reason: lexicon entries are surface tokens; we want full-word matches.
    """
    s = unidecode(str(s).lower())
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", s)

def one_edit_apart(a: str, b: str) -> bool:
    """
    Cheap check for edit distance <= 1.
    We early-exit and avoid full DP for speed; fine for short tokens and small lexicons.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Ensure a is the shorter
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
                # substitute
                i += 1; j += 1
            else:
                # insert/delete on the longer string
                j += 1
    # account for trailing char
    diff += (lb - j) + (la - i)
    return diff <= 1

def has_fuzzy_verlan(tokens, vset: set, max_edit: int = 1) -> bool:
    """
    Gate condition:
      - Exact hit on lexicon tokens, OR
      - Fuzzy hit: any token within <=1 edit from a lexicon entry (length >=3).
    This significantly raises precision by filtering look-alikes.
    """
    if not tokens or not vset:
        return False
    # Exact
    if any(t in vset for t in tokens):
        return True
    # Fuzzy (only for tokens with length >= 3, to reduce spurious matches)
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

# -------------- Encoder (4-bit + BF16) --------------
def load_encoder(model_id: str = MODEL_ID):
    """
    Load the embedding model in 4-bit with BF16 compute.
    If FlashAttention-2 is not available, it falls back to SDPA automatically.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModel.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    return tok, model

def embed_texts(texts, tok, model, max_len=512, batch_size=64):
    """
    Mean-pool over valid tokens with attention mask, then L2-normalize.
    Entire pipeline runs on GPU for speed.
    """
    device = next(model.parameters()).device
    out_list = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [str(x) for x in texts[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hs = model(**enc).last_hidden_state              # [B,T,D]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = (hs * mask).sum(dim=1) / denom         # [B,D]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out_list.append(pooled.detach().float().cpu().numpy())
    return np.vstack(out_list)

def predict_proba(
    texts,
    threshold=0.8,
    max_len=512,
    batch_size=64,
    model_id: str = MODEL_ID,
    head_path: Path = HEAD_PATH,
):
    """
    Get probabilities and raw predictions (before lexicon gate).
    threshold: decision threshold for raw classifier (default 0.80 as per "B").
    """
    tok, enc = load_encoder(model_id)
    clf = joblib.load(head_path)
    X = embed_texts(texts, tok, enc, max_len=max_len, batch_size=batch_size)
    p1 = clf.predict_proba(X)[:, 1]
    pred_raw = (p1 >= threshold).astype(int)
    return pred_raw, p1

if torch is not None:
    class ArcFace(nn.Module):
        def __init__(self, in_features, s=30.0, m=0.20):
            super().__init__()
            self.W = nn.Parameter(torch.randn(in_features, 1))
            nn.init.xavier_uniform_(self.W); self.s, self.m = s, m
        @torch.no_grad()
        def infer(self, x):
            w = nn.functional.normalize(self.W, dim=0)
            x = nn.functional.normalize(x, dim=1)
            cos = (x @ w).squeeze(1)
            return (self.s * cos).unsqueeze(1)

    class CharCNN(nn.Module):
        def __init__(self, vocab_size=256, emb_dim=32, kernels=(2,3,4), channels=64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k) for k in kernels])
            self.out_dim = channels * len(kernels)
        def forward(self, char_ids):
            x = self.emb(char_ids).transpose(1,2)
            feats = [nn.functional.max_pool1d(nn.functional.relu(c(x)), x.size(2)).squeeze(2) for c in self.convs]
            return torch.cat(feats, 1)

    class Detector(nn.Module):
        def __init__(self, encoder, char_cnn):
            super().__init__()
            self.encoder = encoder; self.char = char_cnn
            self.arc = ArcFace(encoder.config.hidden_size + char_cnn.out_dim)
        def infer_logits(self, input_ids, attention_mask, char_ids):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:,0]
            ch  = self.char(char_ids)
            feat = torch.cat([cls, ch], 1)
            return self.arc.infer(feat)  # [B,1]

    class TemperatureScaler(nn.Module):
        def __init__(self, T=1.0):
            super().__init__()
            self.log_T = nn.Parameter(torch.tensor(float(np.log(T)), dtype=torch.float32))
        def forward(self, logits): return logits / torch.exp(self.log_T)
        @torch.no_grad()
        def set_T(self, T): self.log_T.data = torch.tensor(float(np.log(T)))

def make_char_ids(batch, max_char_len=200):
    arr = torch.zeros(len(batch), max_char_len, dtype=torch.long)
    for i, s in enumerate(batch):
        b = s.lower().encode("utf-8", errors="ignore")[:max_char_len]
        if len(b): arr[i,:len(b)] = torch.tensor(list(b), dtype=torch.long)
    return arr

def predict_proba_nn(texts, model_id, head_path, max_len=512, batch_size=64):
    tok, enc = load_encoder(model_id)
    device = next(enc.parameters()).device
    # Build NN modules & load weights/T/thr
    payload = joblib.load(head_path.parent / "nn_head.joblib")
    char = CharCNN().to(device)
    det  = Detector(enc, char).to(device)
    missing, unexpected = det.load_state_dict(payload["state_dict"], strict=False)
    if unexpected: print(f"[warn] unexpected keys: {unexpected}")
    scaler = TemperatureScaler().to(device); scaler.set_T(payload.get("temperature", 1.0))
    thr = float(payload.get("threshold", 0.5))
    # Encode and score
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[i:i+batch_size]]
        encd = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        ch   = make_char_ids(batch).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = det.infer_logits(encd["input_ids"], encd["attention_mask"], ch)
            logits = scaler(logits)
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
    p1 = np.concatenate(all_probs)
    pred_raw = (p1 >= thr).astype(int)
    return pred_raw, p1, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="Single input sentence")
    ap.add_argument("--infile", default=None, help="Batch file: .txt / .csv / .xlsx")
    ap.add_argument("--sheet", default=None, help="Sheet name for .xlsx (optional)")
    ap.add_argument(
        "--column", default="text", help="Text column name for CSV/XLSX (default: 'text')"
    )
    ap.add_argument("--outfile", default=None, help="Output file (.txt/.csv/.xlsx)")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    ap.add_argument("--xlsx", action="store_true", help="Force Excel I/O")
    ap.add_argument(
        "--gate_debug",
        action="store_true",
        help="Print raw pred and lexicon gate decision in single-sentence mode",
    )
    ap.add_argument("--config", default=PROJECT_ROOT / "configs" / "detect.yaml")
    ap.add_argument("--no_gate", action="store_true", help="Disable lexicon gate")
    ap.add_argument("--head", choices=["lr","nn"], default="lr", help="Which head to use for inference")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.5)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 64)
    encoder_id = cfg.get("encoder_id", MODEL_ID)
    model_dir = Path(cfg.get("model_dir", MODEL_DIR))
    head_path = model_dir / "lr_head.joblib"
    use_gate = cfg.get("use_lexicon_gate", True) and not args.no_gate

    # ---- Single sentence path ----
    if args.text:
        if args.head == "lr":
            pred_raw, p1 = predict_proba(
                [args.text], threshold, args.max_len, batch_size, encoder_id, head_path
            )
        else:  # nn
            head_path = model_dir / "nn_head.joblib" 
            pred_raw, p1, threshold = predict_proba_nn(
                [args.text], encoder_id, head_path, args.max_len, batch_size
            )

        toks = tokenize_basic(args.text)
        allow = has_fuzzy_verlan(toks, VSET) if use_gate else True
        pred_final = int(pred_raw[0] == 1 and allow)

        if args.gate_debug:
            print(f"[debug] pred_raw={int(pred_raw[0])}  gate_allow={int(bool(allow))}  tokens={toks}")
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
    if args.head == "lr":
        pred_raw, p1 = predict_proba(texts, threshold, args.max_len, batch_size, encoder_id, head_path)
    else:  # nn
        head_path = model_dir / "nn_head.joblib"  # 忽略 config threshold，使用 nn_head 內建 t*
        pred_raw, p1, threshold = predict_proba_nn(texts, encoder_id, head_path, args.max_len, batch_size)

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
