#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_l8_ablation.py
===================
Taguchi L8 (2^4 fractional) ablation for Verlan detection using a frozen
Salesforce/SFR-Embedding-Mistral encoder + Logistic Regression classifier.

Factors (binary):
 1. Pooling: CLS vs MEAN
 2. L2 normalisation (sentence embedding) off vs on
 3. Probability calibration (Platt / sigmoid) off vs on
 4. Decision threshold: fixed 0.5 vs tuned t* (maximize F1 on validation)

Hard-coded L8 design (covering main effects with 8 runs):

 run_id | Pool | L2 | Calib | Threshold
 -------------------------------------
    1   | CLS  | 0  | 0     | 0.5
    2   | CLS  | 0  | 1     | t*
    3   | CLS  | 1  | 0     | t*
    4   | CLS  | 1  | 1     | 0.5
    5   | MEAN | 0  | 0     | t*
    6   | MEAN | 0  | 1     | 0.5
    7   | MEAN | 1  | 0     | 0.5
    8   | MEAN | 1  | 1     | t*

Outputs:
  - results_l8.csv : consolidated metrics table (one row per run)
  - split_indices.json : deterministic splits (train/val/test)
  - cached embeddings in .cache_embeds/ keyed by file hash + encoder + pooling + params
  - stdout pretty table of results

Metrics per run (on test set):
 ap (Average Precision), auc (ROC-AUC), f1_at_0p5, f1_at_tstar, acc_at_tstar, brier,
 ece (10-bin Expected Calibration Error), youdenJ_val (max on val), t_J (argmax J),
 t_star (F1-opt threshold on val), F1_val_at_t_star, optional pair_win_rate.

CLI Example:
 python run_l8_ablation.py \
   --data_path data/raw/Sentences.xlsx --text_col text --label_col label \
   --pairs_csv data/processed/verlan_pairs.csv --max_len 256 --batch_size 32 \
   --load_in_4bit --bnb_quant_type nf4 --compute_dtype bfloat16 --device_map auto

Recommended dependency versions (not enforced):
 torch>=2.2, transformers>=4.41, accelerate>=0.31, bitsandbytes>=0.43, scikit-learn>=1.4,
 pandas>=2.1, numpy>=1.24, tqdm, openpyxl (for .xlsx)

NOTE: This script intentionally avoids any gazetteer lexicon usage; labels are assumed present.
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse
import logging
import hashlib
import unicodedata
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    from transformers import AutoTokenizer, AutoModel  # fallback w/o bitsandbytes
    BitsAndBytesConfig = None  # type: ignore

# --------------------------------------------------------------------------------------
# Constants & Global Settings
# --------------------------------------------------------------------------------------
SEED = 42
ENCODER_NAME = "Salesforce/SFR-Embedding-Mistral"
DEFAULT_MAX_LEN = 256
DEFAULT_BATCH = 32
CACHE_DIR = Path('.cache_embeds')
SPLIT_FILENAME = 'split_indices.json'
RESULTS_CSV = 'results_l8.csv'

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------------------
# Seeding utilities
# --------------------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------------
# Metric helpers (ECE, threshold scanning, Youden's J)
# --------------------------------------------------------------------------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width bins in [0,1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        weight = mask.mean()
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)

def scan_f1_threshold(probs: np.ndarray, labels: np.ndarray, num: int = 1001) -> Tuple[float, float]:
    """Return (best_threshold, best_f1) scanning evenly spaced thresholds in [0,1]."""
    best_t, best_f1 = 0.5, -1.0
    thresholds = np.linspace(0.0, 1.0, num)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1 or (math.isclose(f1, best_f1) and t < best_t):
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1

def youdens_j(probs: np.ndarray, labels: np.ndarray, num: int = 1001) -> Tuple[float, float]:
    """Return (J_max, t_at_Jmax) scanning thresholds."""
    from sklearn.metrics import confusion_matrix
    best_j, best_t = -1.0, 0.5
    thresholds = np.linspace(0.0, 1.0, num)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j or (math.isclose(j, best_j) and t < best_t):
            best_j, best_t = j, float(t)
    return best_j, best_t

# --------------------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    # Keep accents but ensure NFC & strip control characters
    s = unicodedata.normalize('NFC', str(s))
    return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C' or ch in ('\n','\t'))

def load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type; use .xlsx/.xls or .csv")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and/or {label_col} not in dataset columns: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df['text'] = df['text'].astype(str).map(normalize_text)
    # Ensure labels are int {0,1}
    df['label'] = df['label'].astype(int)
    return df.reset_index(drop=True)

# --------------------------------------------------------------------------------------
# Split management
# --------------------------------------------------------------------------------------
def make_or_load_splits(df: pd.DataFrame, seed: int, split_file: Path) -> Dict[str, List[int]]:
    if split_file.exists():
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        return {k: list(map(int, v)) for k,v in splits.items()}
    labels = df['label']
    # First hold out test 15%
    idx = np.arange(len(df))
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        idx, labels, test_size=0.15, stratify=labels, random_state=seed
    )
    # Split train vs val with 15% val of remaining (i.e., 0.1275 global)
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, y_train_val, test_size=0.15, stratify=y_train_val, random_state=seed
    )
    splits = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    return splits

# --------------------------------------------------------------------------------------
# Embedding / Encoder utilities with caching
# --------------------------------------------------------------------------------------
@dataclass
class EmbedConfig:
    model_name: str = ENCODER_NAME
    max_len: int = DEFAULT_MAX_LEN
    batch_size: int = DEFAULT_BATCH
    load_in_4bit: bool = False
    bnb_quant_type: str = 'nf4'
    compute_dtype: str = 'bfloat16'  # 'float16', etc.
    device_map: str = 'auto'

class SentenceEmbedder:
    """Loads HF model once and encodes lists of texts with pooling + caching.

    Cache key fields: data_hash, model_name, pooling, max_len, fourbit flag.
    """
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.device = None
        self._tokenizer = None
        self._model = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self._model is not None:
            return
        logging.info(f"Loading encoder {self.cfg.model_name} (4bit={self.cfg.load_in_4bit})")
        tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant_cfg = None
        if self.cfg.load_in_4bit and BitsAndBytesConfig is not None:
            dtype = getattr(torch, self.cfg.compute_dtype) if hasattr(torch, self.cfg.compute_dtype) else torch.bfloat16
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=self.cfg.bnb_quant_type,
            )
        try:
            self._model = AutoModel.from_pretrained(
                self.cfg.model_name,
                quantization_config=quant_cfg,
                device_map=self.cfg.device_map,
                torch_dtype=torch.bfloat16 if quant_cfg else None,
                attn_implementation='flash_attention_2' if torch.cuda.is_available() else None,
            )
        except Exception as e:  # Fallback without flash attention
            logging.warning(f"Flash attention load failed ({e}); retrying w/out attn impl override")
            self._model = AutoModel.from_pretrained(
                self.cfg.model_name,
                quantization_config=quant_cfg,
                device_map=self.cfg.device_map,
                torch_dtype=torch.bfloat16 if quant_cfg else None,
            )
        self._model.eval()
        self._tokenizer = tok
        self.device = next(self._model.parameters()).device

    def _hash_texts(self, texts: List[str]) -> str:
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode('utf-8', errors='ignore'))
        return h.hexdigest()[:16]

    def _cache_path(self, data_hash: str, pooling: str, l2: bool) -> Path:
        tag = f"{data_hash}_{self.cfg.model_name.replace('/','-')}_{pooling}_L2{int(l2)}_len{self.cfg.max_len}_4bit{int(self.cfg.load_in_4bit)}.npy"
        return CACHE_DIR / tag

    @torch.no_grad()
    def encode(self, texts: List[str], pooling: str = 'CLS', l2: bool = False) -> np.ndarray:
        assert pooling in {'CLS','MEAN'}
        # For cache, we store separate vectors per pooling + L2 flag (because L2 is run after pooling).
        data_hash = self._hash_texts(texts)
        cache_file = self._cache_path(data_hash, pooling, l2)
        if cache_file.exists():
            logging.info(f"Loading cached embeddings: {cache_file.name}")
            return np.load(cache_file)
        self._load()
        assert self._tokenizer is not None and self._model is not None, "Model/tokenizer failed to load"
        tokenizer = self._tokenizer
        model = self._model
        device = self.device

        all_vecs: List[np.ndarray] = []
        bs = self.cfg.batch_size
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        for i in tqdm(range(0, len(texts), bs), desc=f"Embedding ({pooling},L2={int(l2)})", ncols=100):
            batch_texts = texts[i:i+bs]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=self.cfg.max_len, return_tensors='pt').to(device)
            with torch.autocast(device_type='cuda', dtype=autocast_dtype) if (torch.cuda.is_available() and autocast_dtype) else contextlib_null():
                outputs = model(**enc)
                hidden = outputs.last_hidden_state  # [B,T,D]
                if pooling == 'CLS':
                    vec = hidden[:, 0, :]
                else:  # MEAN pooling
                    attn_mask = enc['attention_mask'].unsqueeze(-1)  # [B,T,1]
                    summed = (hidden * attn_mask).sum(dim=1)
                    counts = attn_mask.sum(dim=1).clamp(min=1)
                    vec = summed / counts
                if l2:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                all_vecs.append(vec.float().cpu().numpy())
        arr = np.vstack(all_vecs)
        np.save(cache_file, arr)
        logging.info(f"Saved embeddings cache: {cache_file.name}")
        return arr

# Simple context manager yielding no-op (used when autocast disabled)
class contextlib_null:
    def __enter__(self): return None
    def __exit__(self, *args): return False

# --------------------------------------------------------------------------------------
# Experiment routine per configuration
# --------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    run_id: int
    pooling: str  # 'CLS' or 'MEAN'
    l2: bool
    calibration: bool
    threshold_mode: str  # 'fixed' or 'tune'

@dataclass
class RunResult:
    run_id: int
    pooling: str
    l2: int
    calibration: int
    threshold_mode: str
    t_star: float
    ap: float
    auc: float
    f1_at_0p5: float
    f1_at_tstar: float
    acc_at_tstar: float
    brier: float
    ece: float
    youdenJ_val: float
    t_J: float
    F1_val_at_t_star: float
    pair_win_rate: Optional[float] = None

# Train + evaluate one config
def run_single_config(cfg: RunConfig, embeds: np.ndarray, labels: np.ndarray, splits: Dict[str, List[int]], val_probs_cache: Dict[str, np.ndarray], test_probs_cache: Dict[str, np.ndarray]) -> RunResult:
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    X_train, y_train = embeds[train_idx], labels[train_idx]
    X_val, y_val = embeds[val_idx], labels[val_idx]
    X_test, y_test = embeds[test_idx], labels[test_idx]

    # Base LR
    lr = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1,
        verbose=0,
    )
    lr.fit(X_train, y_train)

    model_key = f"run{cfg.run_id}"  # used if storing calibrated probabilities

    if cfg.calibration:
        # Calibrate using validation set only
        calib = CalibratedClassifierCV(base_estimator=lr, cv='prefit', method='sigmoid')
        calib.fit(X_val, y_val)
        prob_val = calib.predict_proba(X_val)[:,1]
        prob_test = calib.predict_proba(X_test)[:,1]
    else:
        prob_val = lr.predict_proba(X_val)[:,1]
        prob_test = lr.predict_proba(X_test)[:,1]

    # Threshold selection
    if cfg.threshold_mode == 'tune':
        t_star, f1_val_t = scan_f1_threshold(prob_val, y_val)
    else:
        t_star, f1_val_t = 0.5, f1_score(y_val, (prob_val >= 0.5).astype(int), zero_division=0)

    # Youden's J on val
    J_val, t_J = youdens_j(prob_val, y_val)

    # Test metrics
    preds_0p5 = (prob_test >= 0.5).astype(int)
    f1_0p5 = f1_score(y_test, preds_0p5, zero_division=0)

    preds_t = (prob_test >= t_star).astype(int)
    f1_t = f1_score(y_test, preds_t, zero_division=0)
    acc_t = accuracy_score(y_test, preds_t)
    brier = brier_score_loss(y_test, prob_test)
    ece = compute_ece(prob_test, y_test, n_bins=10)

    # Robust metrics with safe handling
    try:
        ap = average_precision_score(y_test, prob_test)
    except Exception:
        ap = float('nan')
    try:
        auc = roc_auc_score(y_test, prob_test)
    except Exception:
        auc = float('nan')

    return RunResult(
        run_id=cfg.run_id,
        pooling=cfg.pooling,
        l2=int(cfg.l2),
        calibration=int(cfg.calibration),
        threshold_mode=('t*' if cfg.threshold_mode=='tune' else '0.5'),
        t_star=float(t_star),
        ap=float(ap),
        auc=float(auc),
        f1_at_0p5=float(f1_0p5),
        f1_at_tstar=float(f1_t),
        acc_at_tstar=float(acc_t),
        brier=float(brier),
        ece=float(ece),
        youdenJ_val=float(J_val),
        t_J=float(t_J),
        F1_val_at_t_star=float(f1_val_t),
    )

# --------------------------------------------------------------------------------------
# Pair-win rate evaluation
# --------------------------------------------------------------------------------------

def load_pairs(pairs_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not pairs_path:
        return None
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs_csv not found: {pairs_path}")
    if pairs_path.suffix.lower() == '.csv':
        df = pd.read_csv(pairs_path)
    elif pairs_path.suffix.lower() in {'.xlsx','.xls'}:
        df = pd.read_excel(pairs_path)
    else:
        raise ValueError("pairs_csv must be .csv or .xlsx")
    expected_cols = {'text_verlan','text_standard'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"pairs_csv must have columns {expected_cols}")
    df['text_verlan'] = df['text_verlan'].astype(str).map(normalize_text)
    df['text_standard'] = df['text_standard'].astype(str).map(normalize_text)
    return df

def compute_pair_win_rate(df_pairs: pd.DataFrame, embedder: SentenceEmbedder, pooling: str, l2: bool, clf_fn) -> float:
    texts_v = df_pairs['text_verlan'].tolist()
    texts_s = df_pairs['text_standard'].tolist()
    embeds_v = embedder.encode(texts_v, pooling=pooling, l2=l2)
    embeds_s = embedder.encode(texts_s, pooling=pooling, l2=l2)
    probs_v = clf_fn(embeds_v)
    probs_s = clf_fn(embeds_s)
    wins = (probs_v > probs_s).mean()
    return float(wins)

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
L8_MATRIX: List[RunConfig] = [
    RunConfig(1, 'CLS',  False, False, 'fixed'),
    RunConfig(2, 'CLS',  False, True,  'tune'),
    RunConfig(3, 'CLS',  True,  False, 'tune'),
    RunConfig(4, 'CLS',  True,  True,  'fixed'),
    RunConfig(5, 'MEAN', False, False, 'tune'),
    RunConfig(6, 'MEAN', False, True,  'fixed'),
    RunConfig(7, 'MEAN', True,  False, 'fixed'),
    RunConfig(8, 'MEAN', True,  True,  'tune'),
]

# --------------------------------------------------------------------------------------
# Pretty table utility
# --------------------------------------------------------------------------------------

def format_table(results: List[RunResult]) -> str:
    cols = [
        'run','pool','L2','calib','thr','t*','AP','AUC','F1@0.5','F1@t*','ACC@t*','Brier','ECE','J_val','t_J','F1_val_t*','pair_win']
    lines = [' | '.join(f"{c:>9}" for c in cols)]
    for r in results:
        line = ' | '.join([
            f"{r.run_id:>9}", r.pooling.rjust(9), f"{r.l2:>9}", f"{r.calibration:>9}", f"{r.threshold_mode:>9}",
            f"{r.t_star:9.4f}", f"{r.ap:9.4f}", f"{r.auc:9.4f}", f"{r.f1_at_0p5:9.4f}", f"{r.f1_at_tstar:9.4f}",
            f"{r.acc_at_tstar:9.4f}", f"{r.brier:9.4f}", f"{r.ece:9.4f}", f"{r.youdenJ_val:9.4f}", f"{r.t_J:9.4f}",
            f"{r.F1_val_at_t_star:9.4f}", f"{(r.pair_win_rate if r.pair_win_rate is not None else float('nan')):9.4f}",
        ])
        lines.append(line)
    return '\n'.join(lines)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L8 ablation for Verlan detection")
    p.add_argument('--data_path', type=str, default='data/raw/Sentences.xlsx')
    p.add_argument('--text_col', type=str, default='text')
    p.add_argument('--label_col', type=str, default='label')
    p.add_argument('--pairs_csv', type=str, default=None, help='Optional pairs file for pair-win rate')
    p.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN)
    p.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    p.add_argument('--load_in_4bit', action='store_true')
    p.add_argument('--bnb_quant_type', type=str, default='nf4')
    p.add_argument('--compute_dtype', type=str, default='bfloat16')
    p.add_argument('--device_map', type=str, default='auto')
    p.add_argument('--log_level', type=str, default='INFO')
    p.add_argument('--out_dir', type=str, default='.')
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')
    set_seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / SPLIT_FILENAME

    # Load data
    data_path = Path(args.data_path)
    df = load_dataset(data_path, args.text_col, args.label_col)
    splits = make_or_load_splits(df, SEED, split_file)

    labels = df['label'].values.astype(int)

    # Prepare embedder
    embed_cfg = EmbedConfig(
        model_name=ENCODER_NAME,
        max_len=args.max_len,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
        bnb_quant_type=args.bnb_quant_type,
        compute_dtype=args.compute_dtype,
        device_map=args.device_map,
    )
    embedder = SentenceEmbedder(embed_cfg)

    # We'll store embeddings per pooling + L2 combination actually required by L8.
    # Distinct combos appearing in matrix:
    pooling_l2_combos = sorted(set((rc.pooling, rc.l2) for rc in L8_MATRIX))

    text_list = df['text'].tolist()
    embeddings_cache: Dict[Tuple[str,bool], np.ndarray] = {}
    for pooling, l2 in pooling_l2_combos:
        embeddings_cache[(pooling,l2)] = embedder.encode(text_list, pooling=pooling, l2=l2)

    # Optional pairs
    pairs_df = load_pairs(Path(args.pairs_csv)) if args.pairs_csv else None

    results: List[RunResult] = []

    for rc in L8_MATRIX:
        logging.info(f"Running config: run_id={rc.run_id} pool={rc.pooling} l2={rc.l2} calib={rc.calibration} thr={rc.threshold_mode}")
        embeds = embeddings_cache[(rc.pooling, rc.l2)]
        res = run_single_config(rc, embeds, labels, splits, {}, {})

        # Pair-win rate if requested: need probability fn (after calibration if any). Refit inside a closure.
        if pairs_df is not None:
            train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
            X_train, y_train = embeds[train_idx], labels[train_idx]
            X_val, y_val = embeds[val_idx], labels[val_idx]
            # Fit again (could refactor to reuse; overhead minimal relative to encoding)
            lr = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs', n_jobs=-1, verbose=0)
            lr.fit(X_train, y_train)
            final_model: Any = lr
            if rc.calibration:
                calib = CalibratedClassifierCV(base_estimator=lr, cv='prefit', method='sigmoid')
                calib.fit(X_val, y_val)
                final_model = calib
            def clf_fn(x: np.ndarray, m=final_model):
                return m.predict_proba(x)[:,1]
            res.pair_win_rate = compute_pair_win_rate(pairs_df, embedder, rc.pooling, rc.l2, clf_fn)
        results.append(res)

    # DataFrame & CSV
    df_rows = []
    for r in results:
        row = {
            'run_id': r.run_id,
            'pooling': r.pooling,
            'l2': r.l2,
            'calibration': r.calibration,
            'threshold_mode': r.threshold_mode,
            't_star': r.t_star,
            'ap': r.ap,
            'auc': r.auc,
            'f1_at_0p5': r.f1_at_0p5,
            'f1_at_tstar': r.f1_at_tstar,
            'acc_at_tstar': r.acc_at_tstar,
            'brier': r.brier,
            'ece': r.ece,
            'youdenJ_val': r.youdenJ_val,
            't_J': r.t_J,
            'F1_val_at_t_star': r.F1_val_at_t_star,
        }
        if r.pair_win_rate is not None:
            row['pair_win_rate'] = r.pair_win_rate
        df_rows.append(row)

    res_df = pd.DataFrame(df_rows).sort_values('run_id')
    csv_path = out_dir / RESULTS_CSV
    res_df.to_csv(csv_path, index=False)
    logging.info(f"Saved results to {csv_path}")

    print('\n=== L8 Results ===')
    print(format_table(results))

    return 0

if __name__ == '__main__':
    sys.exit(main())
