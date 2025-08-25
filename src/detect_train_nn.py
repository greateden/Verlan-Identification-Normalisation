# -*- coding: utf-8 -*-
"""
Experimental neural verlan detector training script.
This script demonstrates multiple techniques requested by the user:
  - Class imbalance handling via Focal Loss and balanced sampling
  - Margin-based ArcFace classification head
  - Optional supervised contrastive learning
  - Lightweight knowledge distillation from rule-based scores
  - Additional character CNN branch
The code is intentionally compact and serves as a reference implementation.
Comments are in English as requested.
"""

import argparse
import os
from pathlib import Path
from typing import List
from sklearn.model_selection import GroupShuffleSplit

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from detect_infer import load_verlan_set, tokenize_basic, has_fuzzy_verlan

# ---------------------------------------------------------------
# Losses and heads
# ---------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss with logits input."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class ArcFace(nn.Module):
    """Binary ArcFace head."""

    def __init__(self, in_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, 1))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = nn.functional.normalize(self.W, dim=0)
        x = nn.functional.normalize(x, dim=1)
        cos = (x @ w).squeeze(1)
        acos = torch.acos(cos.clamp(-0.999999, 0.999999))
        cos_m = torch.cos(acos + self.m)
        logits = torch.where(y.squeeze(1) > 0.5, cos_m, cos) * self.s
        return logits.unsqueeze(1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time logits (NO label, NO margin)."""
        w = nn.functional.normalize(self.W, dim=0)
        x = nn.functional.normalize(x, dim=1)
        cos = (x @ w).squeeze(1)
        return (self.s * cos).unsqueeze(1)


class CharCNN(nn.Module):
    """Simple character-level CNN branch."""

    def __init__(self, vocab_size: int = 256, emb_dim: int = 32,
                 kernels: List[int] = [2, 3, 4], channels: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, channels, k) for k in kernels]
        )
        self.output_dim = channels * len(kernels)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [B, T]
        x = self.emb(char_ids).transpose(1, 2)  # [B, emb_dim, T]
        feats = []
        for conv in self.convs:
            h = nn.functional.relu(conv(x))
            # Global max pooling over the temporal dimension
            pooled = h.max(dim=2)[0]
            feats.append(pooled)
        return torch.cat(feats, dim=1)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss."""
    # features: [B, D] (assumed normalized)
    # labels: [B, 1]
    features = nn.functional.normalize(features, dim=1)
    labels = labels.squeeze(1)
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
    # remove self-pairs
    mask = mask - torch.eye(features.size(0), device=features.device)
    logits = features @ features.T / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits) * (1 - torch.eye(features.size(0), device=features.device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    loss = -mean_log_prob_pos.mean()
    return loss


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for logits."""
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(T)), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

    @torch.no_grad()
    def temperature(self) -> float:
        return float(torch.exp(self.log_T).cpu().item())


# ---------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------

class VerlanDataset(Dataset):
    """Dataset returning token, char, label and rule score."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128, max_char_len: int = 200):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = float(row["label"])
        p_dict = float(row.get("p_dict", 0.0))
        pair_id = int(row.get("pair_id", idx))
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Character ids
        text_bytes = text.lower().encode("utf-8", errors="ignore")[: self.max_char_len]
        char_ids = torch.zeros(self.max_char_len, dtype=torch.long)
        if len(text_bytes) > 0:
            char_ids[: len(text_bytes)] = torch.tensor(list(text_bytes), dtype=torch.long)
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "char_ids": char_ids,
            "label": torch.tensor([label], dtype=torch.float32),
            "p_dict": torch.tensor([p_dict], dtype=torch.float32),
            "pair_id": torch.tensor(pair_id, dtype=torch.long),
        }
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    collated = {k: torch.stack([b[k] for b in batch]) for k in keys}
    return collated


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.max(tpr - fpr)


def scan_f1(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.linspace(0, 1, 101)
    f1_vals = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        f1_vals.append(f1_score(y_true, preds))
    idx = int(np.argmax(f1_vals))
    return thresholds[idx], f1_vals[idx]


def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["char_ids"].to(device),
                labels=None,
                inference=True,
            )
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_score.append(probs)
            y_true.append(labels.cpu().numpy().ravel())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    ap = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    t_star, f1_star = scan_f1(y_true, y_score)
    return {
        "ap": ap,
        "roc_auc": roc,
        "ks": ks,
        "t_star": t_star,
        "f1_at_t": f1_star,
        "scores": y_score,
        "labels": y_true,
    }


def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["char_ids"].to(device),
                labels=None,
                inference=True,
            )
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class Detector(nn.Module):
    def __init__(self, encoder: AutoModel, char_cnn: CharCNN):
        super().__init__()
        self.encoder = encoder
        self.char_cnn = char_cnn
        self.arc = ArcFace(encoder.config.hidden_size + char_cnn.output_dim)

    def forward(self, input_ids, attention_mask, char_ids, labels=None, inference: bool = False):
        # If the encoder is frozen we can run it in no_grad mode to save memory
        enc_requires_grad = any(p.requires_grad for p in self.encoder.parameters())
        if enc_requires_grad:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        char_feat = self.char_cnn(char_ids)
        feat = torch.cat([cls, char_feat], dim=1)
        if inference or labels is None:
            logits = self.arc.infer(feat)
        else:
            logits = self.arc(feat, labels)
        return logits, feat


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_data():
    sent_path = RAW_DIR / "Sentences.xlsx"
    gaz_path = RAW_DIR / "GazetteerEntries.xlsx"
    sent_df = pd.read_excel(sent_path)
    lex = pd.read_excel(gaz_path)
    vset = load_verlan_set(gaz_path)

    def rule_score(text: str) -> float:
        tokens = tokenize_basic(text)
        return float(has_fuzzy_verlan(tokens, vset))

    if "label" not in sent_df.columns:
        vset_surface = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        sent_df["label"] = sent_df["text"].apply(
            lambda s: int(any(t in vset_surface for t in str(s).lower().split()))
        )
    sent_df["p_dict"] = sent_df["text"].apply(rule_score)
    # Optional group-aware split if 'pair_id' column exists (avoid leakage across pairs)
    if "pair_id" in sent_df.columns:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
        tr_idx, held_idx = next(gss.split(sent_df, groups=sent_df["pair_id"]))
        train_df = sent_df.iloc[tr_idx]
        held_df = sent_df.iloc[held_idx]
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43)
        val_idx, rest_idx = next(gss2.split(held_df, groups=held_df["pair_id"]))
        val_df = held_df.iloc[val_idx]
        rest_df = held_df.iloc[rest_idx]
        gss3 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=44)
        calib_idx, test_idx = next(gss3.split(rest_df, groups=rest_df["pair_id"]))
        calib_df = rest_df.iloc[calib_idx]
        test_df = rest_df.iloc[test_idx]
    else:
        all_df = sent_df.sample(frac=1, random_state=42)
        n = len(all_df)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_df = all_df.iloc[:n_train]
        val_df = all_df.iloc[n_train:n_train + n_val]
        rest_df = all_df.iloc[n_train + n_val:]
        n_rest = len(rest_df)
        n_calib = n_rest // 2
        calib_df = rest_df.iloc[:n_calib]
        test_df = rest_df.iloc[n_calib:]
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            calib_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8,
                    help="Training batch size (default: 8 to reduce GPU memory use)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--train_encoder", action="store_true",
                    help="Fine-tune the transformer encoder instead of freezing it")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModel.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="auto", torch_dtype=torch.bfloat16
    )

    if not args.train_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()

    char_cnn = CharCNN()
    model = Detector(enc, char_cnn).to(device)

    train_df, val_df, calib_df, test_df = load_data()
    train_ds = VerlanDataset(train_df, tok, max_len=args.max_len)
    val_ds = VerlanDataset(val_df, tok, max_len=args.max_len)
    calib_ds = VerlanDataset(calib_df, tok, max_len=args.max_len)

    labels_np = train_df["label"].values
    n_pos = np.sum(labels_np == 1)
    n_neg = np.sum(labels_np == 0)
    weights = np.where(labels_np == 1, n_neg / (n_pos + 1e-9), 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler, drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch["label"].to(device)
            logits, feat = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["char_ids"].to(device),
                labels,
            )
            loss_cls = criterion(logits, labels)
            loss_sup = supcon_loss(feat, labels)
            p_dict = batch["p_dict"].to(device)
            kd = nn.functional.mse_loss(torch.sigmoid(logits), p_dict.detach())
            loss = loss_cls + 0.2 * loss_sup + 0.2 * kd
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: {metrics}")
    # ---------- Calibration on calib split ----------
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(VerlanDataset(test_df, tok, max_len=args.max_len),
                             batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    scaler = TemperatureScaler().to(device)
    scaler.train()
    calib_logits, calib_labels = collect_logits(model, calib_loader, device)
    optim_T = torch.optim.LBFGS([scaler.log_T], lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()
    def _closure():
        optim_T.zero_grad()
        loss = bce(scaler(calib_logits.to(device)), calib_labels.to(device))
        loss.backward()
        return loss
    optim_T.step(_closure)
    T_opt = scaler.temperature()
    print(f"[Calibration] Optimal temperature T = {T_opt:.3f}")

    # Decide threshold on VAL (calibrated)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    val_probs = torch.sigmoid(scaler(val_logits.to(device))).cpu().numpy().ravel()
    ts = np.linspace(0, 1, 501)
    f1s = [f1_score(val_labels.numpy().ravel(), (val_probs >= t).astype(int), zero_division=0) for t in ts]
    t_star = float(ts[int(np.argmax(f1s))])
    print(f"[VAL] Chosen threshold t* = {t_star:.3f} (F1={max(f1s):.3f})")

    # Final test report
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_probs = torch.sigmoid(scaler(test_logits.to(device))).cpu().numpy().ravel()
    test_ap = average_precision_score(test_labels.numpy().ravel(), test_probs)
    test_auc = roc_auc_score(test_labels.numpy().ravel(), test_probs)
    test_preds = (test_probs >= t_star).astype(int)
    test_f1 = f1_score(test_labels.numpy().ravel(), test_preds, zero_division=0)
    fpr, tpr, _ = roc_curve(test_labels.numpy().ravel(), test_probs)
    ks = float(np.max(tpr - fpr))
    print(f"[TEST] AUC={test_auc:.3f} | AP={test_ap:.3f} | KS={ks:.3f} | F1@t*={test_f1:.3f}")

    out_dir = PROJECT_ROOT / "models" / "detect" / "latest"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"state_dict": model.state_dict(),
                 "temperature": T_opt,
                 "threshold": t_star}, out_dir / "nn_head.joblib")


if __name__ == "__main__":
    main()

