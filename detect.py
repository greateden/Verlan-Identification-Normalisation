# -*- coding: utf-8 -*-
"""
Sentence-level Verlan detector
- Encoder: Salesforce/SFR-Embedding-Mistral (4-bit inference, BF16 compute)
- Head: CPU LogisticRegression (class_weight balanced)
- Pooling: mean over valid tokens (attention mask), then L2-normalize
- Works on NVIDIA A4000 (16GB)

依賴版本建議：
- torch>=2.2
- transformers>=4.41
- bitsandbytes>=0.43
- scikit-learn>=1.3
- pandas, numpy, joblib, openpyxl
"""

import os, random, joblib, argparse
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ------------------------ 可調參數 ------------------------
MODEL_ID = "Salesforce/SFR-Embedding-Mistral"
SEED = 42

# 預設 batch / 長度（可用命令列覆蓋）
DEF_BATCH = 32
DEF_MAXLEN = 512

# ------------------------ 穩定性/效率設置 ------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_data():
    print("Loading data …")
    sent_df = pd.read_excel("Sentences.xlsx")
    lex = pd.read_excel("GazetteerEntries.xlsx")
    if "label" not in sent_df.columns:
        vset = set(lex["verlan_form"].dropna().astype(str).str.lower().tolist())
        def has_verlan(s: str) -> int:
            toks = str(s).lower().split()
            return int(any(t in vset for t in toks))
        sent_df["label"] = sent_df["text"].apply(has_verlan)
    train_df, test_df = train_test_split(
        sent_df, test_size=0.15, stratify=sent_df["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
    )
    print(f"Splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def load_encoder():
    # 4-bit 推理 + BF16 計算；有裝 flash-attn2 就自動啟用，沒有則回退 SDPA
    bnb_cfg = BitsAndBytesConfig(
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
        model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    return tok, model

@torch.inference_mode()
def embed_texts(texts: pd.Series, tok, model, batch_size=DEF_BATCH, max_len=DEF_MAXLEN):
    """Mean-pool with attention mask, then L2-normalize. 全流程在 GPU 計算。"""
    device = next(model.parameters()).device
    embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        chunk = texts.iloc[i:i+batch_size].astype(str).tolist()
        enc = tok(
            chunk, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        ).to(device)

        # BF16 自動混合精度
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**enc)
            H = out.last_hidden_state                         # [B, T, D] (GPU)
            mask = enc["attention_mask"].unsqueeze(-1)        # [B, T, 1] (GPU)
            mask = mask.to(H.dtype)
            denom = mask.sum(dim=1).clamp(min=1)              # [B, 1]
            pooled = (H * mask).sum(dim=1) / denom            # [B, D]
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        embs.append(pooled.detach().float().cpu().numpy())
        print(f"Embedded {i + len(chunk)}/{total} texts")
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH)
    ap.add_argument("--max_length", type=int, default=DEF_MAXLEN)
    args = ap.parse_args()

    train_df, val_df, test_df = load_data()
    tok, model = load_encoder()

    print("Embedding train set …")
    X_train = embed_texts(train_df["text"], tok, model, args.batch_size, args.max_length)
    y_train = train_df["label"].values

    print("Embedding val set …")
    X_val = embed_texts(val_df["text"], tok, model, args.batch_size, args.max_length)
    y_val = val_df["label"].values

    print("Training classifier …")
    # class_weight 平衡，max_iter 拉高；lbfgs 對中小型樣本+連續特徵表現穩
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,             # scikit LR 本身沒有 n_jobs 參數（避免報錯）
        solver="lbfgs",
        verbose=0,
    )
    clf.fit(X_train, y_train)

    print("Val results:")
    yv_pred = clf.predict(X_val)
    print(classification_report(y_val, yv_pred, digits=3))

    print("Embedding test set …")
    X_test = embed_texts(test_df["text"], tok, model, args.batch_size, args.max_length)
    y_test = test_df["label"].values
    yp = clf.predict(X_test)
    print("Test F1:", f1_score(y_test, yp))

    os.makedirs("verlan-detector", exist_ok=True)
    joblib.dump(clf, "verlan-detector/lr_head.joblib")
    print("Detect model saved.")

if __name__ == "__main__":
    main()
