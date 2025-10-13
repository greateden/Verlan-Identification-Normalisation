#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot verlan detection with Mistral-7B.

Generates binary decisions (0/1) for the held-out test split and the slang-heavy
evaluation set without any supervised fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.detect.data_utils import (
    PROJECT_ROOT,
    load_slang_test_set,
    load_verlan_dataset,
)
from src.detect.eval_utils import confusion_from_arrays, save_predictions

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SYSTEM_PROMPT = (
    "You are a linguist who identifies verlan (French reversed syllable slang). "
    "Reply with a single digit: '1' if the sentence contains verlan; otherwise reply '0'. "
    "Do not include extra words."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_id: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer: AutoTokenizer, sentence: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Sentence:\n"
                f"{sentence}\n\n"
                "Does this sentence contain verlan? Reply with one digit (0 or 1)."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_response(raw: str) -> int:
    for ch in raw.strip():
        if ch in {"0", "1"}:
            return int(ch)
    raw_lower = raw.lower()
    if "1" in raw_lower or "yes" in raw_lower or "verlan" in raw_lower:
        return 1
    return 0


def generate_predictions(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sentences: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str,
) -> Tuple[np.ndarray, List[str]]:
    first_param = next(model.parameters())
    device = first_param.device
    preds = []
    raw_outputs = []
    do_sample = temperature > 0
    for idx, sent in enumerate(sentences, start=1):
        prompt = build_prompt(tokenizer, sent, system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_tokens = generated[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        raw_outputs.append(response)
        preds.append(parse_response(response))
        print(f"[{idx}/{len(sentences)}] response='{response}' -> pred={preds[-1]}")
    return np.array(preds, dtype=np.int32), raw_outputs


def load_processed_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df.copy()
        df["label"] = df["label"].astype(int)
        return df
    if "verlan_label" in df.columns:
        df = df.copy()
        df["label"] = df["verlan_label"].astype(int)
        return df
    raise ValueError(f"Dataset {path} does not contain 'label' or 'verlan_label' columns.")


def summarise_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    if len(np.unique(y_true)) > 1:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        f1 = 0.0
    conf = confusion_from_arrays(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1": f1,
        "tn": conf["tn"],
        "fp": conf["fp"],
        "fn": conf["fn"],
        "tp": conf["tp"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot verlan detection with Mistral-7B")
    ap.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)

    _, _, test_df = load_verlan_dataset(args.seed)
    slang_df = load_slang_test_set()
    extra_dataset_paths: Dict[str, Path] = {
        "verlan_test_set": PROJECT_ROOT / "data" / "processed" / "verlan_test_set.csv",
        "verlan_test_set_invented": PROJECT_ROOT / "data" / "processed" / "verlan_test_set_invented.csv",
    }

    tokenizer, model = load_model(args.model_id)

    test_texts = test_df["text"].astype(str).tolist()
    slang_texts = slang_df["text"].astype(str).tolist()

    print(f"Running zero-shot inference on {len(test_texts)} test samples …")
    test_preds, test_responses = generate_predictions(
        tokenizer,
        model,
        test_texts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
    )
    test_gold = test_df["label"].astype(int).to_numpy()
    test_acc = float(accuracy_score(test_gold, test_preds)) if len(np.unique(test_gold)) > 0 else 0.0
    test_f1 = float(f1_score(test_gold, test_preds, zero_division=0)) if len(np.unique(test_gold)) > 1 else 0.0
    test_conf = confusion_from_arrays(test_gold, test_preds)
    try:
        print(classification_report(test_gold, test_preds, digits=3))
    except ValueError:
        # classification_report can fail if a class is missing
        pass
    print(f"Test Accuracy@0.5: {test_acc:.3f}")
    print(f"Test F1@0.5: {test_f1:.3f}")

    print(f"Running zero-shot inference on slang-only set ({len(slang_texts)} samples) …")
    slang_preds, slang_responses = generate_predictions(
        tokenizer,
        model,
        slang_texts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
    )
    slang_gold = slang_df["label"].astype(int).to_numpy()
    slang_acc = float(accuracy_score(slang_gold, slang_preds)) if len(slang_gold) else 0.0
    slang_f1 = float(f1_score(slang_gold, slang_preds, zero_division=0)) if len(np.unique(slang_gold)) > 1 else 0.0
    slang_conf = confusion_from_arrays(slang_gold, slang_preds)
    print(f"Slang test Accuracy@0.5: {slang_acc:.3f}")
    print(f"Slang test F1@0.5: {slang_f1:.3f}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        tag = (args.run_id.strip() or f"seed-{args.seed}")
        out_dir = PROJECT_ROOT / "models" / "detect" / "latest" / "mistral_zeroshot" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    test_pred_path = out_dir / "test_predictions.csv"
    slang_pred_path = out_dir / "slang_predictions.csv"

    save_predictions(
        test_df,
        probs=test_preds.astype(float),
        preds=test_preds,
        out_path=test_pred_path,
        extra={"raw_response": test_responses},
    )
    save_predictions(
        slang_df,
        probs=slang_preds.astype(float),
        preds=slang_preds,
        out_path=slang_pred_path,
        extra={"raw_response": slang_responses},
    )

    extra_metrics: Dict[str, Dict[str, float]] = {}
    extra_summary_rows = []
    for name, path in extra_dataset_paths.items():
        if not path.exists():
            print(f"[WARN] Skipping {name}: missing file {path}")
            continue
        extra_df = load_processed_dataset(path)
        texts = extra_df["text"].astype(str).tolist()
        print(f"Running zero-shot inference on {name} ({len(texts)} samples) …")
        preds, responses = generate_predictions(
            tokenizer,
            model,
            texts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
        )
        gold = extra_df["label"].astype(int).to_numpy()
        metrics = summarise_predictions(gold, preds)
        print(f"{name} Accuracy@0.5: {metrics['accuracy']:.3f}")
        print(f"{name} F1@0.5: {metrics['f1']:.3f}")
        csv_path = out_dir / f"{name}_predictions.csv"
        save_predictions(
            extra_df,
            probs=preds.astype(float),
            preds=preds,
            out_path=csv_path,
            extra={"raw_response": responses},
        )
        extra_metrics[name] = metrics
        extra_summary_rows.append({"dataset": name, **metrics})

    extra_summary_path = None
    if extra_summary_rows:
        extra_summary_path = out_dir / "extra_datasets_summary.csv"
        pd.DataFrame(extra_summary_rows).to_csv(extra_summary_path, index=False)
        print(f"[OK] Wrote {extra_summary_path}")

    meta = {
        "model_id": args.model_id,
        "zero_shot": True,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "system_prompt": args.system_prompt,
        "run_id": args.run_id,
        "out_dir": str(out_dir.relative_to(PROJECT_ROOT) if str(out_dir).startswith(str(PROJECT_ROOT)) else out_dir),
        "metrics": {
            "test_f1@0.5": test_f1,
            "test_acc@0.5": test_acc,
            "test_confusion@0.5": test_conf,
            "slang_test_set@0.5": {
                **slang_conf,
                "accuracy": slang_acc,
                "f1": slang_f1,
            },
        },
        "artifacts": {
            "test_predictions": str(test_pred_path.relative_to(PROJECT_ROOT) if str(test_pred_path).startswith(str(PROJECT_ROOT)) else test_pred_path),
            "slang_predictions": str(slang_pred_path.relative_to(PROJECT_ROOT) if str(slang_pred_path).startswith(str(PROJECT_ROOT)) else slang_pred_path),
        },
    }
    if extra_metrics:
        meta["metrics"]["extra_datasets@0.5"] = extra_metrics
    if extra_summary_path is not None:
        meta.setdefault("artifacts", {})["extra_summary"] = str(
            extra_summary_path.relative_to(PROJECT_ROOT) if str(extra_summary_path).startswith(str(PROJECT_ROOT)) else extra_summary_path
        )
    for name in extra_metrics.keys():
        csv_path = out_dir / f"{name}_predictions.csv"
        meta.setdefault("artifacts", {})[f"{name}_predictions"] = str(
            csv_path.relative_to(PROJECT_ROOT) if str(csv_path).startswith(str(PROJECT_ROOT)) else csv_path
        )
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved zero-shot outputs to {out_dir}")


if __name__ == "__main__":
    main()
