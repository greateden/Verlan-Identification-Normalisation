# -*- coding: utf-8 -*-
"""
Inference for Verlan → Standard French conversion (with resized vocab)
- 與訓練保持一致：先擴充 tokenizer（讀 GazetteerEntries.xlsx），再 resize embeddings，再載入 LoRA
"""

import os, sys, argparse
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ADAPTER_DIR = PROJECT_ROOT / "mistral-verlan-conv"
NEW_TOK_FILE = RAW_DIR / "GazetteerEntries.xlsx"
MAX_NEW_TOKENS = 96

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROMPT_TMPL = (
    "### Instruction: Convert any Verlan words back to standard French.\n"
    "### Input: {src}\n"
    "### Response:"
)

def load_and_expand_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    added = 0
    if NEW_TOK_FILE.exists():
        vlist = (
            pd.read_excel(NEW_TOK_FILE)["verlan_form"]
            .dropna().astype(str).str.lower().unique().tolist()
        )
        to_add = [v for v in vlist if v not in tok.get_vocab()]
        added = tok.add_tokens(to_add)
        print(f"[infer] Added {added} Verlan tokens to vocab. New size={len(tok)}")
    else:
        print(f"[infer] WARNING: {NEW_TOK_FILE} not found. Vocab will NOT match training if tokens were added.")
    return tok, added

def build_model(adapter_dir: str = ADAPTER_DIR):
    # 4-bit + BF16，與訓練一致
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 先載 tokenizer 並擴充
    tok, _ = load_and_expand_tokenizer()

    # 載 base，再把 embeddings 調整到與 tokenizer 一致（重要：在載入 LoRA 前）
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # 將 base 的 embedding / lm_head resize 到 tokenizer 長度（pad_to_multiple_of=8 與訓練一致）
    base.resize_token_embeddings(len(tok), pad_to_multiple_of=8)

    # 載入 LoRA 權重（此時 shape 就能對上）
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok

@torch.inference_mode()
def generate_once(model, tok, src_text: str,
                  temperature: float = 0.0, top_p: float = 1.0,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    prompt = PROMPT_TMPL.format(src=src_text.strip())
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1]
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    model, tok = build_model(args.adapter)

    if args.text is None:
        print("請用 --text 測一條句子，或擴充腳本支援批量。")
        sys.exit(0)

    print(generate_once(model, tok, args.text))

if __name__ == "__main__":
    main()
