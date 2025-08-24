# -*- coding: utf-8 -*-
"""
Verlan Conversion SFT on Mistral-7B (A4000 16GB Optimized)
- Quantization: 4-bit NF4 (BitsAndBytesConfig)
- LoRA: file B (includes q/k/v/o + gate_proj + up_proj + down_proj)
- Training args: file A (safe and stable)
- Others: packing=True, max_seq_length=768, TF32 enabled, right-side padding, pad_to_multiple_of=8

Tested environment (reference): transformers>=4.41, peft>=0.11, trl>=0.9, bitsandbytes>=0.43, torch>=2.2
"""

import os
import random
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer

# ------------------------------ Fixed hyperparams/filenames ------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
NEW_TOK_FILE = RAW_DIR / "GazetteerEntries.xlsx"   # your custom lexicon (verlan_form)
CSV_FILE     = PROC_DIR / "verlan_pairs.csv"        # training alignment data (src, tgt)
OUT_DIR      = PROJECT_ROOT / "models" / "convert" / "latest" / "mistral-verlan-conv"
MAX_SEQ_LEN  = 768                       # For 16GB VRAM, 512~1024 is recommended; 768 is a compromise
SEED         = 42

# ------------------------------ Basic environment optimizations ------------------------------
# Recommended settings (reduce fragmentation/increase throughput)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True         # Enable TF32 on Ampere (A4000)
torch.set_float32_matmul_precision("high")
os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)                   # Ensure `tee logs/*.log` doesn't complain about missing directory

# ------------------------------ Tokenizer ------------------------------
print("Loading tokenizer … GPU mode")
# NOTE: use_auth_token is deprecated but kept here; if HF_TOKEN is set via CLI you can omit it
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=True)
tok.padding_side = "right"                            # Avoid overflow/mask issues in fp16
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Add custom Verlan tokens
added = 0
if NEW_TOK_FILE.exists():
    vlist = (
        pd.read_excel(NEW_TOK_FILE)["verlan_form"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )
    # Only add tokens not already in the tokenizer
    to_add = [v for v in vlist if v not in tok.get_vocab()]
    added = tok.add_tokens(to_add)
    print(f"Added {added} Verlan tokens to vocab.")

# ------------------------------ Dataset ------------------------------
print("Loading verlan_pairs.csv …")
df = pd.read_csv(CSV_FILE)

PROMPT_TMPL = (
    "### Instruction: Convert any Verlan words back to standard French.\n"
    "### Input: {src}\n"
    "### Response: {tgt}"
)

# Convert to single 'text' column for SFTTrainer (packing mode minimizes padding)
ds = Dataset.from_pandas(df)
ds = ds.map(
    lambda ex: {"text": PROMPT_TMPL.format(src=ex["src"], tgt=ex["tgt"])},
    remove_columns=list(df.columns),
)

# Guard against extremely long samples (double safety; SFT still truncates to MAX_SEQ_LEN)
def clip_len(x, hard_max=2048):
    t = x["text"]
    return {"text": t[:hard_max]}

ds = ds.map(clip_len)

# ------------------------------ Model: 4-bit quantization ------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading base model with 4-bit quantization …")
# If flash-attn2 is installed, set attn_implementation to "flash_attention_2"
# If not installed, remove this parameter (PyTorch SDPA will be used)
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_auth_token=True,
    )
except Exception:
    # Fallback to PyTorch SDPA
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )

# New tokens require expanding embedding / lm_head (align to multiples of 8 for kernel efficiency)
if added > 0:
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=8)

# QLoRA essential: prepare quantized model for training (norm/cast/grad settings)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Disable KV cache during training

# ------------------------------ LoRA (file B: includes MLP) ------------------------------
# A: q_proj, k_proj, v_proj, o_proj (lightweight)
# B: adds gate_proj, up_proj, down_proj (better performance, more compute)
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ------------------------------ TrainingArguments (file A: safe and stable) ------------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,     # Small batch + gradient accumulation for stability
    gradient_accumulation_steps=8,     # Increases effective batch size
    num_train_epochs=1,
    bf16=True,
    fp16=False,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",                  # Disable wandb for now
    gradient_checkpointing=True,       # Save VRAM (must enable)
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_32bit",         # More stable with 4-bit
    dataloader_num_workers=2,
    seed=SEED,
)

# ------------------------------ Start SFT (packing=True) ------------------------------
print("Starting training … 🏎️")
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tok,
    args=args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=True,                      # Key: pack short samples to greatly reduce padding waste
)

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
print("Convert model done.")
