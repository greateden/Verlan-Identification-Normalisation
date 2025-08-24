# -*- coding: utf-8 -*-
"""
Verlan Conversion SFT on Mistral-7B (A4000 16GB Optimized)
- 量化：4-bit NF4  (BitsAndBytesConfig)
- LoRA：B 檔（含 q/k/v/o + gate_proj + up_proj + down_proj）
- 訓練參數：A 檔（安全穩定）
- 其他：packing=True、max_seq_length=768、TF32 開啟、右側 padding、pad_to_multiple_of=8

測過環境（參考）：transformers>=4.41, peft>=0.11, trl>=0.9, bitsandbytes>=0.43, torch>=2.2
"""

import os
import random
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

# ------------------------------ 固定超參/檔名 ------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_TOK_FILE = "GazetteerEntries.xlsx"   # 你的自定詞表（verlan_form）
CSV_FILE     = "verlan_pairs.csv"        # 訓練對齊資料（src,tgt）
OUT_DIR      = "mistral-verlan-conv"
MAX_SEQ_LEN  = 768                       # 16GB 顯存建議 512~1024；取 768 折衷
SEED         = 42

# ------------------------------ 基本環境優化 ------------------------------
# 建議配置（減少碎片／提高吞吐）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True         # Ampere（A4000）啟用 TF32
torch.set_float32_matmul_precision("high")
os.makedirs("logs", exist_ok=True)                   # 讓 `tee logs/*.log` 不再報不存在

# ------------------------------ Tokenizer ------------------------------
print("Loading tokenizer … GPU mode")
# NOTE: use_auth_token 已棄用，這裡仍保留；若你已在 CLI 配置 HF_TOKEN 也可省略
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=True)
tok.padding_side = "right"                            # 避免 fp16 下的溢位/遮罩異常
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 自定義 Verlan 詞元加入
added = 0
if os.path.exists(NEW_TOK_FILE):
    vlist = (
        pd.read_excel(NEW_TOK_FILE)["verlan_form"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )
    # 僅添加 tokenizer 裡沒有的詞
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

# 轉為單列 text，供 SFTTrainer 使用（packing 模式最省 padding）
ds = Dataset.from_pandas(df)
ds = ds.map(
    lambda ex: {"text": PROMPT_TMPL.format(src=ex["src"], tgt=ex["tgt"])},
    remove_columns=list(df.columns),
)

# 防極端超長樣本（雙保險；SFT 仍會裁到 MAX_SEQ_LEN）
def clip_len(x, hard_max=2048):
    t = x["text"]
    return {"text": t[:hard_max]}

ds = ds.map(clip_len)

# ------------------------------ 模型：4-bit 量化 ------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading base model with 4-bit quantization …")
# 如已安裝 flash-attn2，可把 attn_implementation 設為 "flash_attention_2"
# 若未安裝，請移除該參數（會自動用 PyTorch SDPA）
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
    # 回退到 PyTorch SDPA
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )

# 新 token -> 需要擴展 embedding / lm_head（對齊到 8，利於 kernel 對齊）
if added > 0:
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=8)

# QLoRA 必備：對量化模型做訓練前準備（norm/cast/grad 設置）
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # 訓練期關閉 KV cache

# ------------------------------ LoRA（B 檔：含 MLP） ------------------------------
# A：q_proj,k_proj,v_proj,o_proj（輕量）
# B：再加 gate_proj, up_proj, down_proj（效果↑，計算↑）
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

# ------------------------------ TrainingArguments（A 檔：安全穩定） ------------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,     # 小批次 + 梯度累積，穩
    gradient_accumulation_steps=8,     # 等效 batch ↑
    num_train_epochs=1,
    bf16=True,
    fp16=False,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",                  # 先關掉 wandb
    gradient_checkpointing=True,       # 省顯存（必開）
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_32bit",         # 4-bit 配這個更穩
    dataloader_num_workers=2,
    seed=SEED,
)

# ------------------------------ 啟動 SFT（packing=True） ------------------------------
print("Starting training … 🏎️")
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tok,
    args=args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=True,                      # 關鍵：把短樣本打包，padding 浪費大幅下降
)

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
print("Convert model done.")
