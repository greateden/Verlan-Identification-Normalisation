import os, random, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_TOK_FILE = "GazetteerEntries.xlsx"

# 1) tokenizer
print("Loading tokenizer … GPU mode")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=True)
tok.pad_token = tok.eos_token
if os.path.exists(NEW_TOK_FILE):
    vlist = pd.read_excel(NEW_TOK_FILE)["verlan_form"].dropna().str.lower().unique().tolist()
    added = tok.add_tokens([v for v in vlist if v not in tok.get_vocab()])
    print(f"Added {added} Verlan tokens to vocab.")

# 2) dataset
print("Loading verlan_pairs.csv …")
df = pd.read_csv("verlan_pairs.csv")
ds = Dataset.from_pandas(df)
PROMPT_TMPL = (
    "### Instruction: Convert any Verlan words back to standard French.\n"
    "### Input: {src}\n"
    "### Response: {tgt}"
)
ds = ds.map(lambda ex: {"text": PROMPT_TMPL.format(src=ex["src"], tgt=ex["tgt"])}, remove_columns=list(df.columns))

# 3) model + LoRA (GPU 8-bit)
print("Loading base model 8-bit …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map="auto",
    use_auth_token=True
)
if tok.vocab_size != model.get_input_embeddings().weight.size(0):
    model.resize_token_embeddings(len(tok))

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="mistral-verlan-conv",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    fp16=True,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2
)

print("Starting training … 🏎️")
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tok,
    args=args,
    dataset_text_field="text"
)
trainer.train()
trainer.model.save_pretrained("mistral-verlan-conv")
print("Convert model done.")
