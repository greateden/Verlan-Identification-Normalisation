"""
Sentence-level Verlan detector using 8-bit encodings from Salesforce/SFR
plus a LogisticRegression CPU head. Uses GPU for embed cost.
"""
import os, random, joblib
import pandas as pd, numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel

# reproducibility
random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_ID = "Salesforce/SFR-Embedding-Mistral"

# load data
print("Loading data …")
sent_df = pd.read_excel("Sentences.xlsx")
lex = pd.read_excel("GazetteerEntries.xlsx")
if "label" not in sent_df.columns:
    vset = set(lex["verlan_form"].str.lower().dropna())
    sent_df["label"] = sent_df["text"].apply(lambda t: int(any(tok in vset for tok in t.lower().split())))
train_df, test_df = train_test_split(sent_df, test_size=0.15, stratify=sent_df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)
print(f"Splits: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

# init encoder GPU 8-bit
tok_enc = AutoTokenizer.from_pretrained(MODEL_ID)
model_enc = AutoModel.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map="auto"
)
model_enc.eval()

def embed_texts(texts, batch_size=32, max_len=512):
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size].tolist()
        enc = tok_enc(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out = model_enc(**{k:v.cuda() for k,v in enc.items()})
        mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs.append(pooled.detach().cpu().numpy())
        print(f"Embedded {i+len(chunk)}/{len(texts)} texts")
    return np.vstack(embs)

# embed & train
print("Embedding train set …")
X_train = embed_texts(train_df["text"])
y_train = train_df["label"].values
print("Embedding val set …")
X_val = embed_texts(val_df["text"])
y_val = val_df["label"].values

print("Training classifier …")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Val results:")
print(classification_report(y_val, clf.predict(X_val)))

print("Embedding test set …")
X_test = embed_texts(test_df["text"])
y_test = test_df["label"].values
print("Test F1:", f1_score(y_test, clf.predict(X_test)))

os.makedirs("verlan-detector", exist_ok=True)
joblib.dump(clf, "verlan-detector/lr_head.joblib")
print("Detect model saved.")
