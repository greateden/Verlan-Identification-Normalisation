# Detect Pipeline Variants

All four experiments share the same preprocessing before branching into encoder/head configurations. The diagram below merges the paths while labelling which experiments use each branch.

```mermaid
flowchart TB
    A["Input text"] --> N["Normalize text"]
    N --> TOK["SFR-Embedding-Mistral tokenizer"]

    TOK --> ENC_F["Frozen encoder<br/>(Exp A · C)"]
    TOK --> ENC_T["Trainable encoder<br/>(Exp B · D)"]

    ENC_F --> MEAN["Mean pooling"]
    ENC_T --> MEAN
    MEAN --> L2["L2 normalize"]

    %% Heads
    L2 --> LR_A["LogisticRegression<br/>(Exp A)"]
    L2 --> LIN_B["Linear head<br/>(Exp B · encoder trainable)"]
    L2 --> BERT_C["BERT-style head<br/>(Exp C · encoder frozen)"]
    L2 --> BERT_D["BERT-style head<br/>(Exp D · encoder trainable)"]

    LR_A --> SIG
    LIN_B --> SIG
    BERT_C --> SIG
    BERT_D --> SIG

    SIG["Sigmoid threshold 0.5"] --> OUT["Output label"]
```
