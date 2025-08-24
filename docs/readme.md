# VERLAИ: Automatic Recognition & Standardisation of French Verlan

This repository contains the data, code, and experiments for the project  
**Automatic Recognition & Standardisation of French Verlan**, led by Eden Li (University of Otago)  
under the supervision of Lech Szymanski and Veronica Liesaputra.

---

## ⏳ Submission Countdown (NZT)

<!-- DUE:START -->
```text
⏳ Time remaining: 54 days, 06 hours, 22 minutes
Deadline (NZT): 2025-10-18 00:00 NZDT
Deadline (UTC): 2025-10-17 11:00 UTC
```
<!-- DUE:END -->

---

## 🎯 Project Goals

1. **Automatic detection** of verlan tokens in contemporary French text.  
2. **Standardisation** of detected verlan forms into canonical French equivalents.  
3. Build a **reproducible open pipeline** with dataset, models, and evaluation reports.  
4. Conduct a **sociolinguistic analysis** of verlan usage patterns.

Target venues: **VarDial 2026 (ACL)** or **TALN 2025**.

---

## 📆 Milestones

- **May 2025 (done):** Crawled 1M tokens, VDL v0.1 (raw corpus, seed lexicon).  
- **June 2025 (done):** 30k token annotation, rule+lexicon baseline.  
- **July–Aug 2025 (ongoing):** CamemBERT+CRF fine-tuning, GPT-4o few-shot, Mistral-7B tokenizer test.  
- **Sept–Oct 2025:** Final fine-tuning, full evaluation, fairness audit, draft writing.  
- **Nov 2025:** Zenodo packaging (data+code), submission.

---

## 📂 Repository Structure

<!-- TREE:START -->
```text
project-root/
├── configs/
│   └── environment.yml
├── data/
│   ├── predictions/
│   │   ├── invented.csv
│   │   ├── mixed_pred.csv
│   │   ├── standard_only_pred.csv
│   │   └── verlan_only_pred.csv
│   ├── processed/
│   │   ├── verlan_pairs.csv
│   │   └── verlan_test_set.csv
│   └── raw/
│       ├── GazetteerEntries.xlsx
│       ├── Sentences.xlsx
│       ├── invented_verlan.txt
│       ├── mixed_shuffled.txt
│       ├── standard_only.txt
│       └── verlan_only.txt
├── docs/
│   └── readme.md
├── scripts/
│   ├── ci_update_docs.py
│   └── generate-tree.py
└── src/
    ├── EvaluateThreshold.py
    ├── convert.py
    ├── detect.py
    ├── detect_infer.py
    ├── infer.py
    └── testing.py
```
<!-- TREE:END -->

To update manually:
```text
python scripts/generate-tree.py > repo_tree.txt
```

---

## 🚀 Getting Started

1. Setup environment

```bash
conda env create -f environment.yml
conda activate verlan
```

2. Hugging Face login (for models & datasets)

```bash
huggingface-cli login
```

3. Run detection

```bash
python src/detect.py
```

4. Run inference

```bash
python src/detect_infer.py --input data/raw/mixed_shuffled.txt --output data/predictions/mixed_pred.csv
```


---

## 📊 Current Status
- ✅ Data collection + annotation (Gold Corpus v1 ready).
- ✅ Baseline (rules + dictionary).
- 🔄 CamemBERT+CRF fine-tuning (in progress).
- 🔄 GPT-4o few-shot & Mistral-7B tokenizer (testing).
- ⏳ Final evaluation + fairness audit (Sept–Oct 2025).
- ⏳ Draft writing (Sept–Oct 2025).

---

## 📌 Notes
	•	All results will be released on Zenodo with DOI.
	•	Reproducibility ensured via conda environment + fixed random seeds.
