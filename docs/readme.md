# Documentation

## ⏳ Submission Countdown (NZT)

<!-- DUE:START -->
```text
✅ Deadline passed: -61 days, 01 hours, 58 minutes
Deadline (NZT): 2025-10-18 00:00 NZDT
Deadline (UTC): 2025-10-17 11:00 UTC
```
<!-- DUE:END -->

## 📂 Repository Structure

<!-- TREE:START -->
```text
project-root/
├── README.md
├── configs/
│   ├── convert.yaml
│   ├── detect.yaml
│   └── environment.yml
├── data/
│   ├── predictions/
│   │   ├── 2025-08-29/
│   │   │   └── mixed_shuffled_pred.csv
│   │   ├── 2025-09-02/
│   │   │   └── mixed_shuffled_pred.csv
│   │   ├── 2025-09-03/
│   │   │   ├── mixed_mistral_camembert.csv
│   │   │   ├── mixed_shuffled_pred_lr.csv
│   │   │   └── mixed_shuffled_pred_lr_simple.csv
│   │   ├── 2025-09-08/
│   │   │   ├── invented_shuffled_pred.csv
│   │   │   └── invented_shuffled_pred_NN_mistral_bert.csv
│   │   └── very_old/
│   │       ├── invented.csv
│   │       ├── mixed_pred.csv
│   │       ├── standard_only_pred.csv
│   │       └── verlan_only_pred.csv
│   ├── processed/
│   │   ├── slang_test_set.csv
│   │   ├── verlan_pairs.csv
│   │   ├── verlan_test_set.csv
│   │   └── verlan_test_set_invented.csv
│   └── raw/
│       ├── GazetteerEntries.xlsx
│       ├── Sentences.xlsx
│       ├── Sentences_balanced.xlsx
│       ├── invented_verlan_mixed_paired.txt
│       ├── mixed_shuffled.txt
│       ├── standard_only.txt
│       └── verlan_only.txt
├── detect_train_lr_20250917_141908.out
├── docs/
│   ├── detect_pipeline_flowchart.md
│   ├── detect_pipeline_flowchart.pdf
│   ├── readme.md
│   └── results/
│       ├── l8/
│       │   ├── invented.csv
│       │   └── random.csv
│       ├── lr_ds_balanced/
│       │   ├── prob_dist.png
│       │   └── prob_dist_invented.png
│       ├── lr_ds_balanced_simple/
│       │   └── prob_dist.png
│       ├── lr_with_bert_ds_balanced/
│       │   ├── embedding_space_tsne_old.png
│       │   ├── embedding_space_umap_old.png
│       │   ├── note.md
│       │   └── prob_dist.png
│       ├── lr_with_bert_ds_imbalanced/
│       │   ├── embedding_space_umap.png
│       │   └── prob_dist.png
│       ├── mistral_bert_ds_balanced/
│       │   ├── mistral_bert_umap.png
│       │   ├── prob_dist.png
│       │   └── prob_dist_invented.png
│       ├── mistral_mistral_ds_balanced/
│       │   └── mistral_mistral_umap.png
│       └── only_lr_no_bert_ds_imbalance/
│           ├── embedding_space_pca.png
│           ├── embedding_space_tsne.png
│           ├── embedding_space_umap.png
│           ├── prob_dist_post.png
│           └── prob_dist_pre.png
├── envs/
│   └── aoraki-verlan-e2e.yml
├── experiment_results/
│   ├── ChatGPT 5 Codex High.xlsx
│   ├── E2E+BERT/
│   │   ├── seed-1/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-10/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-11/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-12/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-13/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-14/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-15/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-16/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-17/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-18/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-19/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-2/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-20/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-3/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-4/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-5/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-6/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-7/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-8/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-9/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── trials_summary.csv
│   │   └── trials_summary.json
│   ├── E2E+LR/
│   │   ├── seed-1/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-10/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-11/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-12/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-13/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-14/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-15/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-16/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-17/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-18/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-19/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-2/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-20/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-3/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-4/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-5/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-6/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-7/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-8/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-9/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── trials_summary.csv
│   │   └── trials_summary.json
│   ├── Frozen+BERT/
│   │   ├── seed-1/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-10/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-11/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-12/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-13/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-14/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-15/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-16/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-17/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-18/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-19/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-2/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-20/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-3/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-4/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-5/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-6/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-7/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-8/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-9/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── trials_summary.csv
│   │   └── trials_summary.json
│   ├── Frozen+LR/
│   │   ├── seed-1/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-10/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-11/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-12/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-13/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-14/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-15/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-16/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-17/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-18/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-19/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-2/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-20/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-3/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-4/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-5/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-6/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-7/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-8/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── seed-9/
│   │   │   ├── extra_datasets_summary.csv
│   │   │   ├── lr_head.joblib
│   │   │   ├── meta.json
│   │   │   ├── slang_predictions.csv
│   │   │   ├── test_predictions.csv
│   │   │   ├── verlan_test_set_invented_predictions.csv
│   │   │   └── verlan_test_set_predictions.csv
│   │   ├── trials_summary.csv
│   │   └── trials_summary.json
│   └── mistral zeroshot/
│       ├── seed-42/
│       │   ├── extra_datasets_summary.csv
│       │   ├── meta.json
│       │   ├── slang_predictions.csv
│       │   ├── test_predictions.csv
│       │   ├── verlan_test_set_invented_predictions.csv
│       │   └── verlan_test_set_predictions.csv
│       ├── summary.csv
│       └── summary.json
├── final report/
│   ├── All_4settings_results.csv
│   ├── Summary_4settings.csv
│   ├── good example papers/
│   │   ├── 2024.acl-srw.27.pdf
│   │   ├── K19-1082.pdf
│   │   ├── Marxen_Master_Thesis_2023_08_18.pdf
│   │   ├── translation.md
│   │   └── verlanMla-LEVERLANOU-1991.pdf
│   └── report folder/
│       ├── COSC385_YitianLiEden_Report_Final.pdf
│       ├── Consent Form.pdf
│       ├── EDITABLE info sheet and consent form.docx
│       ├── Info Sheet.pdf
│       ├── Verlan Lexicon - Eléonore.pdf
│       ├── Verlan Lexicon - Lua.pdf
│       ├── cosc4x0style.sty
│       ├── figures/
│       │   ├── Accuracy_distribution_4settings.png
│       │   ├── Artificial Analysis Intelligence Index (14 Oct '25) .png
│       │   ├── CamemBERT.png
│       │   ├── aoraki.png
│       │   ├── deepl_alt_text.png
│       │   ├── deepl_verlan.png
│       │   ├── google_verlan.png
│       │   ├── historical_verlan_comparison.png
│       │   ├── historical_vs_slang_tradeoff.png
│       │   ├── invented_recall_variance.png
│       │   ├── invented_relative_improvement.png
│       │   ├── invented_verlan_comparison.png
│       │   ├── logo.png
│       │   ├── mistral-logo.png
│       │   ├── mistral_bert_umap.png
│       │   ├── slang_controls_comparison.png
│       │   └── slang_fp_vs_main_fp.png
│       ├── gkbib_unsrt.bst
│       ├── myreport V0.pdf
│       ├── myreport V1.1.pdf
│       ├── myreport V1.2.1.pdf
│       ├── myreport V1.2.pdf
│       ├── myreport V1.3.pdf
│       ├── myreport.aux
│       ├── myreport.fdb_latexmk
│       ├── myreport.fls
│       ├── myreport.log
│       ├── myreport.out
│       ├── myreport.synctex.gz
│       ├── myreport.tex
│       ├── myreport.toc
│       ├── myreport_LS.pdf
│       └── myreport_VL.pdf
├── logs/
│   ├── 01frozen_mistral+lr_20250917_141908.out
│   ├── 02e2e_mistral+lr_2689334.out
│   ├── 03frozen_mistral+bert_2704327.out
│   ├── 04e2e_mistral+bert_2704317.out
│   ├── dcgm-2689334.out
│   ├── dcgm-2704317.out
│   ├── dcgm-2704327.out
│   ├── e2e_mistral+bert_2704317.err
│   ├── frozen_mistral+bert_2704327.err
│   └── verlan-bert-e2e-2704317.err
├── models/
│   ├── convert/
│   │   ├── 2025-08-20/
│   │   │   └── mistral-verlan-conv/
│   │   └── latest/
│   │       └── mistral-verlan-conv/
│   └── detect/
│       ├── 2025-08-24 LR/
│       │   └── lr_head.joblib
│       ├── 2025-08-29/
│       │   └── lr_head.joblib
│       ├── 2025-09-02/
│       │   └── lr_head.joblib
│       ├── 2025-09-03/
│       │   └── lr_simple/
│       │       └── lr_head.joblib
│       └── latest/
│           └── lr_head.joblib
├── myreport.aux
├── myreport.fdb_latexmk
├── myreport.fls
├── myreport.log
├── myreport.toc
├── myreport.xdv
├── report_text.txt
├── scripts/
│   ├── aoraki/
│   │   ├── create_env.sh
│   │   ├── mistral_zeroshot.slurm
│   │   ├── submit_bert.sh
│   │   ├── submit_bert_e2e.sh
│   │   ├── submit_e2e.sh
│   │   ├── submit_lr.sh
│   │   ├── submit_mistral_zeroshot.sh
│   │   ├── train_bert.slurm
│   │   ├── train_bert_e2e.slurm
│   │   ├── train_e2e.slurm
│   │   └── train_lr.slurm
│   ├── ci_update_docs.py
│   ├── generate-tree.py
│   ├── plot_report_figures.py
│   ├── run_detect_train_lr_batch.py
│   └── run_l8_ablation.py
├── split_indices.json
├── src/
│   ├── __init__.py
│   ├── convert/
│   │   ├── __init__.py
│   │   ├── convert_infer.py
│   │   └── convert_train.py
│   ├── convert_infer.py
│   ├── detect/
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── detect_eval_mistral_zeroshot.py
│   │   ├── detect_infer.py
│   │   ├── detect_train_bert.py
│   │   ├── detect_train_bert_e2e.py
│   │   ├── detect_train_lr.py
│   │   ├── detect_train_lr_e2e.py
│   │   ├── detect_train_lr_simple.py
│   │   ├── eval_utils.py
│   │   └── old/
│   │       ├── detect_benchmark_mistral_bert.py
│   │       ├── detect_benchmark_mistral_mistral.py
│   │       ├── detect_train_lr_bert.py
│   │       ├── detect_train_mistral_bert.py
│   │       ├── detect_train_mistral_mistral_labmachine.py
│   │       └── detect_train_nn.py
│   ├── detect_infer.py
│   ├── evaluate/
│   │   ├── EvaluateThreshold.py
│   │   ├── __init__.py
│   │   ├── aggregate_slang_confusion.py
│   │   ├── calibration.py
│   │   ├── evaluate_threshold.py
│   │   └── utils.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── run_l8_ablation.py
│   └── plot/
│       ├── __init__.py
│       ├── plot_probability_histogram.py
│       ├── plot_verlan_hist.py
│       └── visualize_embeddings.py
└── tests/
    ├── run_l8_ablation.py
    ├── test_convert_infer.py
    ├── test_detect_infer.py
    └── test_tokenization.py
```
<!-- TREE:END -->
