# Documentation

## ⏳ Submission Countdown (NZT)

<!-- DUE:START -->
```text
⏳ Time remaining: 34 days, 22 hours, 12 minutes
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
├── docs/
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
├── scripts/
│   ├── ci_update_docs.py
│   ├── generate-tree.py
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
│   │   ├── detect_benchmark_mistral_bert.py
│   │   ├── detect_benchmark_mistral_mistral.py
│   │   ├── detect_infer.py
│   │   ├── detect_train_lr.py
│   │   ├── detect_train_lr_bert.py
│   │   ├── detect_train_lr_simple.py
│   │   ├── detect_train_mistral_bert.py
│   │   ├── detect_train_mistral_mistral_labmachine.py
│   │   └── detect_train_nn.py
│   ├── detect_infer.py
│   ├── evaluate/
│   │   ├── EvaluateThreshold.py
│   │   ├── __init__.py
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
