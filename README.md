VERLAИ: Verlan Recognition & Standardisation

Quickstart and key commands to run detection, conversion, and plotting.

Quick Start
- Create env: `conda env create -f configs/environment.yml && conda activate verlan`
- Train detector: `python -m src.detect.detect_train_lr_bert`
- Detect (batch): `python -m src.detect.detect_infer --infile data/raw/mixed_shuffled.txt --outfile data/predictions/mixed_pred.csv --config configs/detect.yaml`
- Convert (single): `python -m src.convert.convert_infer --text "il a fumé un bédo avec ses rebeus" --config configs/convert.yaml`

Plot Probability Histogram
- Draw overlapping histograms for Verlan vs Standard probabilities from a predictions CSV.
- Example (produces docs/results/lr_with_bert_ds_balanced/prob_dist.png):
  `python src/plot/plot_probability_histogram.py --csv data/predictions/2025-08-29/mixed_shuffled_pred.csv --out docs/results/lr_with_bert_ds_balanced/prob_dist.png`
- Options:
  - `--prob-col` and `--label-col` to override auto-detection
  - `--bins` to set number of bins (default 20)

Notes
- Packaging: `src` is a Python package. Prefer running scripts via `python -m src.<module>`.
- Backward compatibility wrappers exist at `src/detect_infer.py` and `src/convert_infer.py`.
- Full project overview and results: see `docs/readme.md`.

