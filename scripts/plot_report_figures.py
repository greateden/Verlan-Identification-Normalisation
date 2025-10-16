#!/usr/bin/env python3
"""Generate the figures used in the final report.

The script re-creates all plots that have been incrementally crafted inside the
CLI.  It relies on the cached experiment artefacts stored under
``experiment_results`` and writes PNGs into ``final report/report folder/figures``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiment_results"
FIG_DIR = ROOT / "final report" / "report folder" / "figures"

# Consistent ordering for bars/markers across plots
TRAINED_MODELS = ["Frozen+LR", "Frozen+BERT", "E2E+LR", "E2E+BERT"]
DISPLAY_LABELS = TRAINED_MODELS + [
    "Mistral-7B Zero Shot",
    "GPT-5 Codex (High) Zero Shot",
]


def _load_trials(model: str) -> pd.DataFrame:
    """Load the per-seed trial summary for a trained model."""
    csv_path = RESULTS_DIR / model / "trials_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing trials_summary for {model}: {csv_path}")
    trials = pd.read_csv(csv_path)
    return trials


def _load_targeted_predictions(
    model: str,
    seed_dir: Path,
    dataset: str,
    variant: str,
) -> pd.DataFrame:
    """Load targeted predictions for a given seed."""
    file_map = {
        "historical": "verlan_test_set_predictions.csv",
        "invented": "verlan_test_set_invented_predictions.csv",
        "slang": "slang_predictions.csv",
    }
    path = seed_dir / file_map[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Missing targeted predictions for {model}: {path}")
    df = pd.read_csv(path)
    return df[df["variant"] == variant].copy()


def _load_zero_shot_targeted(
    model_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Load targeted suites for a zero-shot run."""
    return {
        "historical": pd.read_csv(model_dir / "verlan_test_set_predictions.csv"),
        "invented": pd.read_csv(
            model_dir / "verlan_test_set_invented_predictions.csv"
        ),
        "slang": pd.read_csv(model_dir / "slang_predictions.csv"),
    }


def _compute_confusion_slice(
    df: pd.DataFrame,
    total: int,
    positive_slice: bool,
) -> Tuple[int, int, int, int, float]:
    """Return (tp, fn, tn, fp, metric) for a targeted slice."""
    labels = df["label"].astype(int)
    preds = df["pred_label"].astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    if positive_slice:
        metric = tp / total
    else:
        metric = tn / total
    return tp, fn, tn, fp, metric


def _load_chatgpt_suite() -> pd.DataFrame:
    """Aggregate GPT-5 Codex (High) predictions across targeted workbooks."""
    xls = RESULTS_DIR / "ChatGPT 5 Codex High.xlsx"
    frames: Iterable[pd.DataFrame] = []
    for sheet, label_col in [
        ("verlan_test_set", "label"),
        ("verlan_test_set_invented", "label"),
        ("slang_test_set", "verlan_label"),
    ]:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = df.rename(columns={label_col: "label"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _plot_accuracy_distribution():
    """Regenerate the double box plot with zero-shot markers."""
    metric_names = {"F1": "test_f1@0.5", "Accuracy": "test_acc@0.5"}
    data = {metric: [] for metric in metric_names}
    for model in TRAINED_MODELS:
        trials = _load_trials(model)
        for metric, column in metric_names.items():
            data[metric].append(trials[column].values)

    # zero-shot metrics
    mistral_summary = pd.read_csv(RESULTS_DIR / "mistral zeroshot" / "summary.csv").iloc[
        0
    ]
    zero_metrics = {
        "Mistral-7B Zero Shot": {
            "F1": float(mistral_summary["test_f1@0.5"]),
            "Accuracy": float(mistral_summary["test_acc@0.5"]),
        }
    }
    chat_df = _load_chatgpt_suite()
    chat_labels = chat_df["label"].astype(int)
    chat_preds = chat_df["Output from GPT"].astype(int)
    zero_metrics["GPT-5 Codex (High) Zero Shot"] = {
        "F1": float(f1_score(chat_labels, chat_preds)),
        "Accuracy": float(accuracy_score(chat_labels, chat_preds)),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    positions = np.arange(1, len(TRAINED_MODELS) + 1)

    for ax, metric in zip(axes, metric_names.keys()):
        values = [data[metric][i] for i in range(len(TRAINED_MODELS))]
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="^",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
            ),
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(
                marker="o", markersize=4, markerfacecolor="gray", alpha=0.6
            ),
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for i, seed_values in enumerate(values):
            ax.scatter(
                np.random.normal(positions[i], 0.06, len(seed_values)),
                seed_values,
                color=colors[i],
                alpha=0.7,
                s=18,
                edgecolors="none",
            )

        zero_labels = list(zero_metrics.keys())
        zero_positions = [positions[-1] + i + 1 for i in range(len(zero_labels))]
        markers = ["D", "s"]
        zero_colors = ["#9467bd", "#8c564b"]
        for x, label, marker, color in zip(
            zero_positions, zero_labels, markers, zero_colors
        ):
            ax.scatter(
                x,
                zero_metrics[label][metric],
                marker=marker,
                color=color,
                s=80,
                edgecolors="black",
                label=label,
            )
        ax.set_xticks(np.concatenate([positions, zero_positions]))
        ax.set_xticklabels(TRAINED_MODELS + zero_labels, rotation=20, ha="right")
        ax.set_title(f"{metric} Distribution")
        ax.set_ylabel(metric)
        ax.set_ylim(0.4, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(zero_metrics),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "F1 and Accuracy Distributions Across Training Regimes with Zero-Shot References",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG_DIR / "Accuracy_distribution_4settings.png", dpi=300)
    plt.close(fig)


def _collect_targeted_metrics():
    """Gather targeted metrics and auxiliary statistics."""
    metrics = {
        "Slang specificity": {},
        "Historical recall": {},
        "Invented recall": {},
    }
    stds = {key: {} for key in metrics}
    totals = {
        "Slang specificity": 25,
        "Historical recall": 29,
        "Invented recall": 25,
    }
    variants = {
        "Slang specificity": "slang",
        "Historical recall": "verlan",
        "Invented recall": "verlan",
    }
    positive_slice = {
        "Slang specificity": False,
        "Historical recall": True,
        "Invented recall": True,
    }

    for model in TRAINED_MODELS:
        per_seed_values = {metric: [] for metric in metrics}
        for seed_dir in sorted((RESULTS_DIR / model).glob("seed-*")):
            for metric in metrics:
                subset = _load_targeted_predictions(
                    model=model,
                    seed_dir=seed_dir,
                    dataset=metric.split()[0].lower(),
                    variant=variants[metric],
                )
                _, _, _, _, score = _compute_confusion_slice(
                    subset,
                    total=totals[metric],
                    positive_slice=positive_slice[metric],
                )
                per_seed_values[metric].append(score)
        for metric, values in per_seed_values.items():
            values = np.array(values)
            metrics[metric][model] = values.mean()
            stds[metric][model] = values.std(ddof=0)

    # Zero-shot: Mistral + GPT
    mistral_targeted = _load_zero_shot_targeted(
        RESULTS_DIR / "mistral zeroshot" / "seed-42"
    )
    for metric in metrics:
        dataset = metric.split()[0].lower()
        df = mistral_targeted[dataset]
        df = df[df["variant"] == variants[metric]]
        _, _, _, _, score = _compute_confusion_slice(
            df, totals[metric], positive_slice[metric]
        )
        metrics[metric]["Mistral-7B Zero Shot"] = score

    chat_dfs = {
        "historical": pd.read_excel(
            RESULTS_DIR / "ChatGPT 5 Codex High.xlsx", sheet_name="verlan_test_set"
        ),
        "invented": pd.read_excel(
            RESULTS_DIR / "ChatGPT 5 Codex High.xlsx",
            sheet_name="verlan_test_set_invented",
        ),
        "slang": pd.read_excel(
            RESULTS_DIR / "ChatGPT 5 Codex High.xlsx", sheet_name="slang_test_set"
        ),
    }
    for metric in metrics:
        dataset = metric.split()[0].lower()
        df = chat_dfs[dataset]
        label_col = "label" if "label" in df.columns else "verlan_label"
        df = df[df["variant"] == variants[metric]].rename(
            columns={label_col: "label"}
        )
        _, _, _, _, score = _compute_confusion_slice(
            df, totals[metric], positive_slice[metric]
        )
        metrics[metric]["GPT-5 Codex (High) Zero Shot"] = score

    return metrics, stds


def _plot_slice_bars(metrics: Dict[str, Dict[str, float]]):
    """Generate the three slice bar charts (historical, invented, slang)."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    titles = {
        "Historical recall": "Historical verlan recall across models",
        "Invented recall": "Invented verlan recall across models",
        "Slang specificity": "Slang control specificity across models",
    }
    for metric, title in titles.items():
        values = [metrics[metric].get(label, np.nan) for label in DISPLAY_LABELS]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(
            DISPLAY_LABELS, [v * 100 if not np.isnan(v) else np.nan for v in values], color=colors
        )
        finite_vals = [v for v in values if not np.isnan(v)]
        if finite_vals:
            min_val = min(finite_vals) * 100
            max_val = max(finite_vals) * 100
            span = max_val - min_val
            padding = max(2, span * 0.1)
            lower = max(0, min_val - padding)
            upper = min(100, max_val + padding)
        else:
            lower, upper = 0, 100
        ax.set_ylim(lower, upper)
        ax.set_ylabel("Accuracy / Recall (%)")
        ax.set_title(title)
        ax.set_xticklabels(DISPLAY_LABELS, rotation=20, ha="right")
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 100 + (upper - lower) * 0.02,
                f"{val * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        filename = metric.split()[0].lower() + "_verlan_comparison.png"
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)


def _plot_tradeoff(metrics: Dict[str, Dict[str, float]]):
    """Plot historical recall vs. slang specificity scatter."""
    x_vals = [metrics["Historical recall"].get(label, np.nan) for label in DISPLAY_LABELS]
    y_vals = [metrics["Slang specificity"].get(label, np.nan) for label in DISPLAY_LABELS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for label, x, y, color in zip(DISPLAY_LABELS, x_vals, y_vals, colors):
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x * 100, y * 100, s=90, color=color, edgecolors="black")
        ax.text(x * 100 + 0.3, y * 100 + 0.5, label.split()[0], fontsize=8)
    ax.set_xlabel("Historical verlan recall (%)")
    ax.set_ylabel("Slang specificity (%)")
    ax.set_title("Recall vs. Slang Specificity Trade-off")
    ax.grid(alpha=0.3)
    ax.set_xlim(68, 82)
    ax.set_ylim(40, 85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "historical_vs_slang_tradeoff.png", dpi=300)
    plt.close(fig)


def _plot_invented_variance(metrics: Dict[str, Dict[str, float]], stds: Dict[str, Dict[str, float]]):
    """Plot mean invented recall with std error bars + zero-shot markers."""
    means = [metrics["Invented recall"][model] for model in TRAINED_MODELS]
    std_values = [stds["Invented recall"][model] for model in TRAINED_MODELS]
    x_pos = np.arange(len(TRAINED_MODELS))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        x_pos,
        np.array(means) * 100,
        yerr=np.array(std_values) * 100,
        capsize=5,
        color="#1f77b4",
        alpha=0.6,
    )
    for x, mean, std in zip(x_pos, means, std_values):
        ax.text(x, mean * 100 + std * 100 + 1.5, f"{mean * 100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(TRAINED_MODELS, rotation=15)
    ax.set_ylabel("Invented verlan recall (%)")
    ax.set_title("Invented Verlan Recall Stability (20 seeds)")

    for idx, (label, marker, color) in enumerate(
        [
            ("Mistral-7B Zero Shot", "D", "#9467bd"),
            ("GPT-5 Codex (High) Zero Shot", "s", "#8c564b"),
        ]
    ):
        xpos = len(TRAINED_MODELS) + idx + 0.4
        value = metrics["Invented recall"][label]
        ax.scatter(xpos, value * 100, marker=marker, color=color, s=80, edgecolors="black", label=label)
        ax.text(xpos + 0.05, value * 100, f"{value * 100:.1f}%", va="center", fontsize=9)

    ax.set_xlim(-0.5, len(TRAINED_MODELS) + 1.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "invented_recall_variance.png", dpi=300)
    plt.close(fig)


def _plot_relative_improvement(metrics: Dict[str, Dict[str, float]]):
    """Plot zero-shot recall lift against best trained detector."""
    best_trained_label = "Frozen+BERT"
    baseline = metrics["Invented recall"][best_trained_label]
    values = {
        "Frozen+BERT (trained)": baseline,
        "Mistral-7B Zero Shot": metrics["Invented recall"]["Mistral-7B Zero Shot"],
        "GPT-5 Codex (High) Zero Shot": metrics["Invented recall"][
            "GPT-5 Codex (High) Zero Shot"
        ],
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = ["#1f77b4", "#9467bd", "#8c564b"]
    bar_pos = np.arange(len(values))
    heights = [v * 100 for v in values.values()]
    ax.bar(bar_pos, heights, color=colors)
    for idx, (label, val) in enumerate(zip(values.keys(), heights)):
        if label == "Frozen+BERT (trained)":
            ax.text(idx, val + 2, f"{val:.1f}%", ha="center", fontsize=9)
        else:
            improvement = (val - baseline * 100) / (baseline * 100) * 100
            ax.text(
                idx,
                val + 2,
                f"{val:.1f}%\n(+{improvement:.0f}% vs best trained)",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(values.keys(), rotation=20, ha="right")
    ax.set_ylabel("Invented verlan recall (%)")
    ax.set_title("Zero-Shot Lift on Invented Verlan")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "invented_relative_improvement.png", dpi=300)
    plt.close(fig)


def _plot_slang_fp_correlation(metrics: Dict[str, Dict[str, float]]):
    """Plot slang FP rate against main test false positives."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    main_fp = {}
    for model in TRAINED_MODELS:
        trials = _load_trials(model)
        main_fp[model] = trials["test_fp@0.5"].mean()
    mistral_summary = pd.read_csv(RESULTS_DIR / "mistral zeroshot" / "summary.csv").iloc[
        0
    ]
    main_fp["Mistral-7B Zero Shot"] = float(mistral_summary["test_fp@0.5"])
    # GPT-5 Codex has no main test run, skip its scatter.

    slang_fp_rate = {
        label: 1 - metrics["Slang specificity"].get(label, np.nan) for label in DISPLAY_LABELS
    }

    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for label, color in zip(TRAINED_MODELS + ["Mistral-7B Zero Shot"], colors):
        ax.scatter(
            main_fp[label],
            slang_fp_rate[label] * 100,
            s=80,
            color=color,
            edgecolors="black",
        )
        ax.text(
            main_fp[label] + 2,
            slang_fp_rate[label] * 100 + 0.5,
            label.split()[0],
            fontsize=8,
        )
    ax.set_xlabel("Mean FP on main test split")
    ax.set_ylabel("Slang FP rate (% of slang sentences)")
    ax.set_title("Slang False Alarms vs Main-Split FP Counts")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "slang_fp_vs_main_fp.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate every figure (default if no specific flag is supplied).",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Regenerate accuracy / F1 distribution box plots.",
    )
    parser.add_argument(
        "--slices",
        action="store_true",
        help="Regenerate the three slice bar charts.",
    )
    parser.add_argument(
        "--tradeoff",
        action="store_true",
        help="Regenerate trade-off scatter, invented variance, relative improvement, and FP correlation plots.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    do_all = args.all or not any([args.accuracy, args.slices, args.tradeoff])

    metrics, stds = _collect_targeted_metrics()

    if args.accuracy or do_all:
        _plot_accuracy_distribution()
    if args.slices or do_all:
        _plot_slice_bars(metrics)
    if args.tradeoff or do_all:
        _plot_tradeoff(metrics)
        _plot_invented_variance(metrics, stds)
        _plot_relative_improvement(metrics)
        _plot_slang_fp_correlation(metrics)


if __name__ == "__main__":
    main()
