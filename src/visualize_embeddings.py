# -*- coding: utf-8 -*-
"""Plot sentence embeddings and save the Logistic Regression decision boundary."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .detect_train import load_data, load_encoder, embed_texts

# Repository root and model path
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "detect" / "latest" / "lr_head.joblib"
DEFAULT_OUTFILE = ROOT / "docs" / "results" / "embedding_space.png"


def main(num_points: int = 200, outfile: Path = DEFAULT_OUTFILE) -> None:
    """Visualise embedding space and LR boundary, saving the plot to ``outfile``."""
    # Load a small subset of the training data
    train_df, _, _ = load_data()
    df = train_df.sample(n=min(num_points, len(train_df)), random_state=42)

    # Encode sentences into dense vectors
    tok, model = load_encoder()
    embeds = embed_texts(df["text"], tok, model)

    # Load trained Logistic Regression head
    lr = joblib.load(MODEL_PATH)

    # Reduce dimensionality to 2D for visualisation
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeds)

    # Project LR weights into the PCA space
    w = lr.coef_.ravel()
    b = lr.intercept_[0]
    w2 = pca.components_ @ w
    b2 = b + w @ pca.mean_

    # Compute decision boundary line: w2[0]*x + w2[1]*y + b2 = 0
    xs = np.linspace(reduced[:, 0].min() - 1, reduced[:, 0].max() + 1, 100)
    ys = -(w2[0] * xs + b2) / w2[1]

    # Plot scatter of points and the boundary line
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=df["label"], cmap="coolwarm", edgecolors="k", alpha=0.7)
    plt.plot(xs, ys, color="green", linewidth=2, label="LR boundary")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Embeddings with Logistic Regression decision boundary")
    plt.legend()
    plt.tight_layout()
    # Save figure instead of displaying it
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()
    print(f"Plot saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise embedding space and LR boundary")
    parser.add_argument("--num-points", type=int, default=200, help="Number of points to sample")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_OUTFILE,
        help="Path to save the generated plot",
    )
    args = parser.parse_args()
    main(args.num_points, args.outfile)
