# -*- coding: utf-8 -*-
"""Plot sentence embeddings and save the Logistic Regression decision boundary."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

try:
    from umap import UMAP
except Exception:  # pragma: no cover - optional dependency
    UMAP = None

from .detect_train import load_data, load_encoder, embed_texts

# Repository root and model path
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "detect" / "latest" / "lr_head.joblib"
DEFAULT_OUTFILE = ROOT / "docs" / "results" / "embedding_space.png"
REDUCERS = {"pca", "tsne", "umap"}


def main(num_points: int = 200, outfile: Path = DEFAULT_OUTFILE, reducer: str = "pca") -> None:
    """Visualise embedding space and LR boundary, saving the plot to ``outfile``.

    Parameters
    ----------
    num_points:
        Number of sentences sampled from the training set.
    outfile:
        Path to the image written to disk.
    reducer:
        Dimensionality reduction algorithm: ``pca``, ``tsne`` or ``umap``.
    """
    reducer = reducer.lower()
    if reducer not in REDUCERS:
        raise ValueError(f"Unknown reducer '{reducer}'. Choose from {sorted(REDUCERS)}")

    # Load a subset of the training data
    train_df, _, _ = load_data()
    df = train_df.sample(n=min(num_points, len(train_df)), random_state=42)

    # Encode sentences into dense vectors
    tok, model = load_encoder()
    embeds = embed_texts(df["text"], tok, model)
    # Load trained Logistic Regression head for colouring / PCA projection
    lr = joblib.load(MODEL_PATH)

    # Reduce dimensionality to 2D
    if reducer == "pca":
        reducer_model = PCA(n_components=2, random_state=42)
    elif reducer == "tsne":
        reducer_model = TSNE(n_components=2, init="pca", random_state=42)
    else:  # reducer == "umap"
        if UMAP is None:
            raise ImportError("umap-learn is required for UMAP; install with 'pip install umap-learn'")
        reducer_model = UMAP(n_components=2, random_state=42)
    reduced = reducer_model.fit_transform(embeds)

    # Compute decision boundary in the reduced space
    if reducer == "pca":
        # Project original LR weights into PCA space
        w = lr.coef_.ravel()
        b = lr.intercept_[0]
        w2 = reducer_model.components_ @ w
        b2 = b + w @ reducer_model.mean_
    else:
        # Fit a fresh LR on the 2D embeddings for non-linear reducers
        lr2 = LogisticRegression().fit(reduced, df["label"])
        w2 = lr2.coef_.ravel()
        b2 = lr2.intercept_[0]

    xs = np.linspace(reduced[:, 0].min() - 1, reduced[:, 0].max() + 1, 200)
    ys = -(w2[0] * xs + b2) / w2[1]

    # Plot scatter and boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=df["label"],
        cmap="coolwarm",
        edgecolors="k",
        alpha=0.7,
    )
    plt.plot(xs, ys, color="green", linewidth=2, label="LR boundary")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"Embeddings reduced with {reducer.upper()}" )
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
    parser.add_argument("--reducer", type=str, default="pca", choices=sorted(REDUCERS), help="Dimensionality reduction method")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_OUTFILE,
        help="Path to save the generated plot",
    )
    args = parser.parse_args()
    main(args.num_points, args.outfile, args.reducer)
