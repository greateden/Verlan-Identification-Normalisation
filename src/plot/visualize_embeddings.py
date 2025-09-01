# -*- coding: utf-8 -*-
"""Plot sentence embeddings and save the Logistic Regression decision boundary.

Adds a --head argument to select which LR head to use. The visualisation
aligns the dimensionality reduction with the head's expected features:
 - If the LR head expects D features, reduce the D-dim embeddings.
 - If it expects D+1 features (embedding + heuristic), compute the
   heuristic feature for the sampled texts and reduce the concatenated
   [embeddings, heuristic] matrix so the PCA projection matches the head.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from transformers import CamembertTokenizer

try:
    from umap import UMAP
except Exception:  # pragma: no cover - optional dependency
    UMAP = None

from ..detect.detect_train_lr_bert import (
    load_data,
    load_encoder,
    embed_texts,
    sentence_has_verlan,
)

# Repository root and model path
ROOT = Path(__file__).resolve().parents[2]
HEAD_DEFAULT = ROOT / "models" / "detect" / "latest" / "lr_head.joblib"
DEFAULT_OUTFILE = ROOT / "docs" / "results" / "embedding_space.png"
REDUCERS = {"pca", "tsne", "umap"}


def _resolve_head_path(head: Path) -> Path:
    """Resolve --head to an actual joblib file.

    Accepts a direct path to a .joblib file or a directory that contains
    lr_head.joblib. Raises FileNotFoundError with a helpful message otherwise.
    """
    head = Path(head)
    if head.is_dir():
        candidate = head / "lr_head.joblib"
    else:
        candidate = head
    if not candidate.exists():
        # Provide a hint with available versions
        detect_dir = ROOT / "models" / "detect"
        available = []
        if detect_dir.exists():
            for p in sorted(detect_dir.iterdir()):
                if p.is_dir() and (p / "lr_head.joblib").exists():
                    available.append(str(p.relative_to(ROOT)))
        hint = f"Available: {', '.join(available)}" if available else "No heads found under models/detect"
        raise FileNotFoundError(
            f"Could not find LR head at '{head}'. If you passed a directory, it must contain 'lr_head.joblib'. {hint}"
        )
    return candidate


def main(
    num_points: int = 200,
    outfile: Path = DEFAULT_OUTFILE,
    reducer: str = "pca",
    head: Path = HEAD_DEFAULT,
) -> None:
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
    head_path = _resolve_head_path(head)
    print(f"Loading LR head from: {head_path}")
    lr = joblib.load(head_path)

    # Align features: some heads are trained with an extra heuristic feature
    # (CamemBERT-based binary). If so, compute it for the sampled texts and
    # concatenate before reduction so that PCA projection of weights is valid.
    try:
        expected = int(getattr(lr, "n_features_in_", lr.coef_.shape[1]))
    except Exception:
        expected = embeds.shape[1]

    if expected == embeds.shape[1] + 1:
        cam_tok = CamembertTokenizer.from_pretrained("camembert-base")
        v = np.array([sentence_has_verlan(t, cam_tok) for t in df["text"]], dtype=np.float32).reshape(-1, 1)
        feats = np.hstack([embeds, v])
    elif expected == embeds.shape[1]:
        feats = embeds
    else:
        raise ValueError(
            f"Feature mismatch: embeddings have {embeds.shape[1]} dims but LR expects {expected}. "
            "Ensure you're using a compatible encoder/head pair."
        )

    # Reduce dimensionality to 2D
    if reducer == "pca":
        reducer_model = PCA(n_components=2, random_state=42)
    elif reducer == "tsne":
        reducer_model = TSNE(n_components=2, init="pca", random_state=42)
    else:  # reducer == "umap"
        if UMAP is None:
            raise ImportError("umap-learn is required for UMAP; install with 'pip install umap-learn'")
        reducer_model = UMAP(n_components=2, random_state=42)
    reduced = reducer_model.fit_transform(feats)

    # Compute decision boundary in the reduced space
    if reducer == "pca":
        # Project original LR weights into PCA space (on aligned features)
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
    # Show which head was used (dirname if standard layout, else filename)
    try:
        head_tag = head_path.parent.name if head_path.name == "lr_head.joblib" else head_path.name
    except Exception:
        head_tag = "?"
    plt.title(f"Embeddings reduced with {reducer.upper()} (Head: {head_tag})")
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
    parser.add_argument(
        "--head",
        type=Path,
        default=HEAD_DEFAULT,
        help=(
            "Path to LR head .joblib or a model directory containing it. "
            "Default: models/detect/latest/lr_head.joblib"
        ),
    )
    args = parser.parse_args()
    main(args.num_points, args.outfile, args.reducer, args.head)
