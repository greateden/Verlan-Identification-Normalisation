import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Tuple


def temperature_scale(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply temperature scaling to probability scores.

    Parameters
    ----------
    probs : np.ndarray
        Model probabilities for the positive class.
    labels : np.ndarray
        Ground truth binary labels.

    Returns
    -------
    Tuple[np.ndarray, float]
        Calibrated probabilities and the learned temperature value.
    """
    logits = torch.logit(torch.clamp(torch.tensor(probs), 1e-6, 1 - 1e-6))
    y = torch.tensor(labels, dtype=torch.float32)
    T = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scaled = torch.sigmoid(logits / T)
    return scaled.numpy(), float(T.item())


def platt_scale(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calibrate scores using Platt scaling (logistic regression)."""
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(scores.reshape(-1, 1), labels)
    return lr.predict_proba(scores.reshape(-1, 1))[:, 1]


def isotonic_calibration(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calibrate scores using isotonic regression."""
    iso = IsotonicRegression(out_of_bounds="clip")
    return iso.fit_transform(scores, labels)


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """Find the threshold that maximises F1 score or Youden's J.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    probs : np.ndarray
        Probability scores for the positive class.
    metric : str, optional
        Metric to maximise: "f1" or "youden".

    Returns
    -------
    Tuple[float, float]
        Best threshold and its score.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        if metric == "youden":
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            score = tpr - fpr
        else:
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            score = 2 * precision * recall / (precision + recall + 1e-12)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score
