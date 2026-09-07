# -*- coding: utf-8 -*-
"""
Evaluation metrics for binary benign/malignant classification.

Malignant tumors are the positive class, benign tumors the negative class.
Metrics follow Section 2.7 of the manuscript: accuracy, sensitivity
(recall), specificity, precision, F1 score, AUC-ROC, PR-AUC, Brier score,
expected calibration error (ECE), and decision-curve net benefit.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) with malignant as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(tn), int(fp), int(fn)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, float]:
    """Compute all headline metrics at a fixed decision threshold (0.5)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / max(tp + fn, 1),          # recall (malignant)
        "specificity": tn / max(tn + fp, 1),          # benign recognition
        "precision": tp / max(tp + fp, 1),            # PPV
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE with `n_bins` equal-width bins over [0, 1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray,
                                eps: float = 1e-6) -> Tuple[float, float]:
    """Logistic calibration slope and intercept (fitted on the logit)."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(y_prob).astype(float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit, np.asarray(y_true).astype(int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def decision_curve_net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                               thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    """Net benefit across threshold probabilities.

    NB(p_t) = TP/n - FP/n * (p_t / (1 - p_t))
    Returns an array of shape (len(thresholds), 3) with columns
    [model, treat_all, treat_none].
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append((nb_model, nb_all, 0.0))
    return np.asarray(rows), thresholds


class TemperatureScaler:
    """Temperature scaling for probability calibration (fit on the
    validation set only, per the manuscript protocol)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray,
            lr: float = 0.01, max_iter: int = 500):
        import torch

        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.tensor(np.asarray(y_true), dtype=torch.long)
        # Optimize log-temperature so that T = exp(log_T) stays positive.
        log_temperature = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits_t / log_temperature.exp(), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.exp().item())
        return self

    def transform_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return temperature-scaled malignant-class probabilities."""
        import torch

        logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
        prob = torch.softmax(logits_t / self.temperature, dim=1)[:, 1]
        return prob.detach().cpu().numpy()
