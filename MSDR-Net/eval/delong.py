

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    midranks_sorted = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    midranks = np.empty(n)
    midranks[order] = midranks_sorted
    return midranks


def _fast_delong(predictions_sorted: np.ndarray, n_pos: int):
    """predictions_sorted: (k, n) score matrix, positives in the first columns."""
    k, n_total = predictions_sorted.shape
    n_neg = n_total - n_pos
    positives = predictions_sorted[:, :n_pos]
    negatives = predictions_sorted[:, n_pos:]
    tx = np.empty((k, n_pos))
    ty = np.empty((k, n_neg))
    tz = np.empty((k, n_total))
    for r in range(k):
        tx[r, :] = _compute_midrank(positives[r, :])
        ty[r, :] = _compute_midrank(negatives[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / n_pos + sy / n_neg
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true, scores_ref, scores_cmp):
    """Two-sided DeLong test. Returns (auc_ref, auc_cmp, z, p, ci_lo, ci_hi)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    preds = np.vstack([np.asarray(scores_ref), np.asarray(scores_cmp)])[:, order]
    n_pos = int((y_true == 1).sum())
    aucs, cov = _fast_delong(preds, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), 1.0, float("nan"), float("nan")
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(var))
    z = float(diff / se)
    p = float(2.0 * stats.norm.sf(abs(z)))
    z975 = stats.norm.ppf(0.975)
    return (float(aucs[0]), float(aucs[1]), z, p,
            float(diff - z975 * se), float(diff + z975 * se))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparators", required=True, nargs="+")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    rows = []
    for cmp_path in args.comparators:
        comparator = pd.read_csv(cmp_path)
        merged = reference.merge(comparator, on=["patient_id", "y_true"],
                                 suffixes=("_ref", "_cmp"))
        auc_ref, auc_cmp, z, p, lo, hi = delong_roc_test(
            merged["y_true"].to_numpy(),
            merged["y_score_ref"].to_numpy(), merged["y_score_cmp"].to_numpy())
        rows.append({"comparator": os.path.splitext(os.path.basename(cmp_path))[0],
                     "auc_reference": auc_ref, "auc_comparator": auc_cmp,
                     "auc_diff": auc_ref - auc_cmp, "ci_lower": lo, "ci_upper": hi,
                     "delong_z": z, "delong_p": p})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
