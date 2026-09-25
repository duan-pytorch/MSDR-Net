

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def confusion_at_threshold(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def threshold_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    cm = confusion_at_threshold(y_true, y_score, threshold)
    tp, fn, tn, fp = cm["tp"], cm["fn"], cm["tn"], cm["fp"]
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * sensitivity * precision, sensitivity + precision)
    return {
        "accuracy": _safe_div(tp + tn, tp + fn + tn + fp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) > 1 else float("nan"),
    }


def clopper_pearson_ci(k: int, n: int, level: float = 0.95):
    """Exact (Clopper-Pearson) two-sided confidence interval for a proportion."""
    alpha = 1.0 - level
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def auc_hanley_mcneil(y_true, y_score, level: float = 0.95):
    """AUC with Hanley-McNeil standard error; CI truncated to [0, 1]."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = float(np.sqrt(var))
    z = stats.norm.ppf(1 - (1 - level) / 2)
    return auc, float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def report_with_cis(y_true, y_score, threshold: float = 0.5,
                    level: float = 0.95) -> pd.DataFrame:
    cm = confusion_at_threshold(y_true, y_score, threshold)
    tp, fn, tn, fp = cm["tp"], cm["fn"], cm["tn"], cm["fp"]
    rows = []
    for name, k, n in [("accuracy", tp + tn, tp + fn + tn + fp),
                       ("sensitivity", tp, tp + fn),
                       ("specificity", tn, tn + fp),
                       ("precision", tp, tp + fp)]:
        lo, hi = clopper_pearson_ci(k, n, level)
        rows.append({"metric": name, "value": k / n, "ci_lower": lo, "ci_upper": hi,
                     "counts": f"{k}/{n}"})
    auc, lo, hi = auc_hanley_mcneil(y_true, y_score, level)
    rows.append({"metric": "auc", "value": auc, "ci_lower": lo, "ci_upper": hi,
                 "counts": "-"})
    m = threshold_metrics(y_true, y_score, threshold)
    rows.append({"metric": "f1", "value": m["f1"], "ci_lower": np.nan,
                 "ci_upper": np.nan, "counts": "-"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="CSV with columns: patient_id, y_true, y_score")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)
    table = report_with_cis(preds["y_true"].to_numpy(), preds["y_score"].to_numpy(),
                            threshold=args.threshold)
    print(table.to_string(index=False,
                          float_format=lambda v: f"{v:.4f}" if pd.notna(v) else ""))


if __name__ == "__main__":
    main()
