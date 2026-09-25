

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, precision_recall_curve, auc

plt.rcParams["font.family"] = "Arial"

ECE_BINS = 10
FIG_DPI = 600


def expected_calibration_error(y_true, y_prob, n_bins: int = ECE_BINS) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > edges[i]) & (y_prob <= edges[i + 1])
        if mask.sum() > 0:
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y_true, y_prob):
    """Calibration intercept (slope fixed to 1) and slope (both free)."""
    y_true = np.asarray(y_true).astype(float)
    lp = _logit(np.asarray(y_prob))

    def nll_intercept_only(a):
        z = a + lp
        return np.mean(np.logaddexp(0, z) - y_true * z)

    res_a = minimize_scalar(nll_intercept_only)
    intercept = res_a.x

    def nll_free(params):
        a, b = params
        z = a + b * lp
        return np.mean(np.logaddexp(0, z) - y_true * z)

    from scipy.optimize import minimize
    res = minimize(nll_free, x0=np.array([intercept, 1.0]), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def fit_temperature(y_true_val, y_prob_val) -> float:
    """Fit the temperature T on validation data by minimizing NLL."""
    y_true = np.asarray(y_true_val).astype(float)
    lp = _logit(np.asarray(y_prob_val))

    def nll(log_t):
        z = lp / np.exp(log_t)
        return float(np.sum(log_loss(y_true, 1.0 / (1.0 + np.exp(-z)),
                                     labels=[0, 1])))

    res = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(y_prob, temperature: float) -> np.ndarray:
    z = _logit(np.asarray(y_prob)) / temperature
    return 1.0 / (1.0 + np.exp(-z))


def calibration_table(y_true, y_prob_before, y_prob_after) -> pd.DataFrame:
    rows = []
    for label, probs in [("before", y_prob_before), ("after", y_prob_after)]:
        intercept, slope = calibration_intercept_slope(y_true, probs)
        rows.append({
            "stage": label,
            "brier": brier_score_loss(y_true, probs),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "ece_10bins": expected_calibration_error(y_true, probs),
        })
    precision, recall, _ = precision_recall_curve(y_true, y_prob_before)
    rows[0]["pr_auc"] = float(auc(recall, precision))
    rows[1]["pr_auc"] = float(auc(recall, precision))
    return pd.DataFrame(rows)


def plot_calibration(y_true, y_prob_before, y_prob_after, out_path: str,
                     n_bins: int = 6) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    for probs, marker, label in [(y_prob_before, "o", "Before calibration"),
                                 (y_prob_after, "s", "After temperature scaling")]:
        quantiles = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
        xs, ys = [], []
        for i in range(n_bins):
            mask = (probs >= quantiles[i]) & (probs <= quantiles[i + 1])
            if mask.sum() > 0:
                xs.append(float(np.mean(probs[mask])))
                ys.append(float(np.mean(np.asarray(y_true)[mask])))
        ax.plot(xs, ys, marker=marker, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed malignancy fraction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    print(f"Saved calibration figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True, help="validation-set predictions CSV")
    parser.add_argument("--test", required=True, help="test-set predictions CSV")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)
    temperature = fit_temperature(val["y_true"].to_numpy(), val["y_score"].to_numpy())
    print(f"Temperature fitted on validation set: T = {temperature:.4f}")

    table = calibration_table(test["y_true"].to_numpy(),
                              test["y_score"].to_numpy(),
                              apply_temperature(test["y_score"].to_numpy(), temperature))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        table.to_csv(os.path.join(args.output, "calibration_metrics.csv"), index=False)
        plot_calibration(test["y_true"].to_numpy(), test["y_score"].to_numpy(),
                         apply_temperature(test["y_score"].to_numpy(), temperature),
                         os.path.join(args.output, "calibration_curve.png"))


if __name__ == "__main__":
    main()
