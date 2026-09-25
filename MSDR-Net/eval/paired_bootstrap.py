

import argparse
import os

import numpy as np
import pandas as pd

RESAMPLES = 50_000
SEED = 2025
LEVEL = 0.95


def _align(reference: pd.DataFrame, comparator: pd.DataFrame):
    merged = reference.merge(comparator, on=["patient_id", "y_true"],
                             suffixes=("_ref", "_cmp"))
    if len(merged) != len(reference):
        raise ValueError("Prediction files could not be aligned by patient_id/y_true.")
    return (merged["y_true"].to_numpy().astype(int),
            merged["y_score_ref"].to_numpy(),
            merged["y_score_cmp"].to_numpy())


def discordance_counts(y_true, y_score_ref, y_score_cmp, threshold: float = 0.5) -> dict:
    """Paired discordance table (correctness of each model at the threshold)."""
    correct_ref = ((y_score_ref >= threshold).astype(int) == y_true)
    correct_cmp = ((y_score_cmp >= threshold).astype(int) == y_true)
    return {
        "a_both_correct": int((correct_ref & correct_cmp).sum()),
        "b_ref_only": int((correct_ref & ~correct_cmp).sum()),
        "c_cmp_only": int((~correct_ref & correct_cmp).sum()),
        "d_both_wrong": int((~correct_ref & ~correct_cmp).sum()),
    }


def _metrics_from_counts(tp, fn, tn, fp):
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * sensitivity * precision / (sensitivity + precision)
    return accuracy, sensitivity, specificity, f1


def paired_bootstrap(y_true, y_score_ref, y_score_cmp, threshold: float = 0.5,
                     resamples: int = RESAMPLES, seed: int = SEED,
                     level: float = LEVEL) -> dict:
    """Joint patient-level bootstrap of paired metric differences (ref - cmp)."""
    y_true = np.asarray(y_true).astype(int)
    pred_ref = (np.asarray(y_score_ref) >= threshold).astype(int)
    pred_cmp = (np.asarray(y_score_cmp) >= threshold).astype(int)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))  # joint resampling of patients

    yt = y_true[idx]
    pr = pred_ref[idx]
    pc = pred_cmp[idx]

    tp_r = ((pr == 1) & (yt == 1)).sum(axis=1)
    fn_r = ((pr == 0) & (yt == 1)).sum(axis=1)
    tn_r = ((pr == 0) & (yt == 0)).sum(axis=1)
    fp_r = ((pr == 1) & (yt == 0)).sum(axis=1)
    tp_c = ((pc == 1) & (yt == 1)).sum(axis=1)
    fn_c = ((pc == 0) & (yt == 1)).sum(axis=1)
    tn_c = ((pc == 0) & (yt == 0)).sum(axis=1)
    fp_c = ((pc == 1) & (yt == 0)).sum(axis=1)

    m_ref = _metrics_from_counts(tp_r, fn_r, tn_r, fp_r)
    m_cmp = _metrics_from_counts(tp_c, fn_c, tn_c, fp_c)
    alpha = 1.0 - level

    result = {}
    for name, d in zip(["accuracy", "sensitivity", "specificity", "f1"],
                       [r - c for r, c in zip(m_ref, m_cmp)]):
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result[name] = {"diff": float(np.mean(d)), "ci_lower": float(lo),
                        "ci_upper": float(hi)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="CSV of the reference (MSDR-Net) predictions")
    parser.add_argument("--comparators", required=True, nargs="+",
                        help="CSV(s) of comparator predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    rows = []
    for cmp_path in args.comparators:
        comparator = pd.read_csv(cmp_path)
        y_true, s_ref, s_cmp = _align(reference, comparator)
        disc = discordance_counts(y_true, s_ref, s_cmp, args.threshold)
        boot = paired_bootstrap(y_true, s_ref, s_cmp, args.threshold,
                                args.resamples, args.seed)
        cmp_name = os.path.splitext(os.path.basename(cmp_path))[0]
        for metric, stats_ in boot.items():
            rows.append({"comparator": cmp_name, "metric": metric,
                         "diff": stats_["diff"], "ci_lower": stats_["ci_lower"],
                         "ci_upper": stats_["ci_upper"],
                         "b_ref_only": disc["b_ref_only"],
                         "c_cmp_only": disc["c_cmp_only"]})
        # Sanity check of the c = 0 property (see module docstring).
        if disc["c_cmp_only"] == 0:
            bad = [m for m, s in boot.items() if s["ci_lower"] < 0]
            if bad:
                raise AssertionError(
                    f"Impossible negative lower bound(s) for {bad} despite c = 0; "
                    "the pairing is broken - check the implementation.")
        print(f"[{cmp_name}] discordance: {disc}")

    out = pd.DataFrame(rows)
    output = args.output or "paired_bootstrap_results.csv"
    out.to_csv(output, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
