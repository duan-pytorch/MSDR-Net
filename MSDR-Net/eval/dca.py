

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Arial"

FIG_DPI = 600


def net_benefit(y_true, y_score, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n = len(y_true)
    prevalence = y_true.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        pred = y_score >= pt
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        weight = pt / (1 - pt) if pt < 1 else np.inf
        nb_model.append(tp / n - fp / n * weight)
        nb_all.append(prevalence - (1 - prevalence) * weight)
    return np.asarray(nb_model), np.asarray(nb_all)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)
    thresholds = np.arange(args.min_threshold, args.max_threshold + 1e-9, args.step)
    nb_model, nb_all = net_benefit(preds["y_true"].to_numpy(),
                                   preds["y_score"].to_numpy(), thresholds)
    nb_none = np.zeros_like(thresholds)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(thresholds, nb_model, color="tab:red", lw=2, label="MSDR-Net")
    ax.plot(thresholds, nb_all, color="gray", ls="--", label="Treat all as malignant")
    ax.plot(thresholds, nb_none, color="black", lw=1.5, label="Treat all as benign")
    ax.set_xlabel("Threshold Probability of Malignancy")
    ax.set_ylabel("Net Benefit")
    ax.set_ylim(min(-0.1, nb_model.min() - 0.02), max(0.3, nb_model.max() + 0.02))
    ax.legend(loc="upper right")
    fig.tight_layout()

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        fig.savefig(os.path.join(args.output, "decision_curve.png"), dpi=FIG_DPI)
        pd.DataFrame({"threshold": thresholds, "net_benefit_model": nb_model,
                      "net_benefit_treat_all": nb_all,
                      "net_benefit_treat_none": nb_none}).to_csv(
            os.path.join(args.output, "dca_values.csv"), index=False)
        print(f"Saved decision curve to {args.output}")
    else:
        fig.savefig("decision_curve.png", dpi=FIG_DPI)
        print("Saved decision curve to decision_curve.png")


if __name__ == "__main__":
    main()
