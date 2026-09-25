
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from cross_validation import _train_fold
from eval.metrics import auc_hanley_mcneil, threshold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default="msdr_net")
    parser.add_argument("--center-a", default="A", help="development center")
    parser.add_argument("--center-b", default="B", help="held-out evaluation center")
    parser.add_argument("--seed", type=int, default=2022,
                        help="single run seed for the final locked model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(cfg["output"]["dir"], "external_validation")
    os.makedirs(output_dir, exist_ok=True)

    index = pd.read_csv(cfg["data"]["index_csv"])
    dev = index[index["center"] == args.center_a].reset_index(drop=True)
    heldout = index[index["center"] == args.center_b].reset_index(drop=True)

    # Patient-level train/validation partition within Institution A,
    # stratified by label (7:2 ratio of the overall 7:2:1 scheme).
    rng = np.random.RandomState(args.seed)
    train_parts, val_parts = [], []
    for _, group in dev.groupby("label"):
        perm = rng.permutation(len(group))
        n_val = max(1, int(round(len(group) * 2 / 9)))
        val_parts.append(group.iloc[perm[:n_val]])
        train_parts.append(group.iloc[perm[n_val:]])
    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)

    # Train the locked model entirely within Institution A, then evaluate
    # exactly once on the held-out Institution B cohort.
    preds_b = _train_fold(cfg, args.model, args.seed, train_df, heldout, device)

    threshold = cfg["evaluation"]["decision_threshold"]
    metrics = threshold_metrics(preds_b["y_true"].to_numpy(),
                                preds_b["y_score"].to_numpy(), threshold=threshold)
    auc, auc_lo, auc_hi = auc_hanley_mcneil(preds_b["y_true"].to_numpy(),
                                            preds_b["y_score"].to_numpy())
    metrics["auc"] = auc
    metrics["auc_ci_lower"] = auc_lo
    metrics["auc_ci_upper"] = auc_hi
    preds_b.to_csv(os.path.join(output_dir, "external_predictions.csv"), index=False)
    pd.Series(metrics).to_csv(os.path.join(output_dir, "external_metrics.csv"))
    print(pd.Series(metrics))


if __name__ == "__main__":
    main()
