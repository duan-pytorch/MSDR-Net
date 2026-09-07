# -*- coding: utf-8 -*-
"""
Evaluation script for MSDR-Net.

Computes the full evaluation battery of the manuscript (Section 2.7) on a
locked test set, at the prespecified decision threshold of 0.5 on the
Softmax-normalized malignant-class probability:

    * accuracy / sensitivity / specificity / precision / F1 / AUC / PR-AUC
    * calibration: Brier score, ECE (10 bins), calibration slope/intercept
    * optional temperature scaling fitted on the validation split
    * decision curve analysis (net benefit vs. treat-all / treat-none)

All metrics are written to CSV files under --out-dir.

Example:
    python evaluate.py --data-root ./data --checkpoint checkpoints/msdr_net_full_seed2022_best.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import SpinalTumorDataset, load_manifest, scan_split_dir
from models import msdr_net
from utils import (TemperatureScaler, brier_score, calibration_slope_intercept,
                   classification_metrics, decision_curve_net_benefit,
                   expected_calibration_error, get_device, load_checkpoint,
                   set_seed)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MSDR-Net")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ablation", type=str, default=None,
                   choices=[None, "no_eca", "single_branch"])
    p.add_argument("--threshold", type=float, default=0.5,
                   help="prespecified decision threshold (0.5)")
    p.add_argument("--temperature-scaling", action="store_true",
                   help="fit temperature scaling on the val split")
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--out-dir", type=str, default="./results")
    return p.parse_args()


@torch.no_grad()
def infer(model, loader, device):
    """Collect logits, malignant-class probabilities and labels."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True))[:, 1]
    return logits, probs, labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = (load_manifest(args.manifest, args.split)
               if args.manifest else scan_split_dir(args.data_root, args.split))
    if not samples:
        raise RuntimeError(f"No samples found for split '{args.split}'.")
    dataset = SpinalTumorDataset(samples, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = msdr_net(num_classes=2, ablation=args.ablation).to(device)
    load_checkpoint(args.checkpoint, model, map_location="cpu")

    logits, probs, labels = infer(model, loader, device)

    # Optional: temperature scaling fitted on the validation split only.
    if args.temperature_scaling:
        val_samples = (load_manifest(args.manifest, "val")
                       if args.manifest else scan_split_dir(args.data_root, "val"))
        if val_samples:
            val_loader = DataLoader(
                SpinalTumorDataset(val_samples, image_size=args.image_size),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            val_logits, _, val_labels = infer(model, val_loader, device)
            scaler = TemperatureScaler().fit(val_logits, val_labels)
            probs = scaler.transform_proba(logits)
            print(f"Temperature scaling applied (T = {scaler.temperature:.3f})")
        else:
            print("WARNING: no validation split found; skipping temperature scaling.")

    # Headline classification metrics at the prespecified 0.5 threshold.
    m = classification_metrics(labels, probs, threshold=args.threshold)
    m["brier"] = brier_score(labels, probs)
    m["ece_10bins"] = expected_calibration_error(labels, probs, n_bins=10)
    slope, intercept = calibration_slope_intercept(labels, probs)
    m["calibration_slope"], m["calibration_intercept"] = slope, intercept

    rows = [
        ("Accuracy (%)", 100 * m["accuracy"]),
        ("Sensitivity (%)", 100 * m["sensitivity"]),
        ("Specificity (%)", 100 * m["specificity"]),
        ("Precision (%)", 100 * m["precision"]),
        ("F1 score", m["f1"]),
        ("AUC-ROC", m["auc"]),
        ("PR-AUC", m["pr_auc"]),
        ("Brier score", m["brier"]),
        ("ECE (10 bins)", m["ece_10bins"]),
        ("Calibration slope", m["calibration_slope"]),
        ("Calibration intercept", m["calibration_intercept"]),
        ("TP", m["tp"]), ("TN", m["tn"]), ("FP", m["fp"]), ("FN", m["fn"]),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value"])
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.split}.csv")
    df.to_csv(metrics_path, index=False)
    print(df.to_string(index=False))

    # Per-case predictions.
    pred_path = os.path.join(args.out_dir, f"predictions_{args.split}.csv")
    pd.DataFrame({
        "path": [p for p, _ in samples],
        "label": labels,
        "prob_malignant": probs,
        "pred": (probs >= args.threshold).astype(int),
    }).to_csv(pred_path, index=False)

    # Decision curve analysis (net benefit vs. threshold probability).
    nb, thresholds = decision_curve_net_benefit(labels, probs)
    dca_path = os.path.join(args.out_dir, f"dca_{args.split}.csv")
    pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_model": nb[:, 0],
        "net_benefit_treat_all": nb[:, 1],
        "net_benefit_treat_none": nb[:, 2],
    }).to_csv(dca_path, index=False)

    print(f"\nSaved: {metrics_path}\nSaved: {pred_path}\nSaved: {dca_path}")


if __name__ == "__main__":
    main()
