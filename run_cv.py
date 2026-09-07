# -*- coding: utf-8 -*-
"""
Five-fold stratified cross-validation for MSDR-Net (stability assessment).

Following the manuscript protocol: five-fold stratified cross-validation is
performed on the combined training+validation cohort; each fold is a single
prespecified run (seed 2024); patient-level out-of-fold predictions are
pooled, and variability is reported as the standard deviation across folds.
Per-fold metrics and pooled summaries are saved as CSV files.

Example:
    python run_cv.py --data-root ./data --epochs 200 --batch-size 16
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data import (SpinalTumorDataset, TrainAugment, compute_class_weights,
                  load_manifest, scan_split_dir)
from models import msdr_net
from private_params import resolve as _resolve_private
from train import run_epoch
from utils import classification_metrics, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="5-fold CV for MSDR-Net")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2024,
                   help="single prespecified run per fold (seed 2024)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out-dir", type=str, default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.manifest:
        samples = load_manifest(args.manifest, "train") + load_manifest(args.manifest, "val")
    else:
        samples = scan_split_dir(args.data_root, "train") + scan_split_dir(args.data_root, "val")
    if not samples:
        raise RuntimeError("No training+validation samples found for cross-validation.")

    paths = np.array([p for p, _ in samples])
    labels = np.array([y for _, y in samples])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows, oof_probs, oof_labels = [], np.full(len(samples), np.nan), labels

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        set_seed(args.seed)

        tr = [samples[i] for i in tr_idx]
        va = [samples[i] for i in va_idx]
        tr_loader = DataLoader(SpinalTumorDataset(tr, args.image_size, TrainAugment()),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
        va_loader = DataLoader(SpinalTumorDataset(va, args.image_size),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

        model = msdr_net(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=compute_class_weights([y for _, y in tr]).to(device))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(_resolve_private("INIT_LR")),
            betas=(0.9, 0.999), weight_decay=float(_resolve_private("WEIGHT_DECAY")))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(_resolve_private("LR_MILESTONES")),
            gamma=float(_resolve_private("LR_GAMMA")))
        scaler = None if args.no_amp else torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        best_auc, best_probs = -1.0, None
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, _, _ = run_epoch(model, tr_loader, criterion,
                                              device, optimizer, scaler)
            va_loss, va_acc, va_probs, va_labels = run_epoch(model, va_loader,
                                                             criterion, device)
            scheduler.step()
            va_auc = roc_auc_score(va_labels, va_probs)
            if va_auc > best_auc:
                best_auc, best_probs = va_auc, va_probs.copy()
            if epoch % 20 == 0 or epoch == 1:
                print(f"fold {fold} epoch {epoch:03d} | train loss {tr_loss:.4f} | "
                      f"val acc {va_acc:.4f} auc {va_auc:.4f}")

        oof_probs[va_idx] = best_probs
        m = classification_metrics(va_labels, best_probs, threshold=0.5)
        fold_rows.append({
            "fold": fold,
            "accuracy_%": 100 * m["accuracy"],
            "sensitivity_%": 100 * m["sensitivity"],
            "specificity_%": 100 * m["specificity"],
            "precision_%": 100 * m["precision"],
            "f1": m["f1"],
            "auc": m["auc"],
            "tp": m["tp"], "fn": m["fn"], "tn": m["tn"], "fp": m["fp"],
        })

    df = pd.DataFrame(fold_rows)
    summary = df[["accuracy_%", "sensitivity_%", "specificity_%",
                  "precision_%", "f1", "auc"]].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    fold_path = os.path.join(args.out_dir, "cv_per_fold_metrics.csv")
    summ_path = os.path.join(args.out_dir, "cv_summary.csv")
    oof_path = os.path.join(args.out_dir, "cv_out_of_fold_predictions.csv")
    df.to_csv(fold_path, index=False)
    summary.to_csv(summ_path)
    pd.DataFrame({"path": paths, "label": oof_labels,
                  "oof_prob_malignant": oof_probs}).to_csv(oof_path, index=False)

    print("\n===== Cross-validation summary (mean +/- std) =====")
    print(summary.to_string())
    print(f"\nSaved: {fold_path}\nSaved: {summ_path}\nSaved: {oof_path}")


if __name__ == "__main__":
    main()
