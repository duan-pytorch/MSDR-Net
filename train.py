# -*- coding: utf-8 -*-
"""
Training script for MSDR-Net.

Reproduces the training protocol of the manuscript (Section 2.5 / 2.6):

    * weighted cross-entropy loss (class imbalance handling)
    * AdamW optimizer; MultiStepLR schedule
    * automatic mixed precision (AMP)
    * early stopping on validation AUC (patience 20); checkpoint at the
      best validation AUC
    * three independent runs (seeds 2021/2022/2023); the run with the
      highest validation AUC is carried forward

NOTE: several exact hyperparameters (initial LR, weight decay, LR-decay
milestones, class weights, MLP head configuration, ECA kernel mapping
constants) are redacted in this public release and replaced by reasonable
placeholder values. For the exact key parameters, please contact:
    a13995115025@163.com

Example:
    python train.py --data-root ./data --epochs 200 --batch-size 16 --seed 2022
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (SpinalTumorDataset, TrainAugment, compute_class_weights,
                  load_manifest, scan_split_dir)
from models import count_parameters, msdr_net
from private_params import resolve as _resolve_private
from utils import AverageMeter, get_device, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train MSDR-Net")
    p.add_argument("--data-root", type=str, default="./data",
                   help="root with train/val/[test] subfolders")
    p.add_argument("--manifest", type=str, default=None,
                   help="optional CSV manifest with [path, label, split]")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stopping patience (epochs, monitor: val AUC)")
    p.add_argument("--ablation", type=str, default=None,
                   choices=[None, "no_eca", "single_branch"],
                   help="ablation variant (None = full MSDR-Net)")
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--no-amp", action="store_true",
                   help="disable automatic mixed precision")
    return p.parse_args()


def build_samples(args, split):
    if args.manifest:
        return load_manifest(args.manifest, split)
    return scan_split_dir(args.data_root, split)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One training or evaluation epoch. Returns (loss, acc, probs, labels)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    meter = AverageMeter()
    correct, total = 0, 0
    all_probs, all_labels = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        meter.update(loss.item(), x.size(0))
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        probs = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return meter.avg, correct / max(total, 1), probs, labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_samples = build_samples(args, "train")
    val_samples = build_samples(args, "val")
    if not train_samples or not val_samples:
        raise RuntimeError(
            "No training/validation samples found. Expected "
            "<data-root>/train/{benign,malignant}/*.png and "
            "<data-root>/val/{benign,malignant}/*.png, or a CSV manifest "
            "via --manifest with columns [path, label, split].")

    train_set = SpinalTumorDataset(train_samples, image_size=args.image_size,
                                   transform=TrainAugment())
    val_set = SpinalTumorDataset(val_samples, image_size=args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = msdr_net(num_classes=2, ablation=args.ablation).to(device)
    print(f"MSDR-Net (ablation={args.ablation}) - "
          f"{count_parameters(model) / 1e6:.1f}M trainable parameters")

    # ------------------------------------------------------------------
    # Loss / optimizer / schedule
    # NOTE: exact hyperparameters are redacted; reasonable placeholder
    # values are resolved automatically (see private_params.py). Contact
    # a13995115025@163.com for the exact key parameters.
    # ------------------------------------------------------------------
    class_weights = compute_class_weights(train_set.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = float(_resolve_private("INIT_LR"))
    wd = float(_resolve_private("WEIGHT_DECAY"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=wd)
    milestones = list(_resolve_private("LR_MILESTONES"))
    lr_gamma = float(_resolve_private("LR_GAMMA"))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_gamma)
    scaler = None if args.no_amp else torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ------------------------------------------------------------------
    # Training loop with early stopping (monitor: validation AUC)
    # ------------------------------------------------------------------
    tag = "full" if args.ablation is None else args.ablation
    best_auc, best_epoch, bad_epochs = -1.0, -1, 0
    ckpt_path = os.path.join(args.out_dir, f"msdr_net_{tag}_seed{args.seed}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion,
                                          device, optimizer, scaler)
        va_loss, va_acc, va_probs, va_labels = run_epoch(model, val_loader,
                                                         criterion, device)
        scheduler.step()
        va_auc = roc_auc_score(va_labels, va_probs) if len(np.unique(va_labels)) > 1 else float("nan")

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{time.time() - t0:.1f}s")

        if va_auc > best_auc:
            best_auc, best_epoch, bad_epochs = va_auc, epoch, 0
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": va_auc,
                "val_acc": va_acc,
                "args": vars(args),
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val AUC {best_auc:.4f} @ epoch {best_epoch}).")
                break

    print(f"Training finished. Best val AUC {best_auc:.4f} "
          f"(epoch {best_epoch}); checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
