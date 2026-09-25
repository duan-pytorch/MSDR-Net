

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data.dataset import (SpinalTumorDataset, build_eval_transform,
                          build_train_transform, compute_class_weights)
from models.msdr_net import build_model
from models.baselines import build_baseline, BASELINE_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(name: str):
    if name in BASELINE_NAMES:
        return build_baseline(name)
    return build_model(name)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for imgs, labels, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = torch.softmax(model(imgs), dim=1)[:, 1]  # malignant-class probability
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_ids.extend(ids)
    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float((y_pred == y_true).mean()),
    }
    predictions = pd.DataFrame({"patient_id": all_ids, "y_true": y_true, "y_score": y_score})
    return metrics, predictions


def train_one_run(cfg: dict, model_name: str, seed: int, output_dir: str) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["training"]

    train_set = SpinalTumorDataset(cfg["data"]["index_csv"], cfg["data"]["image_root"],
                                   "train", transform=build_train_transform())
    val_set = SpinalTumorDataset(cfg["data"]["index_csv"], cfg["data"]["image_root"],
                                 "val", transform=build_eval_transform())
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"], pin_memory=True)

    model = get_model(model_name).to(device)
    class_weights = compute_class_weights(train_set.df["label"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"],
                                  betas=tuple(tcfg["betas"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=tcfg["lr_milestones"], gamma=tcfg["lr_gamma"])
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and device.type == "cuda")

    best_auc, best_epoch, epochs_without_improvement = -1.0, -1, 0
    patience = tcfg["early_stopping"]["patience"]
    ckpt_path = os.path.join(output_dir, f"{model_name}_seed{seed}_best.pth")
    history = []

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tcfg["amp"] and device.type == "cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(labels)
            n_seen += len(labels)
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_seen, 1),
                        "val_auc": val_metrics["auc"],
                        "val_accuracy": val_metrics["accuracy"]})
        print(f"[{model_name} seed={seed}] epoch {epoch:3d} "
              f"train_loss={history[-1]['train_loss']:.4f} "
              f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch = val_metrics["auc"], epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_auc": best_auc, "seed": seed, "model": model_name}, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best val AUC {best_auc:.4f} "
                      f"at epoch {best_epoch})")
                break

    pd.DataFrame(history).to_csv(os.path.join(output_dir, f"{model_name}_seed{seed}_log.csv"),
                                 index=False)

    # Record the run summary (Supplementary Table S8 fields).
    _, val_preds = evaluate(model, val_loader, device)
    summary = {"model": model_name, "seed": seed, "checkpoint_epoch": best_epoch,
               "val_auc": best_auc,
               "val_accuracy": float((val_preds["y_score"] >= 0.5).astype(int)
                                     .eq(val_preds["y_true"]).mean())}
    with open(os.path.join(output_dir, f"{model_name}_seed{seed}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", required=True,
                        help="msdr_net[_full|_no_eca|_single|_single_eca] or a baseline name")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = os.path.join(cfg["output"]["dir"], "runs")
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    summary = train_one_run(cfg, args.model, args.seed, output_dir)
    summary["wall_time_min"] = (time.time() - t0) / 60.0
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
