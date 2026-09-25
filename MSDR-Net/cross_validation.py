

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from data.dataset import (build_eval_transform, build_train_transform,
                          compute_class_weights)
from eval.metrics import confusion_at_threshold, threshold_metrics
from train import get_model, set_seed


class FoldDataset(Dataset):
    """In-memory dataset view over a dataframe slice (used for CV folds)."""

    def __init__(self, df: pd.DataFrame, image_root: str, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = np.load(os.path.join(self.image_root,
                                   os.path.splitext(str(row["image_path"]))[0] + ".npy"))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, int(row["label"]), str(row["patient_id"])


def _train_fold(cfg: dict, model_name: str, seed: int,
                train_df: pd.DataFrame, val_df: pd.DataFrame, device) -> pd.DataFrame:
    """Train one fold and return out-of-fold predictions for its held-out fold."""
    tcfg = cfg["training"]
    train_loader = DataLoader(FoldDataset(train_df, cfg["data"]["image_root"],
                                          build_train_transform()),
                              batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=tcfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(FoldDataset(val_df, cfg["data"]["image_root"],
                                        build_eval_transform()),
                            batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"], pin_memory=True)

    set_seed(seed)
    model = get_model(model_name).to(device)
    # Class weights are recomputed within the training portion of each fold.
    class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"],
                                  betas=tuple(tcfg["betas"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=tcfg["lr_milestones"], gamma=tcfg["lr_gamma"])
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and device.type == "cuda")

    def _predict() -> pd.DataFrame:
        model.eval()
        probs, trues, ids = [], [], []
        with torch.no_grad():
            for imgs, labels, pids in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                probs.append(torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy())
                trues.append(labels.numpy())
                ids.extend(pids)
        return pd.DataFrame({"patient_id": ids, "y_true": np.concatenate(trues),
                             "y_score": np.concatenate(probs)})

    best_auc, epochs_without_improvement, best_state = -1.0, 0, None
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tcfg["amp"] and device.type == "cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        preds = _predict()
        auc = roc_auc_score(preds["y_true"], preds["y_score"])
        if auc > best_auc:
            best_auc = auc
            epochs_without_improvement = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= tcfg["early_stopping"]["patience"]:
                break

    model.load_state_dict(best_state)
    return _predict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default="msdr_net")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(cfg["output"]["dir"], "cross_validation")
    os.makedirs(output_dir, exist_ok=True)

    index = pd.read_csv(cfg["data"]["index_csv"])
    dev = index[index["split"].isin(["train", "val"])].reset_index(drop=True)

    fold_rows, cm_rows, oof_all = [], [], []
    for fold in sorted(dev["fold"].unique()):
        val_df = dev[dev["fold"] == fold]
        train_df = dev[dev["fold"] != fold]
        preds = _train_fold(cfg, args.model, args.seed, train_df, val_df, device)
        preds["fold"] = fold
        oof_all.append(preds)

        m = threshold_metrics(preds["y_true"].to_numpy(), preds["y_score"].to_numpy(),
                              threshold=cfg["evaluation"]["decision_threshold"])
        cm = confusion_at_threshold(preds["y_true"].to_numpy(), preds["y_score"].to_numpy(),
                                    threshold=cfg["evaluation"]["decision_threshold"])
        fold_rows.append({"fold": fold, "n_benign": int((val_df["label"] == 0).sum()),
                          "n_malignant": int((val_df["label"] == 1).sum()), **m})
        cm_rows.append({"fold": fold, **cm})
        print(f"Fold {fold}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(output_dir, "per_fold_metrics.csv"), index=False)
    pd.DataFrame(cm_rows).to_csv(os.path.join(output_dir, "per_fold_confusion.csv"),
                                 index=False)
    pd.concat(oof_all).to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    metric_cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    summary = {}
    for c in metric_cols:
        summary[f"{c}_mean"] = folds[c].mean()
        summary[f"{c}_sd"] = folds[c].std(ddof=1)
    pd.Series(summary).to_csv(os.path.join(output_dir, "cv_summary.csv"))
    print(pd.Series(summary))


if __name__ == "__main__":
    main()
