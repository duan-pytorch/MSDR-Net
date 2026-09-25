

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def build_train_transform() -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.10, contrast=0.10),
        T.ToTensor(),
    ])


def build_eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor()])


class SpinalTumorDataset(Dataset):
    """One image per patient; labels: 0 = benign, 1 = malignant (positive)."""

    def __init__(self, index_csv: str, image_root: str, split: str,
                 transform=None, extension: str = ".npy"):
        index = pd.read_csv(index_csv)
        self.df = index[index["split"] == split].reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.extension = extension

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = os.path.join(self.image_root, str(row["image_path"]))
        base, ext = os.path.splitext(path)
        if self.extension == ".npy":
            img = np.load(base + ".npy")  # H x W x 3 float32 in [0, 1]
        else:
            img = np.asarray(
                __import__("PIL.Image", fromlist=["Image"]).open(path)
            ).astype(np.float32) / 65535.0
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, int(row["label"]), str(row["patient_id"])


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse class frequency weights: w_c = N / (n_classes * N_c).

    For the manuscript training set (350 cases: 273 benign, 77 malignant) this
    yields w_neg = 0.64 and w_pos = 2.27.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
