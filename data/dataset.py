# -*- coding: utf-8 -*-
"""
Dataset and preprocessing pipeline for MSDR-Net.

Expected data layout (patient-level PNG images, 16-bit or 8-bit):

    <root>/<split>/<class_name>/<patient_id>.png

    e.g.  data/train/benign/P0001.png
          data/train/malignant/P0123.png
          data/val/benign/P0007.png
          data/test/malignant/P0456.png

Alternatively, a CSV manifest with columns [path, label, split] can be used
(see ``--manifest`` in train.py / evaluate.py). Labels: 0 = benign,
1 = malignant.

Preprocessing (Section 2.2 of the manuscript):
    1. Percentile clipping to the 0.5th-99.5th intensity range (per image).
    2. Image-wise Z-score normalization (zero mean, unit variance).
       NOTE: Z-score normalization is ALWAYS applied before CLAHE.
    3. CLAHE (tile size 8x8, clip limit 2.0) for edge enhancement.
    4. Resize to 224x224 with bilinear interpolation.
    5. Single-channel grayscale replicated into 3 channels.
"""

import os
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ("benign", "malignant")


def percentile_clip(img: np.ndarray, low: float = 0.5,
                    high: float = 99.5) -> np.ndarray:
    """Clip intensities to the [low, high] percentile range of the image."""
    lo, hi = np.percentile(img, (low, high))
    return np.clip(img, lo, hi)


def zscore_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Image-wise Z-score normalization: (x - mu) / sigma."""
    mu = float(img.mean())
    sigma = float(img.std())
    return (img.astype(np.float32) - mu) / (sigma + eps)


def apply_clahe(img: np.ndarray, tile_size: int = 8,
                clip_limit: float = 2.0) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    Applied AFTER Z-score normalization (the image is rescaled to [0, 255]
    first, then CLAHE is applied, following the manuscript pipeline).
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img8 = ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    out = clahe.apply(img8).astype(np.float32) / 255.0
    return out


def preprocess_image(img: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Full preprocessing chain for a single grayscale image.

    Returns a float32 array of shape (3, image_size, image_size).
    """
    if img.ndim == 3:  # collapse any channel dimension to grayscale
        img = img[..., 0]
    img = img.astype(np.float32)
    img = percentile_clip(img)                 # 0.5th-99.5th percentile
    img = zscore_normalize(img)                # Z-score BEFORE CLAHE
    img = apply_clahe(img, tile_size=8, clip_limit=2.0)
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_LINEAR)
    img = np.repeat(img[None, :, :], 3, axis=0)  # replicate to 3 channels
    return img.astype(np.float32)


def load_grayscale(path: str) -> np.ndarray:
    """Load an image as grayscale, preserving 16-bit depth if present."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img


class SpinalTumorDataset(Dataset):
    """Patient-level spinal tumor dataset.

    Args:
        samples: list of (image_path, label) tuples; label in {0, 1}.
        image_size: spatial size to which ROIs are resized (224).
        transform: optional callable applied to the preprocessed tensor
            (used for training-time data augmentation).
    """

    def __init__(self, samples: Sequence[Tuple[str, int]],
                 image_size: int = 224,
                 transform: Optional[Callable] = None):
        self.samples = list(samples)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = load_grayscale(path)
        img = preprocess_image(img, self.image_size)  # (3, H, W) float32
        x = torch.from_numpy(img)
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(label, dtype=torch.long)

    @property
    def labels(self) -> List[int]:
        return [y for _, y in self.samples]


def scan_split_dir(root: str, split: str) -> List[Tuple[str, int]]:
    """Scan <root>/<split>/<class_name>/*.png into (path, label) tuples."""
    samples: List[Tuple[str, int]] = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(root, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


def load_manifest(manifest_path: str, split: str) -> List[Tuple[str, int]]:
    """Load (path, label) tuples for one split from a CSV manifest with
    columns [path, label, split]."""
    import pandas as pd

    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split]
    return list(zip(df["path"].tolist(), df["label"].astype(int).tolist()))


def compute_class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-class-frequency weights: w_c = N / (K * N_c).

    NOTE: the exact class-weight constants used in the paper are redacted
    (see private_params.py); when available they take precedence, otherwise
    the weights are recomputed here from the training-split frequencies.
    """
    try:
        from private_params import resolve as _resolve_private
        w_neg = _resolve_private("CLASS_WEIGHT_NEG")
        w_pos = _resolve_private("CLASS_WEIGHT_POS")
        if w_neg is not None and w_pos is not None:
            return torch.tensor([float(w_neg), float(w_pos)],
                                dtype=torch.float32)
    except Exception:
        pass

    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
