

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLACHE_TILE_SIZE = 8        # 8 x 8 tiles
CLAHE_CLIP_LIMIT = 2.0
OUTPUT_SIZE = 224           # bilinear resize to 224 x 224


def zscore(img: np.ndarray) -> np.ndarray:
    mean, std = float(img.mean()), float(img.std())
    return (img - mean) / std if std > 0 else img - mean


def preprocess_image(arr: np.ndarray) -> np.ndarray:
    """Apply Z-score -> CLAHE -> resize -> 3-channel replication."""
    arr = arr.astype(np.float64)
    arr = zscore(arr)

    # Rescale to 16-bit for CLAHE (lossless linear mapping).
    lo, hi = float(arr.min()), float(arr.max())
    arr16 = ((arr - lo) / (hi - lo) * 65535.0).round().astype(np.uint16) if hi > lo \
        else np.zeros(arr.shape, dtype=np.uint16)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=(CLACHE_TILE_SIZE, CLACHE_TILE_SIZE))
    arr16 = clahe.apply(arr16)

    out = cv2.resize(arr16, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    out = out.astype(np.float32) / 65535.0
    return np.stack([out, out, out], axis=-1)  # H x W x 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="CSV index (patient_id, image_path)")
    parser.add_argument("--images", required=True, help="Root directory of ROI crops")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    index = pd.read_csv(args.index)
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Preprocessing"):
        arr = np.asarray(Image.open(os.path.join(args.images, str(row["image_path"]))))
        out = preprocess_image(arr)
        # Store as float32 .npy to preserve exact values for training.
        np.save(os.path.join(args.output, f"{row['patient_id']}.npy"), out)

    params = {
        "normalization": "image-wise Z-score (before CLAHE)",
        "clahe": {"tile_size": CLACHE_TILE_SIZE, "clip_limit": CLAHE_CLIP_LIMIT,
                  "bit_depth": "16-bit intermediate"},
        "resize": {"size": OUTPUT_SIZE, "interpolation": "bilinear"},
        "channels": "single grayscale channel replicated to three",
    }
    with open(os.path.join(args.output, "preprocess_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"Processed {len(index)} images to {args.output}")


if __name__ == "__main__":
    main()
