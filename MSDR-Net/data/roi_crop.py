

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

INITIAL_SIZE = 160          # initial crop size in pixels
STEP = 4                    # expansion step in pixels per direction
GRADIENT_THRESHOLD = 0.05   # normalized intensity units
MAX_SCALE = 1.5             # maximum expansion relative to the initial size


def _read_normalized(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path)).astype(np.float64)
    max_val = arr.max()
    return arr / max_val if max_val > 0 else arr


def _edge_gradient(img: np.ndarray, top: int, bottom: int, left: int, right: int,
                   direction: str) -> float:
    """Mean absolute intensity gradient along the given (outer) crop edge."""
    if direction == "up" and top > 0:
        return float(np.mean(np.abs(img[top, left:right] - img[top - 1, left:right])))
    if direction == "down" and bottom < img.shape[0]:
        return float(np.mean(np.abs(img[bottom - 1, left:right] - img[bottom, left:right])))
    if direction == "left" and left > 0:
        return float(np.mean(np.abs(img[top:bottom, left] - img[top:bottom, left - 1])))
    if direction == "right" and right < img.shape[1]:
        return float(np.mean(np.abs(img[top:bottom, right - 1] - img[top:bottom, right])))
    return np.inf  # image border reached: no further expansion possible


def rule_based_crop(img: np.ndarray, seed_x: float, seed_y: float,
                    initial_size: int = INITIAL_SIZE, step: int = STEP,
                    threshold: float = GRADIENT_THRESHOLD,
                    max_scale: float = MAX_SCALE) -> np.ndarray:
    h, w = img.shape
    half = initial_size // 2
    cx, cy = int(round(seed_x)), int(round(seed_y))
    top, bottom = max(0, cy - half), min(h, cy + half)
    left, right = max(0, cx - half), min(w, cx + half)

    max_half = int(half * max_scale)
    expandable = {"up": True, "down": True, "left": True, "right": True}
    while any(expandable.values()):
        for direction in list(expandable):
            if not expandable[direction]:
                continue
            if direction in ("up", "down"):
                if (bottom - top) // 2 >= max_half:
                    expandable[direction] = False
                    continue
            else:
                if (right - left) // 2 >= max_half:
                    expandable[direction] = False
                    continue
            grad = _edge_gradient(img, top, bottom, left, right, direction)
            if grad < threshold:
                expandable[direction] = False
                continue
            if direction == "up":
                top = max(0, top - step)
            elif direction == "down":
                bottom = min(h, bottom + step)
            elif direction == "left":
                left = max(0, left - step)
            else:
                right = min(w, right + step)
    return img[top:bottom, left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="CSV index with seed_x/seed_y columns")
    parser.add_argument("--images", required=True, help="Root directory of 16-bit PNG images")
    parser.add_argument("--output", required=True, help="Output directory for ROI crops")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    index = pd.read_csv(args.index)
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Cropping ROIs"):
        img = _read_normalized(os.path.join(args.images, str(row["image_path"])))
        crop = rule_based_crop(img, row["seed_x"], row["seed_y"])
        out = (np.clip(crop, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
        Image.fromarray(out, mode="I;16").save(
            os.path.join(args.output, f"{row['patient_id']}.png"))
    print(f"Cropped {len(index)} ROIs to {args.output}")


if __name__ == "__main__":
    main()
