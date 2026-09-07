# -*- coding: utf-8 -*-
"""
Single-image inference with a trained MSDR-Net checkpoint.

Example:
    python predict.py --checkpoint checkpoints/msdr_net_full_seed2022_best.pt \
        --image path/to/roi.png
"""

import argparse

import torch

from data import CLASS_NAMES, load_grayscale, preprocess_image
from models import msdr_net
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="MSDR-Net single-image inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True,
                   help="path to a grayscale ROI image (png/jpg/tif)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--ablation", type=str, default=None,
                   choices=[None, "no_eca", "single_branch"])
    p.add_argument("--seed", type=int, default=2022)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    model = msdr_net(num_classes=2, ablation=args.ablation).to(device)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()

    img = load_grayscale(args.image)
    x = torch.from_numpy(preprocess_image(img, args.image_size))
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(x)[0].cpu().numpy()

    pred = int(probs.argmax())
    print(f"Image: {args.image}")
    print(f"Prediction: {CLASS_NAMES[pred]} "
          f"(P(benign)={probs[0]:.4f}, P(malignant)={probs[1]:.4f})")


if __name__ == "__main__":
    main()
