# -*- coding: utf-8 -*-
"""
Training-time data augmentation for MSDR-Net (applied to the training set
only; Section 2.5 of the manuscript):

    * random horizontal flipping (probability 0.5)
    * random rotation within +/-10 degrees
    * random translation of up to 5% of the image dimensions
    * random brightness/contrast perturbation within +/-10%

Implemented on torch tensors of shape (3, H, W) with values in [0, 1].
"""

import random
from typing import Tuple

import torch


class RandomAffine2D:
    """Random rotation (+/- degrees) and translation (fraction of size)."""

    def __init__(self, degrees: float = 10.0, translate: float = 0.05):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.v2.functional as TF

        angle = random.uniform(-self.degrees, self.degrees)
        h, w = x.shape[-2], x.shape[-1]
        max_dx = self.translate * w
        max_dy = self.translate * h
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)
        return TF.affine(x, angle=angle, translate=[dx, dy], scale=1.0,
                         shear=0.0)


class RandomBrightnessContrast:
    """Random brightness/contrast perturbation within +/- `strength`."""

    def __init__(self, strength: float = 0.10):
        self.strength = strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b = 1.0 + random.uniform(-self.strength, self.strength)
        c = 1.0 + random.uniform(-self.strength, self.strength)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * c + mean * b
        return x.clamp(0.0, 1.0)


class TrainAugment:
    """Composed training-time augmentation pipeline."""

    def __init__(self, hflip_p: float = 0.5, degrees: float = 10.0,
                 translate: float = 0.05, bc_strength: float = 0.10):
        self.hflip_p = hflip_p
        self.affine = RandomAffine2D(degrees=degrees, translate=translate)
        self.bc = RandomBrightnessContrast(strength=bc_strength)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.hflip_p:
            x = torch.flip(x, dims=[-1])  # horizontal flip
        x = self.affine(x)
        x = self.bc(x)
        return x
