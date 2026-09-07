# -*- coding: utf-8 -*-
"""Common utilities: reproducibility, logging, checkpoint I/O."""

import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int = 2022, deterministic: bool = True):
    """Set random seeds; cuDNN deterministic mode on, benchmark mode off."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    return ckpt


class AverageMeter:
    """Track a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count = 0.0, 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        return False
