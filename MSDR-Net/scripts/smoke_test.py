

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msdr_net import ECA, build_model  # noqa: E402
from data.dataset import compute_class_weights  # noqa: E402


def main() -> None:
    print("== MSDR-Net variants ==")
    x = torch.randn(2, 3, 224, 224)
    for name in ["msdr_net_full", "msdr_net_no_eca", "msdr_net_single_eca",
                 "msdr_net_single"]:
        model = build_model(name)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:22s} params = {n_params / 1e6:8.3f}M   output = {tuple(out.shape)}")
        assert out.shape == (2, 2)

    print("== ECA adaptive kernel sizes ==")
    expected = {64: 3, 128: 5, 256: 5, 512: 5, 1024: 5}
    for channels, k_expected in expected.items():
        k = ECA(channels).kernel_size
        print(f"  channels = {channels:5d} -> k = {k} (expected {k_expected})")
        assert k == k_expected

    print("== Class weights (350-case training set: 273 benign, 77 malignant) ==")
    labels = np.array([0] * 273 + [1] * 77)
    weights = compute_class_weights(labels)
    print(f"  w_neg (benign) = {weights[0]:.2f}   w_pos (malignant) = {weights[1]:.2f}")
    assert abs(weights[0].item() - 0.64) < 0.01
    assert abs(weights[1].item() - 2.27) < 0.01

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
