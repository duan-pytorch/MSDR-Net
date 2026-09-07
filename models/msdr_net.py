# -*- coding: utf-8 -*-
"""
MSDR-Net: Multi-Scale Deep Residual Network with Efficient Channel Attention
for benign-malignant classification of spinal tumors on T2-weighted
fat-suppressed MRI.

Architecture (see docs/architecture.png and the manuscript, Table 1 / Table S5):

    Input 224x224x3
      -> Stem: 3x3 Conv (s=1, p=1) + BN + ReLU                [224x224x64]
      -> Stage 1: MBRB (1x1 || 3x3 || 5x5) + ECA + residual   [224x224x64]
      -> Down 1: 3x3 Conv (s=2, p=1) + BN + ReLU              [112x112x128]
      -> Stage 2: MBRB + ECA + residual                       [112x112x128]
      -> Down 2: 3x3 Conv (s=2, p=1) + BN + ReLU              [56x56x256]
      -> Stage 3: MBRB + ECA + residual                       [56x56x256]
      -> Down 3: 3x3 Conv (s=2, p=1) + BN + ReLU              [28x28x512]
      -> Stage 4: MBRB + ECA + residual                       [28x28x512]
      -> Down 4: 3x3 Conv (s=2, p=1) + BN + ReLU              [14x14x1024]
      -> Stage 5: MBRB + ECA + residual                       [14x14x1024]
      -> Global Average Pooling
      -> MLP head (1024 -> hidden -> 2) + Softmax

Ablation variants supported (Section 2.7 / Table 6 of the manuscript):
    * full model:            ablation=None
    * remove ECA:            ablation="no_eca"
    * single-path 3x3 conv:  ablation="single_branch" (ECA removed as well)
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Private / redacted key parameters (see private_params.py for details).
try:
    from private_params import resolve as _resolve_private
except ImportError:  # allow running this file directly from any cwd
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from private_params import resolve as _resolve_private


# -------------------------------------------------------------------------
# Efficient Channel Attention (ECA) module, without dimensionality reduction
# -------------------------------------------------------------------------
class ECAModule(nn.Module):
    """Efficient Channel Attention.

    A global average pooling produces a 1-D channel descriptor; a 1-D
    convolution with an adaptive kernel size ``k`` (shared across channels)
    followed by a sigmoid gate yields channel weights omega; the input
    feature map is rescaled channel-wise:  out = omega (.) F_agg.

    The kernel size is adaptively mapped from the channel number C:

        k = | log2(C) / gamma + beta / gamma |_odd

    NOTE: the exact mapping constants (gamma, beta) are redacted in this
    public release -- reasonable placeholder values are used automatically.
    Contact a13995115025@163.com for the exact key parameters.
    """

    def __init__(self, channels: int, gamma: Optional[float] = None,
                 beta: Optional[float] = None, kernel_size: Optional[int] = None):
        super().__init__()
        if kernel_size is None:
            g = gamma if gamma is not None else _resolve_private("ECA_GAMMA")
            b = beta if beta is not None else _resolve_private("ECA_BETA")
            t = int(abs((math.log2(channels) + b) / g))
            kernel_size = t if t % 2 else t + 1  # nearest odd number
        self.kernel_size = kernel_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)       # (B, 1, C)
        y = self.conv(y)                          # (B, 1, C)
        y = self.sigmoid(y)                       # channel weights omega
        y = y.transpose(-1, -2).unsqueeze(-1)     # (B, C, 1, 1)
        return x * y.expand_as(x)                 # omega (.) F_agg


# -------------------------------------------------------------------------
# Multi-Branch Residual Block (MBRB)
# -------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """Standard convolution followed by BN and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiBranchResidualBlock(nn.Module):
    """Multi-branch residual block with parallel multi-scale convolutions.

    Three parallel convolutional pathways with distinct local receptive
    fields:

        * Path 1 (point-level features): 1x1 convolution
        * Path 2 (local textures):       3x3 convolution
        * Path 3 (regional context):     5x5 convolution

    The three spatial feature maps are fused by element-wise addition
    (multi-scale fusion). The aggregated feature is then recalibrated by an
    ECA module and fused with the block input through an identity-mapping
    residual connection.
    """

    def __init__(self, channels: int, use_eca: bool = True,
                 eca_kernel_size: Optional[int] = None):
        super().__init__()
        self.branch1 = ConvBNReLU(channels, channels, kernel_size=1)
        self.branch3 = ConvBNReLU(channels, channels, kernel_size=3)
        self.branch5 = ConvBNReLU(channels, channels, kernel_size=5)

        self.use_eca = use_eca
        self.eca = ECAModule(channels, kernel_size=eca_kernel_size) if use_eca else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale fusion: element-wise addition of the three pathways.
        f_agg = self.branch1(x) + self.branch3(x) + self.branch5(x)
        if self.eca is not None:
            f_agg = self.eca(f_agg)
        # Identity-mapping residual connection.
        return self.relu(f_agg + x)


class SingleBranchResidualBlock(nn.Module):
    """Ablation variant: single standard 3x3 convolutional path (no ECA).

    Replaces the three multi-scale pathways of the MBRB with a single
    standard 3x3 convolution, as described in the ablation study.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvBNReLU(channels, channels, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


# -------------------------------------------------------------------------
# MSDR-Net
# -------------------------------------------------------------------------
class MSDRNet(nn.Module):
    """Multi-Scale Deep Residual Network (MSDR-Net).

    A five-level encoder; each stage consists of an MBRB followed by an ECA
    module. Between consecutive stages, a 3x3 convolution with stride 2 and
    padding 1 halves the spatial dimensions and doubles the channels
    (64, 128, 256, 512, 1024 across Stages 1-5). After the final stage,
    global average pooling aggregates spatial features and an MLP head
    outputs the binary (benign/malignant) class logits.

    Args:
        in_channels: number of input image channels (3 by default; the
            single-channel grayscale image is replicated to 3 channels).
        num_classes: number of output classes (2: benign / malignant).
        widths: channel width of each encoder stage.
        ablation: None (full model), "no_eca", or "single_branch".
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 widths: Sequence[int] = (64, 128, 256, 512, 1024),
                 ablation: Optional[str] = None):
        super().__init__()
        assert ablation in (None, "no_eca", "single_branch"), \
            f"Unknown ablation mode: {ablation}"

        # --- Stem: 3x3 Conv + BN + ReLU (spatial size preserved) ---------
        self.stem = ConvBNReLU(in_channels, widths[0], kernel_size=3, stride=1)

        # --- Encoder stages ----------------------------------------------
        stages = []
        downsamples = []
        for i, c in enumerate(widths):
            if ablation == "single_branch":
                stages.append(SingleBranchResidualBlock(c))
            else:
                stages.append(MultiBranchResidualBlock(
                    c, use_eca=(ablation != "no_eca")))
            if i < len(widths) - 1:
                # Between consecutive stages: 3x3 Conv, stride=2, padding=1
                # -> spatial size halved, channels doubled.
                downsamples.append(
                    ConvBNReLU(c, widths[i + 1], kernel_size=3, stride=2))
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)

        # --- Classification head: GAP + MLP -------------------------------
        # NOTE: the exact MLP hidden configuration is redacted in this
        # public release; a reasonable placeholder is used automatically.
        # Contact a13995115025@163.com for the exact key parameters.
        hidden = int(_resolve_private("MLP_HIDDEN_DIM"))
        dropout = float(_resolve_private("MLP_DROPOUT"))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization (no ImageNet-pretrained weights exist for
        this custom architecture)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.gap(x)
        return self.classifier(x)  # raw logits; apply Softmax externally

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return Softmax class probabilities (column 1 = malignant)."""
        return torch.softmax(self.forward(x), dim=1)


def msdr_net(num_classes: int = 2, ablation: Optional[str] = None,
             in_channels: int = 3) -> MSDRNet:
    """Factory function mirroring the paper configuration."""
    return MSDRNet(in_channels=in_channels, num_classes=num_classes,
                   ablation=ablation)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for ab in (None, "no_eca", "single_branch"):
        net = msdr_net(ablation=ab)
        n = count_parameters(net)
        print(f"MSDR-Net (ablation={ab}): {n / 1e6:.1f}M parameters")
        x = torch.randn(2, 3, 224, 224)
        y = net(x)
        print(f"  input {tuple(x.shape)} -> output {tuple(y.shape)}")
