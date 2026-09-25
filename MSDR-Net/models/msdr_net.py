

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):


    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels)) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.kernel_size = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                      # B x C x 1 x 1
        y = y.squeeze(-1).transpose(-1, -2)       # B x 1 x C
        y = self.conv(y)                          # B x 1 x C
        y = y.transpose(-1, -2).unsqueeze(-1)     # B x C x 1 x 1
        return x * torch.sigmoid(y)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MBRB(nn.Module):


    def __init__(self, channels: int, use_multiscale: bool = True, use_eca: bool = True):
        super().__init__()
        if use_multiscale:
            self.branches = nn.ModuleList([
                ConvBNReLU(channels, channels, kernel_size=1),
                ConvBNReLU(channels, channels, kernel_size=3),
                ConvBNReLU(channels, channels, kernel_size=5),
            ])
        else:
            self.branches = nn.ModuleList([ConvBNReLU(channels, channels, kernel_size=3)])
        self.eca = ECA(channels) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        out = self.eca(out)
        return out + x


class MSDRNet(nn.Module):
    """Five-level encoder with MBRB + ECA stages and an MLP classification head."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 widths=(64, 128, 256, 512, 1024),
                 use_multiscale: bool = True, use_eca: bool = True):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, widths[0], kernel_size=3)

        stages = []
        prev = widths[0]
        for i, c in enumerate(widths):
            if i > 0:
                # Downsampling at the entry of Stages 2-5: 3x3 conv, stride 2, padding 1.
                stages.append(ConvBNReLU(prev, c, kernel_size=3, stride=2))
            stages.append(MBRB(c, use_multiscale=use_multiscale, use_eca=use_eca))
            prev = c
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Kaiming initialization (random init; no ImageNet-pretrained weights
        # are available for this custom architecture).
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def build_model(name: str, **kwargs) -> nn.Module:
    """Model factory. `name` selects the complete model or an ablation variant."""
    variants = {
        "msdr_net": dict(use_multiscale=True, use_eca=True),
        "msdr_net_full": dict(use_multiscale=True, use_eca=True),
        "msdr_net_no_eca": dict(use_multiscale=True, use_eca=False),
        "msdr_net_single_eca": dict(use_multiscale=False, use_eca=True),
        "msdr_net_single": dict(use_multiscale=False, use_eca=False),
    }
    if name not in variants:
        raise KeyError(f"Unknown MSDR-Net variant '{name}'. Choose from {sorted(variants)}.")
    return MSDRNet(**variants[name], **kwargs)
