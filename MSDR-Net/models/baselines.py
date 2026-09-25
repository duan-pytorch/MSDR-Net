

import torch.nn as nn
from torchvision import models as tvm


def _replace_head(num_features: int, num_classes: int = 2) -> nn.Sequential:
    """Binary classification head matching the two-class output of MSDR-Net."""
    return nn.Linear(num_features, num_classes)


def build_baseline(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = _replace_head(model.fc.in_features, num_classes)
    elif name == "alexnet":
        weights = tvm.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.alexnet(weights=weights)
        model.classifier[-1] = _replace_head(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b3":
        weights = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b3(weights=weights)
        model.classifier[-1] = _replace_head(model.classifier[-1].in_features, num_classes)
    elif name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.convnext_tiny(weights=weights)
        model.classifier[-1] = _replace_head(model.classifier[-1].in_features, num_classes)
    elif name == "swin_t":
        weights = tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.swin_t(weights=weights)
        model.head = _replace_head(model.head.in_features, num_classes)
    elif name == "lead_cnn":
        # Lead-CNN (Khan et al., 2025) is used as published; if the authors'
        # implementation is unavailable, a faithful reproduction should be
        # placed here and instantiated with the same head replacement pattern.
        raise NotImplementedError(
            "Lead-CNN is not part of torchvision. Provide the reproduction in "
            "models/lead_cnn.py and register it here."
        )
    else:
        raise KeyError(f"Unknown baseline '{name}'.")
    return model


BASELINE_NAMES = ["resnet50", "alexnet", "efficientnet_b3", "convnext_tiny",
                  "swin_t", "lead_cnn"]
