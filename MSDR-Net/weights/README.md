# Trained model weights

Place the trained checkpoints in this directory (or download them from the
GitHub Releases page of this repository). Because FP32 checkpoints exceed
GitHub's per-file git limit (e.g., MSDR-Net, 55.4M parameters, ~220 MB), the
weights are distributed as **GitHub Release assets** rather than tracked files.

## Expected files

One checkpoint per architecture, corresponding to the run carried forward to
the single test-set evaluation (highest validation AUC among seeds
2021/2022/2023; manuscript Supplementary Table S8):

| File | Architecture | Run seed |
|---|---|---|
| `msdr_net_seed2022_best.pth` | MSDR-Net (complete) | 2022 |
| `convnext_tiny_seed2022_best.pth` | ConvNeXt-Tiny | 2022 |
| `efficientnet_b3_seed2022_best.pth` | EfficientNet-B3 | 2022 |
| `resnet50_seed2022_best.pth` | ResNet50 | 2022 |
| `swin_t_seed2021_best.pth` | Swin-T | 2021 |
| `lead_cnn_seed2022_best.pth` | Lead-CNN | 2022 |
| `alexnet_seed2022_best.pth` | AlexNet | 2022 |

Ablation variants (2 x 2 factorial design, manuscript Table 6):

| File | Configuration |
|---|---|
| `msdr_net_no_eca_best.pth` | multi-scale, ECA removed |
| `msdr_net_single_best.pth` | single-path, ECA removed |
| `msdr_net_single_eca_best.pth` | single-path + ECA |

Leave-one-center-out external validation:

| File | Description |
|---|---|
| `msdr_net_institution_a_best.pth` | locked model developed entirely on Institution A |

Each checkpoint is a `torch.save` dictionary with keys `model_state_dict`,
`epoch`, `val_auc`, `seed`, and `model`.

## Loading

```python
import torch
from models import build_model

model = build_model("msdr_net")
checkpoint = torch.load("weights/msdr_net_seed2022_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
