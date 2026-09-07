# MSDR-Net

**Multi-Scale Deep Residual Network with Efficient Channel Attention for Benign–Malignant Classification of Spinal Tumors on T2-Weighted Fat-Suppressed MRI**

This repository contains the implementation of **MSDR-Net**, described in the manuscript *"Multi-Scale Convolution with Efficient Channel Attention Improves Benign-Malignant Classification of Spinal Tumors on T2-Weighted Fat-Suppressed MRI"*.



## Architecture overview

MSDR-Net is a five-level encoder that accepts 224 × 224 × 3 ROI images and outputs binary (benign/malignant) classification probabilities:

| Block | Operation | Output size |
|---|---|---|
| Input | — | 224×224×3 |
| Stem | 3×3 Conv + BN + ReLU | 224×224×64 |
| Stage 1 | MBRB (1×1 ‖ 3×3 ‖ 5×5) + ECA + residual | 224×224×64 |
| Down 1 | 3×3 Conv, stride 2, padding 1 + BN + ReLU | 112×112×128 |
| Stage 2 | MBRB + ECA + residual | 112×112×128 |
| Down 2 | 3×3 Conv, stride 2, padding 1 + BN + ReLU | 56×56×256 |
| Stage 3 | MBRB + ECA + residual | 56×56×256 |
| Down 3 | 3×3 Conv, stride 2, padding 1 + BN + ReLU | 28×28×512 |
| Stage 4 | MBRB + ECA + residual | 28×28×512 |
| Down 4 | 3×3 Conv, stride 2, padding 1 + BN + ReLU | 14×14×1024 |
| Stage 5 | MBRB + ECA + residual | 14×14×1024 |
| Classifier | Global average pooling + MLP (→ hidden → 2) + Softmax | 2 |

- **MBRB (Multi-Branch Residual Block):** parallel 1×1, 3×3, and 5×5 convolutional pathways capture point-level features, local textures, and regional context; the three scales are fused by element-wise addition.
- **ECA (Efficient Channel Attention):** global average pooling → adaptive 1-D convolution (kernel size *k* mapped from the channel number, without dimensionality reduction) → sigmoid gating → channel-wise rescaling, followed by an identity residual connection.

Ablation variants (matching the manuscript) are supported via `--ablation {no_eca, single_branch}`.

## Repository layout

```
MSDR-Net/
├── models/msdr_net.py      # ECA, MBRB, MSDR-Net (and ablation variants)
├── data/dataset.py         # dataset, DICOM-derived preprocessing pipeline
├── data/transforms.py      # training-time data augmentation
├── utils/common.py         # seeds, checkpoints, misc
├── utils/metrics.py        # classification/calibration/DCA metrics
├── train.py                # training (weighted CE, AdamW, MultiStepLR, AMP, early stop)
├── run_cv.py               # five-fold stratified cross-validation
├── evaluate.py             # test-set metrics, calibration, decision curve analysis
├── predict.py              # single-image inference
├── configs/default.yaml    # default configuration
├── private_params.py       # redacted key parameters (see below)
└── docs/architecture.png   # model architecture figure
```

## Data preparation

Patient-level ROI images (grayscale PNG; 16-bit recommended) organized as:

```
data/
├── train/{benign,malignant}/*.png
├── val/{benign,malignant}/*.png
└── test/{benign,malignant}/*.png
```

Alternatively, pass a CSV manifest (`--manifest`) with columns `path,label,split`
(`label`: 0 = benign, 1 = malignant; `split` ∈ {train, val, test}).

The preprocessing chain (per image): 0.5th–99.5th percentile clipping →
image-wise Z-score normalization → CLAHE (8×8 tiles, clip limit 2.0) →
bilinear resize to 224×224 → replication into 3 channels. Z-score
normalization is always applied **before** CLAHE.

## Usage

Install dependencies (Python ≥ 3.9):

```bash
pip install -r requirements.txt
```

Train (three independent runs with seeds 2021/2022/2023 were used in the
paper; keep the run with the highest validation AUC):

```bash
python train.py --data-root ./data --epochs 200 --batch-size 16 --seed 2022
```

Five-fold stratified cross-validation on the combined training+validation
set (single prespecified run per fold, seed 2024):

```bash
python run_cv.py --data-root ./data
```

Evaluate on the locked test set (metrics / calibration / DCA saved as CSV):

```bash
python evaluate.py --data-root ./data \
    --checkpoint checkpoints/msdr_net_full_seed2022_best.pt \
    --temperature-scaling
```

Single-image inference:

```bash
python predict.py --checkpoint checkpoints/msdr_net_full_seed2022_best.pt \
    --image path/to/roi.png
```

## ⚠️ Redacted key parameters

Several key hyperparameters are **intentionally redacted** in this public
release (they appear as `None` in `private_params.py` and as placeholder
values in `configs/default.yaml`), including:

- the ECA adaptive kernel mapping constants (γ, β),
- the MLP classification-head configuration (hidden dimension, dropout),
- the class re-weighting factors of the weighted cross-entropy loss,
- the learning-rate schedule details (initial LR, weight decay, decay milestones and factor).

The released code runs out of the box because every redacted value
automatically falls back to a **reasonable placeholder value** — these
placeholders are *not* necessarily the exact values reported in the paper.

> **If you need these key parameters for reproduction, please contact:**
> **a13995115025@163.com**

## Reproducibility

- Kaiming initialization (no ImageNet-pretrained weights exist for this custom architecture).
- cuDNN deterministic mode enabled, benchmark mode disabled.
- All evaluation at the prespecified decision threshold of 0.5 on the Softmax-normalized malignant-class probability; malignant = positive class.

## Citation

If you find this code useful, please cite our manuscript (citation to be
updated upon publication).

## Disclaimer

This code is provided for research purposes only and is not a certified
medical device. Patient imaging data are **not** included in this
repository; de-identified data are available from the corresponding authors
upon reasonable request, subject to a data use agreement.
