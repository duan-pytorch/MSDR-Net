# MSDR-Net

Official implementation of **"Multi-Scale Convolution with Efficient Channel Attention Improves Benign–Malignant Classification of Spinal Tumors on T2-Weighted Fat-Suppressed MRI"**.

MSDR-Net is a multi-scale deep residual network with efficient channel attention (ECA) for automated benign–malignant classification of spinal tumors on sagittal T2-weighted fat-suppressed MRI.

## Model overview

- **Input**: single sagittal T2-weighted fat-suppressed MR image, ROI-cropped and resized to 224 × 224 × 3.
- **Stem**: 3 × 3 Conv + BN + ReLU (64 channels).
- **Encoder**: five stages of **Multi-Branch Residual Blocks (MBRB)** followed by **ECA** modules; stages are linked by 3 × 3 stride-2 convolutions that halve the spatial size and double the channels (64 → 128 → 256 → 512 → 1024).
- **MBRB**: parallel 1 × 1 / 3 × 3 / 5 × 5 convolutional branches fused by element-wise addition.
- **ECA**: global average pooling → adaptive 1D convolution (kernel size k = 3 for Stage 1, k = 5 for Stages 2–5, from γ = 2, b = 1) → sigmoid channel weights; the recalibrated feature map is fused with the block input through an identity residual connection.
- **Head**: global average pooling → MLP (1024 → 256 → 2) with dropout (0.5) → Softmax.

Parameter counts (verifiable with `eval/efficiency.py`):

| Configuration | Parameters |
|---|---|
| Complete MSDR-Net (multi-scale + ECA) | 55.4M |
| Multi-scale, ECA removed | 55.4M |
| Single-path + ECA | ~19.1M |
| Single-path, ECA removed | ~19.1M |

## Repository structure

```
MSDR-Net/
├── configs/
│   └── default_config.yaml   # full training configuration and hyperparameter search space
├── models/
│   ├── msdr_net.py           # MSDR-Net and the 2 x 2 factorial ablation variants
│   └── baselines.py          # ResNet50, AlexNet, EfficientNet-B3, ConvNeXt-Tiny, Swin-T, Lead-CNN
├── data/
│   ├── dicom_convert.py      # DICOM rescale (slope/intercept), percentile clipping, 16-bit PNG export
│   ├── roi_crop.py           # rule-based ROI cropping (no trainable parameters)
│   ├── preprocess.py         # Z-score normalization, CLAHE, resize, channel replication
│   └── dataset.py            # PyTorch datasets and the training-only augmentation pipeline
├── train.py                  # single-run training (3 independent seeds; best-validation-AUC checkpointing)
├── cross_validation.py       # five-fold stratified cross-validation stability assessment
├── external_validation.py    # leave-one-center-out (Institution A -> Institution B) experiment
├── eval/
│   ├── metrics.py            # accuracy/sensitivity/specificity/precision/F1 (Clopper-Pearson), AUC (Hanley-McNeil)
│   ├── paired_bootstrap.py   # paired, patient-level JOINT bootstrap CIs (50,000 resamples, seed 2025)
│   ├── delong.py             # DeLong test for correlated ROC curves
│   ├── mcnemar_exact.R       # exact two-sided McNemar test (R package exact2x2, v1.6.6)
│   ├── calibration.py        # Brier score, calibration slope/intercept, ECE (10 bins), temperature scaling
│   ├── dca.py                # decision curve analysis (net benefit vs. treat-all / treat-none)
│   └── efficiency.py         # parameter count, MACs (fvcore, 1 MAC = 1 FLOP), GPU/CPU latency protocol
├── weights/
│   └── README.md             # trained model weights (download / placement instructions)
├── requirements.txt
└── LICENSE
```

## Environment

```bash
pip install -r requirements.txt
```

Developed and tested with PyTorch 2.1.0 on a single NVIDIA GeForce RTX 4080 (16 GB). Automatic mixed precision (AMP) is used during training. The exact McNemar test additionally requires R with the `exact2x2` package (v1.6.6).

## Data availability

The de-identified imaging data are **not publicly available**, owing to patient privacy protections and the constraints of the ethics approval and the inter-institutional data use agreement. Patient-level data, including de-identified test-set predictions and data partition identifiers, are likewise not shared. To train or evaluate the models on your own data, prepare a CSV index as described in `data/README.md` (see below) and follow the pipeline scripts.

## Reproducing the experiments

The full protocol follows the five-stage, chronological model-development pipeline described in the manuscript (Section 2.6): the test set is locked first, configuration selection is performed by five-fold cross-validation on the combined training/validation set, the frozen configuration is trained with three independent seeds (2021/2022/2023) with best-validation-AUC checkpointing, the cross-validation stability assessment is re-run once per fold with a single prespecified seed (2024) only after freezing, and the locked model is evaluated on the test set exactly once.

1. **Data preparation**
   ```bash
   python data/dicom_convert.py --input /path/to/dicom_root --output /path/to/png_root
   python data/roi_crop.py --index index.csv --images /path/to/png_root --output /path/to/roi_root
   python data/preprocess.py --index roi_index.csv --images /path/to/roi_root --output /path/to/processed_root
   ```
   All preprocessing parameters (normalization statistics, CLAHE parameters, resize interpolation) are serialized to JSON alongside the outputs.

2. **Training (one run per seed)**
   ```bash
   python train.py --config configs/default_config.yaml --model msdr_net --seed 2021
   python train.py --config configs/default_config.yaml --model msdr_net --seed 2022
   python train.py --config configs/default_config.yaml --model msdr_net --seed 2023
   ```
   The run with the highest validation AUC is carried forward to test-set evaluation.

3. **Five-fold cross-validation stability assessment**
   ```bash
   python cross_validation.py --config configs/default_config.yaml --model msdr_net --seed 2024
   ```

4. **Test-set evaluation and statistical analysis**
   ```bash
   python eval/metrics.py --predictions predictions_msdr_net_test.csv
   python eval/paired_bootstrap.py --reference predictions_msdr_net_test.csv --comparators predictions_convnext_test.csv ...
   python eval/delong.py --reference predictions_msdr_net_test.csv --comparators predictions_convnext_test.csv ...
   Rscript eval/mcnemar_exact.R predictions_msdr_net_test.csv predictions_convnext_test.csv
   python eval/calibration.py --val predictions_msdr_net_val.csv --test predictions_msdr_net_test.csv
   python eval/dca.py --predictions predictions_msdr_net_test.csv
   ```

5. **Computational efficiency**
   ```bash
   python eval/efficiency.py --config configs/default_config.yaml --model msdr_net
   ```

6. **Ablation variants (2 x 2 factorial design)**
   ```bash
   python train.py --config configs/default_config.yaml --model msdr_net_full          # multi-scale + ECA
   python train.py --config configs/default_config.yaml --model msdr_net_no_eca        # multi-scale, ECA removed
   python train.py --config configs/default_config.yaml --model msdr_net_single        # single-path, ECA removed
   python train.py --config configs/default_config.yaml --model msdr_net_single_eca    # single-path + ECA
   ```

7. **Leave-one-center-out external validation**
   ```bash
   python external_validation.py --config configs/default_config.yaml --model msdr_net
   ```

## Note on the paired bootstrap

`eval/paired_bootstrap.py` implements **patient-level joint resampling**: in every resample, patients are drawn with replacement and both models' metrics are recomputed within the same resample, preserving the paired discordance structure. (An earlier implementation that resampled the two models' prediction vectors independently has been corrected; see the revision history of the manuscript.)

## Trained weights

Trained weights for the runs carried forward to test-set evaluation are distributed via GitHub Releases (see `weights/README.md`).

## License

This project is released under the MIT License (see `LICENSE`).

## Citation

If you find this code useful, please cite our manuscript (citation details will be updated upon publication).
