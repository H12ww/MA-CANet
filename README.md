# MA-CANet: Multi-scale Attention-enhanced Convolutional Network for fNIRS Motion Artifact Removal

A deep learning framework for removing motion artifacts from functional near-infrared spectroscopy (fNIRS) signals.

## Overview

Motion artifacts are a major challenge in fNIRS-based brain-computer interfaces and neuroimaging studies. MA-CANet proposes a novel U-Net-style encoder-decoder architecture enhanced with:

- **Multi-Scale Convolutional (MS-Conv) stem** — captures artifacts at multiple timescales simultaneously using parallel 1D convolutions (kernels 3, 7, 15, 31)
- **Squeeze-and-Excitation (SE) attention blocks** — adaptively re-weights channels to focus on artifact-free signal components
- **Hybrid loss** — combines MSE, frequency-domain loss, and SSIM for perceptually faithful reconstruction

MA-CANet is benchmarked against six baseline methods (bandpass filter, wavelet thresholding, spline interpolation, TDDR, PCA, and DAE) across two public datasets.

## Environment Setup

**Requirements**: Python 3.10+, CUDA-capable GPU recommended.

```bash
# Create conda environment
conda create -n fnirs python=3.10
conda activate fnirs

# Install dependencies
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.x, MNE-NIRS, MNE-Python, SciPy, PyWavelets, h5py, wfdb, NumPy, Matplotlib, Seaborn, PyYAML, TensorBoard.

## Directory Structure

```
fnirs-artifact-removal/
├── CLAUDE.md                     # Project context for Claude Code
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── configs/
│   └── default.yaml              # Hyperparameters, paths, training config
├── data/
│   ├── raw/
│   │   ├── BIDSdata_fNIRS_motion_artifact/   # Dataset①: 20 subjects (.snirf)
│   │   └── motion-artifact-contaminated-fnirs-and-eeg-data-1.0.0/  # Dataset②: PhysioBank
│   ├── processed/                # Preprocessed numpy arrays
│   └── semi_synthetic/           # Generated (noisy, clean) training pairs
├── src/
│   ├── data/
│   │   ├── snirf_reader.py       # BIDS .snirf reader
│   │   ├── wfdb_reader.py        # PhysioBank WFDB reader
│   │   ├── preprocess.py         # OD conversion, Beer-Lambert, normalization
│   │   ├── dataset.py            # PyTorch Dataset classes
│   │   └── augmentation.py       # Artifact generation for semi-synthetic data
│   ├── models/
│   │   ├── macanet.py            # MA-CANet main model + ablation variants
│   │   ├── modules.py            # MS-Conv, SE, Encoder/Decoder blocks
│   │   └── baselines.py          # All six baseline method wrappers
│   ├── training/
│   │   ├── trainer.py            # Training loop with early stopping
│   │   └── losses.py             # HybridLoss (MSE + Freq + SSIM)
│   └── evaluation/
│       ├── metrics.py            # ΔSNR, RMSE, Pearson r, SSIM, η
│       └── visualize.py          # Publication-quality figures
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_preprocess.py
│   ├── 03_generate_semi_synthetic.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_compare_baselines.py
│   └── 07_ablation_study.py
├── notebooks/
│   └── eda.ipynb
└── outputs/
    ├── checkpoints/              # Saved model weights (.pth)
    ├── logs/                     # TensorBoard logs
    └── figures/                  # Generated plots (300 DPI)
```

## Quick Start

### 1. Explore the raw data

```bash
python scripts/01_explore_data.py
```

### 2. Preprocess (raw → processed numpy arrays)

```bash
python scripts/02_preprocess.py --config configs/default.yaml
```

### 3. Generate semi-synthetic training pairs

```bash
python scripts/03_generate_semi_synthetic.py --config configs/default.yaml
```

### 4. Train MA-CANet

```bash
python scripts/04_train.py --config configs/default.yaml
# Monitor with TensorBoard:
tensorboard --logdir outputs/logs
```

### 5. Evaluate on the test set

```bash
python scripts/05_evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pth
```

### 6. Compare with baselines

```bash
python scripts/06_compare_baselines.py --config configs/default.yaml
```

### 7. Run ablation study

```bash
python scripts/07_ablation_study.py --config configs/default.yaml
```

## Datasets

| Dataset | Subjects | Format | Usage |
|---------|----------|--------|-------|
| BIDSdata_fNIRS_motion_artifact | 20 | BIDS .snirf | Training & evaluation |
| PhysioBank motion-artifact fNIRS | 9 | WFDB .dat/.hea | Cross-dataset generalization only |

Event types in Dataset①: `HT` (clean ground truth), `SM` (mild artifacts), `LM` (severe artifacts).

## Evaluation Metrics

| Metric | Direction |
|--------|-----------|
| ΔSNR (dB) — SNR improvement | Higher is better |
| RMSE | Lower is better |
| Pearson r | Closer to 1.0 |
| SSIM | Closer to 1.0 |
| η (%) — Residual artifact ratio | Lower is better |

Statistical significance tested via Wilcoxon signed-rank test (paired t-test as alternative).

## Reproducibility

All random seeds are fixed to `42`. Experiments are fully configured via `configs/default.yaml` with no hardcoded hyperparameters in source code.
