"""Publication-quality visualization utilities for fNIRS artifact removal results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Default figure settings for 300 DPI publication figures
FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"


def plot_waveform_comparison(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    fs: float = 10.0,
    channel_idx: int = 0,
    title: str = "fNIRS Signal Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot noisy, denoised, and clean waveforms on the same axes.

    Args:
        noisy: Contaminated signal of shape (n_channels, n_timepoints).
        denoised: Processed output of same shape.
        clean: Artifact-free reference of same shape.
        fs: Sampling frequency in Hz.
        channel_idx: Which channel to plot.
        title: Figure title.
        save_path: If provided, save the figure to this path at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError


def plot_spectrum(
    signals: Dict[str, np.ndarray],
    fs: float = 10.0,
    channel_idx: int = 0,
    freq_range: Tuple[float, float] = (0.0, 1.0),
    title: str = "Power Spectral Density",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot power spectral density for multiple signals on the same axes.

    Args:
        signals: Dict mapping label → signal array of shape (n_channels, n_timepoints).
        fs: Sampling frequency in Hz.
        channel_idx: Which channel to plot.
        freq_range: (low, high) frequency range in Hz to display.
        title: Figure title.
        save_path: If provided, save the figure at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    title: str = "Method Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart comparing all methods across all metrics.

    Args:
        metrics: Dict mapping method_name → {metric_name: value}.
        metric_names: Subset of metrics to plot. If None, all are plotted.
        title: Figure title.
        save_path: If provided, save the figure at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError


def plot_subject_results(
    per_subject_metrics: Dict[str, Dict[str, np.ndarray]],
    metric: str = "delta_snr",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Boxplot of per-subject metric scores for each method.

    Args:
        per_subject_metrics: Dict mapping method_name → {metric_name: array_of_scores}.
        metric: Which metric to visualise.
        title: Figure title (auto-generated from metric name if None).
        save_path: If provided, save the figure at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot train / validation loss curves.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists returned by Trainer.fit().
        save_path: If provided, save the figure at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError


def plot_ablation_results(
    ablation_metrics: Dict[str, Dict[str, float]],
    metric: str = "delta_snr",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart showing ablation study results (A1 → A5).

    Args:
        ablation_metrics: Dict mapping ablation_id (e.g. 'A1') → metrics dict.
        metric: Metric to display on y-axis.
        save_path: If provided, save the figure at FIGURE_DPI.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError
