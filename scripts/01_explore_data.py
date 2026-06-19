#!/usr/bin/env python3
"""Exploratory Data Analysis for sub-01 fNIRS data.

Outputs (all saved to outputs/figures/):
    01_full_timeseries.png    — full HbO recording with color-coded event segments
    01_waveform_comparison.png — HT vs SM vs LM mean ± std waveforms (HbO and HbR)
    01_channel_snr.png        — per-channel signal variability and SNR

Usage:
    python scripts/01_explore_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; swap to "TkAgg" for live display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.snirf_reader import (
    EventData,
    SNIRFData,
    find_events_file,
    load_events,
    load_snirf,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SUBJECT_ID   = "sub-01"
CHANNEL_IDX  = 0   # S1_D1 (first source-detector pair) for per-channel plots
MOL_TO_UMOL  = 1e6  # display HbO / HbR in µmol/L

BIDS_DIR    = PROJECT_ROOT / "data" / "raw" / "BIDSdata_fNIRS_motion_artifact"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

EVENT_COLORS: dict[str, str] = {
    "BS": "#AAAAAA",   # grey
    "HT": "#2CA02C",   # green
    "SM": "#FF7F0E",   # orange
    "LM": "#D62728",   # red
    "R":  "#1F77B4",   # blue
}
EVENT_LABELS: dict[str, str] = {
    "BS": "Baseline (BS)",
    "HT": "Hand Tapping — no movement (HT, clean)",
    "SM": "Hand Tapping + Small Movement (SM)",
    "LM": "Hand Tapping + Large Movement (LM)",
    "R":  "Rest (R)",
}

DPI        = 300
FS_TITLE   = 13
FS_LABEL   = 11
FS_TICK    = 9
FS_LEGEND  = 8


# ── Statistics ────────────────────────────────────────────────────────────────

def print_statistics(data: SNIRFData, events: EventData) -> None:
    """Print dataset dimensions and event counts to stdout."""
    W = 62
    print("\n" + "=" * W)
    print(f"  DATA SUMMARY — {data.subject_id}")
    print("=" * W)
    print(f"  Sampling rate   : {data.sfreq:.1f} Hz")
    print(f"  Duration        : {data.times[-1]:.1f} s  ({data.times[-1]/60:.1f} min)")
    print(f"  Channel pairs   : {data.n_pairs}  ({data.src_det_pairs})")
    print(f"  Wavelengths     : {data.wavelengths} nm")
    print(f"  raw_intensity   : shape {data.raw_intensity.shape}  (CW amplitude)")
    print(f"  HbO / HbR       : shape {data.hbo.shape}  each  (mol/L via Beer-Lambert)")
    print()
    print(f"  Events  ({events.n_events} total):")
    print(f"  {'Type':<6}  {'Count':>6}  {'Duration (s)':>13}  {'Total (s)':>10}")
    print("  " + "-" * 42)
    for et in ["BS", "HT", "SM", "LM", "R"]:
        segs = events.get_type(et)
        if segs:
            durs = [d for _, d in segs]
            print(f"  {et:<6}  {len(segs):>6}  {durs[0]:>13.0f}  {sum(durs):>10.0f}")
    print("=" * W + "\n")


# ── Figure 1: Full time series ────────────────────────────────────────────────

def plot_full_timeseries(
    data: SNIRFData, events: EventData, save_path: Path
) -> None:
    """Plot full HbO recording with color-coded event background regions.

    One channel is plotted over the entire ~40-minute recording.
    Each event occurrence is shaded with its event-type colour.
    """
    hbo   = data.hbo[CHANNEL_IDX] * MOL_TO_UMOL   # µmol/L
    times = data.times / 60.0                        # minutes

    fig, ax = plt.subplots(figsize=(16, 4))

    # Shade event windows
    for et, segs in events.by_type().items():
        color = EVENT_COLORS[et]
        alpha = 0.20 if et in ("BS", "R") else 0.35
        first = True
        for onset, duration in segs:
            ax.axvspan(
                onset / 60, (onset + duration) / 60,
                color=color, alpha=alpha, linewidth=0,
                label=EVENT_LABELS[et] if first else None,
            )
            first = False

    # Signal trace
    ax.plot(times, hbo, color="#222222", linewidth=0.55, alpha=0.95, zorder=5)

    # Collect unique legend handles (one per event type)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, mpatches.Patch] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(
        list(seen.values()), list(seen.keys()),
        loc="upper right", fontsize=FS_LEGEND, ncol=2,
        framealpha=0.92, edgecolor="#CCCCCC",
    )

    ch_name = data.src_det_pairs[CHANNEL_IDX]
    ax.set_xlabel("Time (min)", fontsize=FS_LABEL)
    ax.set_ylabel("ΔHbO (µmol/L)", fontsize=FS_LABEL)
    ax.set_title(
        f"Full fNIRS Recording — {data.subject_id}  ·  Channel {ch_name} HbO",
        fontsize=FS_TITLE, fontweight="bold",
    )
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(times[0], times[-1])
    ax.margins(y=0.08)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path.name)


# ── Figure 2: Waveform comparison ─────────────────────────────────────────────

def plot_waveform_comparison(
    data: SNIRFData, events: EventData, save_path: Path
) -> None:
    """Plot mean ± std waveforms for HT / SM / LM across all segments.

    DC offset is removed per segment (subtract segment mean) so that
    different conditions can be overlaid on the same axes.
    Left panel: HbO; right panel: HbR.
    """
    CONDITIONS = [
        ("HT", EVENT_COLORS["HT"], "HT  — no movement (clean)"),
        ("SM", EVENT_COLORS["SM"], "SM — small movement"),
        ("LM", EVENT_COLORS["LM"], "LM — large movement"),
    ]
    fs  = data.sfreq
    n_t = data.hbo.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    panel_cfg = [
        ("HbO", data.hbo[CHANNEL_IDX] * MOL_TO_UMOL),
        ("HbR", data.hbr[CHANNEL_IDX] * MOL_TO_UMOL),
    ]
    ch_name = data.src_det_pairs[CHANNEL_IDX]

    for ax, (mol_label, signal) in zip(axes, panel_cfg):
        for et, color, label in CONDITIONS:
            segs: list[np.ndarray] = []
            for onset, duration in events.get_type(et):
                s = int(round(onset * fs))
                e = min(int(round((onset + duration) * fs)), n_t)
                if e > s:
                    seg = signal[s:e].copy()
                    seg -= seg.mean()   # remove DC offset
                    segs.append(seg)

            if not segs:
                continue

            # Align to shortest length (segments should all be 100 samples)
            min_len = min(len(s) for s in segs)
            mat = np.stack([s[:min_len] for s in segs], axis=0)  # (N, L)
            t   = np.arange(min_len) / fs

            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)

            ax.plot(t, mean, color=color, linewidth=2.0, label=label, zorder=4)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
        ax.set_xlabel("Time within segment (s)", fontsize=FS_LABEL)
        ax.set_ylabel("Δ Concentration (µmol/L)", fontsize=FS_LABEL)
        ax.set_title(f"{mol_label}  ·  Channel {ch_name}", fontsize=FS_TITLE - 1, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND, loc="upper right")
        ax.tick_params(labelsize=FS_TICK)

    n_segs = len(events.get_type("HT"))
    fig.suptitle(
        f"Waveform Comparison by Condition — {data.subject_id}"
        f"  (mean ± std, n={n_segs} segments each, DC removed)",
        fontsize=FS_TITLE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path.name)


# ── Figure 3: Per-channel SNR ─────────────────────────────────────────────────

def plot_channel_snr(
    data: SNIRFData, events: EventData, save_path: Path
) -> None:
    """Per-channel signal quality: within-segment std and SNR.

    Left panel: bar chart of within-segment std (µmol/L) for HT / SM / LM.
    Right panel: channel SNR (dB) defined as
        SNR = 10·log₁₀(σ²_HT / σ²_artifact)
    where σ²_artifact = max(σ²_LM − σ²_HT, 0.01·σ²_HT).
    A positive SNR means the clean-signal variance exceeds artifact variance.
    """
    n_pairs   = data.n_pairs
    ch_labels = data.src_det_pairs
    fs        = data.sfreq

    # ── Compute per-channel within-segment std for each condition ─────────────
    cond_stds: dict[str, np.ndarray] = {et: np.zeros(n_pairs) for et in ["HT", "SM", "LM"]}
    snr_db = np.zeros(n_pairs)

    for ch in range(n_pairs):
        hbo_ch = data.hbo[ch] * MOL_TO_UMOL
        n_t    = len(hbo_ch)

        for et in ["HT", "SM", "LM"]:
            seg_stds: list[float] = []
            for onset, duration in events.get_type(et):
                s = int(round(onset * fs))
                e = min(int(round((onset + duration) * fs)), n_t)
                if e > s:
                    seg_stds.append(float(hbo_ch[s:e].std()))
            if seg_stds:
                cond_stds[et][ch] = float(np.mean(seg_stds))

        # SNR: ratio of HT variance to artifact variance (LM excess over HT)
        ht_var       = max(cond_stds["HT"][ch] ** 2, 1e-12)
        lm_var       = max(cond_stds["LM"][ch] ** 2, 1e-12)
        artifact_var = max(lm_var - ht_var, ht_var * 0.01)   # ≥1% of HT variance
        snr_db[ch]   = 10 * np.log10(ht_var / artifact_var)

    # ── Plotting ──────────────────────────────────────────────────────────────
    x      = np.arange(n_pairs)
    width  = 0.26
    colors = {"HT": EVENT_COLORS["HT"], "SM": EVENT_COLORS["SM"], "LM": EVENT_COLORS["LM"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: within-segment std grouped bar chart
    for i, (et, color) in enumerate(colors.items()):
        ax1.bar(
            x + (i - 1) * width, cond_stds[et],
            width=width, color=color, alpha=0.85,
            label=et, edgecolor="white", linewidth=0.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(ch_labels, rotation=35, ha="right", fontsize=FS_TICK)
    ax1.set_xlabel("Channel (source–detector pair)", fontsize=FS_LABEL)
    ax1.set_ylabel("Within-segment σ  (µmol/L)", fontsize=FS_LABEL)
    ax1.set_title(
        "Signal Variability per Condition",
        fontsize=FS_TITLE - 1, fontweight="bold",
    )
    ax1.legend(title="Condition", fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)
    ax1.tick_params(axis="y", labelsize=FS_TICK)

    # Right: SNR bar chart
    bar_cols = [EVENT_COLORS["HT"] if v >= 0 else EVENT_COLORS["LM"] for v in snr_db]
    bars = ax2.bar(
        x, snr_db, color=bar_cols, alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    ax2.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    for bar, val in zip(bars, snr_db):
        y_off = 0.25 if val >= 0 else -0.85
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + y_off,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(ch_labels, rotation=35, ha="right", fontsize=FS_TICK)
    ax2.set_xlabel("Channel (source–detector pair)", fontsize=FS_LABEL)
    ax2.set_ylabel("SNR  (dB)", fontsize=FS_LABEL)
    ax2.set_title(
        r"Channel SNR  [$10\cdot\log_{10}(\sigma^2_{HT}\,/\,\sigma^2_{artifact})$]",
        fontsize=FS_TITLE - 1, fontweight="bold",
    )
    ax2.tick_params(axis="y", labelsize=FS_TICK)

    n_segs = len(events.get_type("HT"))
    fig.suptitle(
        f"Signal Quality Statistics — {data.subject_id}  (HbO, n={n_segs} segments per condition)",
        fontsize=FS_TITLE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path.name)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # sub-18 has no nirs/ subdir; all others do
    snirf_path = BIDS_DIR / SUBJECT_ID / "nirs" / f"{SUBJECT_ID}_task-tapping_nirs.snirf"
    if not snirf_path.exists():
        # Fallback: search directly in subject dir
        candidates = list((BIDS_DIR / SUBJECT_ID).glob("*.snirf"))
        if not candidates:
            logger.error("No .snirf file found for %s", SUBJECT_ID)
            sys.exit(1)
        snirf_path = candidates[0]

    logger.info("Loading %s …", snirf_path)
    data   = load_snirf(snirf_path, ppf=6.0)
    events = load_events(find_events_file(snirf_path))

    print_statistics(data, events)

    logger.info("Generating Figure 1: full time series …")
    plot_full_timeseries(data, events, FIGURES_DIR / "01_full_timeseries.png")

    logger.info("Generating Figure 2: waveform comparison …")
    plot_waveform_comparison(data, events, FIGURES_DIR / "01_waveform_comparison.png")

    logger.info("Generating Figure 3: channel SNR …")
    plot_channel_snr(data, events, FIGURES_DIR / "01_channel_snr.png")

    print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
