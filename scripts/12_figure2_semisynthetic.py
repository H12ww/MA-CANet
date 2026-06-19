"""Figure 2: Semi-synthetic data construction pipeline diagram.

Style consistent with Figure 1 (Times New Roman, 300 DPI, PDF).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
PAPER_DIR  = Path(__file__).resolve().parents[1] / "paper" / "figures"

# ── style ──────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
available = {f.name for f in fm.fontManager.ttflist}
font = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
plt.rcParams.update({
    "font.family":      font,
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.linewidth":   0.7,
    "lines.linewidth":  1.2,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.5,
})

FS = 10.0
T  = 512
np.random.seed(42)
t  = np.arange(T) / FS     # time axis (seconds)

# ── generate representative signals ──────────────────────────────────────
# Clean HT background (quasi-sinusoidal HbO-like)
freq1, freq2 = 0.05, 0.03
clean = (0.6 * np.sin(2 * np.pi * freq1 * t)
         + 0.3 * np.sin(2 * np.pi * freq2 * t + 0.8)
         + 0.05 * np.random.randn(T))

# Spike artifact
spike = np.zeros(T)
spike_idx = 120
sigma_s = 0.12 * FS
spike = 2.5 * np.exp(-0.5 * ((np.arange(T) - spike_idx) / sigma_s) ** 2)

# Baseline shift
shift = np.zeros(T)
k = 8.0 / (2.0 * FS)
shift_centre = 300
shift = 1.8 / (1 + np.exp(-k * (np.arange(T) - shift_centre)))
shift -= shift[0]

# Combined noisy
noisy_spike = clean + spike
noisy_shift = clean + shift
noisy_both  = clean + spike * 0.8 + shift * 0.7
noisy_mild  = clean + 0.15 * spike

# ── figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7.2), dpi=300)

# --- Row 0: title boxes ---
# --- Row 1: signal panels (clean | artifact types) ---
# --- Row 2: composition row ---
# --- Row 3: split info ---

# Use gridspec for fine control
outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[3.5, 1.6, 1.2],
                          hspace=0.42)

# ── Top row: 5 signal panels ────────────────────────────────────────────
top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.28)

panels = [
    ("Clean (HT)",          clean,       "#2166ac", None),
    ("Spike artifact\n(30%)", noisy_spike, "#d73027", spike),
    ("Baseline shift\n(30%)", noisy_shift, "#f46d43", shift),
    ("Combined\n(30%)",      noisy_both,  "#984ea3", spike * 0.8 + shift * 0.7),
    ("Mild artifact\n(10%)", noisy_mild,  "#4dac26", 0.15 * spike),
]

ax_list = []
for col, (title, sig, color, art) in enumerate(panels):
    ax = fig.add_subplot(top[col])
    ax_list.append(ax)

    if art is not None:
        ax.fill_between(t, clean - 0.05, clean + 0.05, alpha=0.18,
                        color="#2166ac", label="Clean")
        ax.plot(t, sig, color=color, lw=1.1, label="Noisy")
        ax.plot(t, clean, color="#2166ac", lw=0.8, alpha=0.6, ls="--")
    else:
        ax.plot(t, sig, color=color, lw=1.2)

    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xlim(0, T / FS)
    ax.set_xlabel("Time (s)", fontsize=7)
    if col == 0:
        ax.set_ylabel(r"$\Delta$[HbO] (a.u.)", fontsize=7)
    else:
        ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)

    # Highlight artifact region for spike
    if "Spike" in title:
        ax.axvspan(spike_idx / FS - 2, spike_idx / FS + 2, alpha=0.12,
                   color="#d73027", label="Artifact")
    if "Baseline" in title or "Combined" in title:
        ax.axvspan(shift_centre / FS - 5, T / FS, alpha=0.10,
                   color="#f46d43")

# arrow from clean → noisy panels
ax_clean = ax_list[0]
for col in range(1, 5):
    ax_target = ax_list[col]
    fig.add_artist(
        FancyArrowPatch(
            posA=ax_clean.get_position().get_points()[1],
            posB=ax_target.get_position().get_points()[0],
            arrowstyle="->", mutation_scale=10,
            color="#555555", lw=0.8,
            transform=fig.transFigure,
        )
    )

# ── Middle row: composition & SNR info ──────────────────────────────────
mid = fig.add_subplot(outer[1])
mid.set_axis_off()

# Box: SNR range
snr_text = "Target SNR: $\\mathcal{U}(-10, 10)$ dB"
mid.text(0.5, 0.72, "Artifact Scaling to Target SNR",
         ha="center", va="center", fontsize=10, fontweight="bold",
         transform=mid.transAxes)
mid.text(0.5, 0.44, snr_text,
         ha="center", va="center", fontsize=9, transform=mid.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4",
                   edgecolor="#b8860b", lw=1.0))
mid.text(0.5, 0.14,
         r"noisy $=$ clean $+$ artifact $\times$ scale($\hat{\sigma}^2_{\rm clean}$, "
         r"$\hat{\sigma}^2_{\rm art}$, SNR$_{\rm target}$)",
         ha="center", va="center", fontsize=8.5, transform=mid.transAxes,
         color="#333333")

# ── Bottom row: subject split ────────────────────────────────────────────
bot = fig.add_subplot(outer[2])
bot.set_axis_off()

split_info = [
    ("Training set\nsub-01 – sub-14\n3,500 pairs (70%)",  "#1b7837", 0.18),
    ("Validation set\nsub-15 – sub-17\n750 pairs (15%)",  "#5aae61", 0.50),
    ("Test set\nsub-18 – sub-20\n750 pairs (15%)",        "#a6dba0", 0.82),
]

for text, color, xpos in split_info:
    fancy = FancyBboxPatch(
        (xpos - 0.16, 0.08), 0.32, 0.78,
        boxstyle="round,pad=0.03",
        facecolor=color, edgecolor="white", lw=1.5,
        transform=bot.transAxes, clip_on=False, alpha=0.85,
    )
    bot.add_patch(fancy)
    bot.text(xpos, 0.47, text, ha="center", va="center",
             fontsize=8.5, color="white", fontweight="bold",
             transform=bot.transAxes, linespacing=1.5)

bot.text(0.5, -0.06,
         "Total: 5,000 pairs  |  Split by subject (no data leakage)",
         ha="center", va="center", fontsize=8, color="#444444",
         transform=bot.transAxes)

# ── Main title ───────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 2 — Semi-Synthetic Training Data Construction Pipeline",
    fontsize=11, y=0.99,
)

# ── Save ─────────────────────────────────────────────────────────────────
for out_dir in [OUTPUT_DIR, PAPER_DIR]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "Figure2_semisynthetic.pdf",
                dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved: {out_dir / 'Figure2_semisynthetic.pdf'}")

plt.close(fig)
print("Done.")
