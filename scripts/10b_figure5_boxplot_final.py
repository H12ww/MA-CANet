"""生成 Figure 5 最终版箱线图（9 方法 × 5 指标）。

读取 outputs/comparison_scores_final.csv，输出 outputs/figures/figure5_final.pdf。

用法::

    python scripts/10b_figure5_boxplot_final.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


METHOD_ORDER = [
    "Bandpass", "Wavelet", "Spline", "TDDR", "PCA",
    "DAE-Large", "CNNwP", "LSTM-AE", "MA-CANet",
]

METRIC_COLS = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]

METRIC_LABELS = {
    "delta_snr":  r"$\Delta$SNR (dB)",
    "rmse":       "RMSE",
    "pearson_r":  "Pearson r",
    "ssim":       "SSIM",
    "eta":        r"$\eta$ (%)",
}

# 前 8 个方法用蓝色系，MA-CANet 用金色
_PALETTE = [
    "#5B8DB8", "#7BAD7B", "#D4785A", "#9B6B9B",
    "#C4A35A", "#5B9B9B", "#8B7BAD", "#C47B7B",
    "#D4A017",  # MA-CANet 金色
]


def setup_style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update({
        "font.size":       9,
        "axes.titlesize": 10,
        "axes.labelsize":  9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.linewidth":  0.8,
        "grid.linewidth":  0.4,
        "grid.alpha":      0.5,
    })


def main() -> None:
    scores_path = Path("outputs/comparison_scores_final.csv")
    if not scores_path.exists():
        print(f"[ERROR] 未找到 {scores_path}，请先运行 06b_compare_baselines_final.py")
        return

    scores_df = pd.read_csv(scores_path)

    # 筛选存在于数据中的方法
    present = scores_df["method"].unique().tolist()
    methods = [m for m in METHOD_ORDER if m in present]
    n_methods = len(methods)
    ma_idx = methods.index("MA-CANet") if "MA-CANet" in methods else -1

    setup_style()

    nrow, ncol = 2, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 8), dpi=300)
    axes_flat = axes.flatten()

    for ax_idx, metric in enumerate(METRIC_COLS):
        ax = axes_flat[ax_idx]

        data_per_method = [
            scores_df[scores_df["method"] == m][metric].dropna().values
            for m in methods
        ]

        bp = ax.boxplot(
            data_per_method,
            positions=list(range(n_methods)),
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
            showfliers=True,
        )

        for i, (patch, color) in enumerate(zip(bp["boxes"], _PALETTE[:n_methods])):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)

        # MA-CANet 加粗红色边框
        if ma_idx >= 0:
            bp["boxes"][ma_idx].set_linewidth(2.2)
            bp["boxes"][ma_idx].set_edgecolor("#c00000")
            bp["boxes"][ma_idx].set_alpha(0.9)

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(list(range(n_methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "eta":
            ax.set_yscale("symlog", linthresh=50)
            ax.set_ylim(-10, None)

    # 关闭多余子图
    for ax in axes_flat[len(METRIC_COLS):]:
        ax.set_visible(False)

    fig.suptitle(
        "Figure 5 — Comparison of Motion Artifact Removal Methods (Test Set, n=750)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path("outputs/figures/figure5_final.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Figure 5 Final 已保存：{out_path}")


if __name__ == "__main__":
    main()
