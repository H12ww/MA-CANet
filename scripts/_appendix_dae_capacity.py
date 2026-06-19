"""附录图：DAE 容量饱和曲线。

展示纯 DAE 架构在不同参数量下的 ΔSNR，以及 MA-CANet 作为对比参考线，
说明性能提升来自架构设计而非参数容量。

输出：outputs/figures/figure_appendix_dae_capacity.pdf

用法::

    python scripts/_appendix_dae_capacity.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 四档 DAE 实测数据（测试集 ΔSNR 均值）
DAE_POINTS = [
    (22_209,  5.19, "DAE-Base\n(22K)"),
    (51_329,  5.39, "SmallDAE\n(51K)"),
    (114_625, 5.27, "SmallDAE\n(115K)"),
    (322_081, 6.06, "DAE-Large\n(322K)"),
]

# MA-CANet 参考点
MACANET = (320_801, 13.60)


def setup_style() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update({
        "font.size":      10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth":  0.9,
        "grid.linewidth":  0.4,
        "grid.alpha":      0.5,
    })


def main() -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    params = [p for p, _, _ in DAE_POINTS]
    dsnrs  = [d for _, d, _ in DAE_POINTS]
    labels = [l for _, _, l in DAE_POINTS]

    # DAE 折线 + 散点
    ax.plot(params, dsnrs, color="#3A7EBF", linewidth=1.5, zorder=2,
            linestyle="-", marker=None)
    ax.scatter(params, dsnrs, color="#3A7EBF", s=60, zorder=3, label="Pure DAE variants")

    # 标注 DAE 数据点
    for p, d, lbl in DAE_POINTS:
        ax.annotate(
            lbl,
            xy=(p, d),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5,
            color="#3A7EBF",
        )

    # MA-CANet 参考点（红色五角星）
    ax.scatter([MACANET[0]], [MACANET[1]],
               marker="*", color="#C00000", s=220, zorder=5,
               label="MA-CANet (ours)")
    ax.annotate(
        f"MA-CANet\n({MACANET[1]:.2f} dB)",
        xy=MACANET,
        xytext=(30, -18),
        textcoords="offset points",
        ha="left", va="top",
        fontsize=8.5,
        color="#C00000",
        arrowprops=dict(arrowstyle="-", color="#C00000", lw=0.8),
    )

    # "+7.54 dB from architecture" 双向箭头
    x_mid   = 320_000
    y_low   = DAE_POINTS[-1][1]   # EnhancedDAE dSNR
    y_high  = MACANET[1]
    y_mid   = (y_low + y_high) / 2
    ax.annotate(
        "", xy=(x_mid, y_high), xytext=(x_mid, y_low),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0),
    )
    ax.text(x_mid + 15_000, y_mid, "+7.54 dB\n(architecture)",
            ha="left", va="center", fontsize=8, color="#555555")

    # "DAE capacity saturation" 标注
    ax.text(
        70_000, 5.55,
        "DAE capacity\nsaturation",
        ha="center", va="bottom", fontsize=8, color="#3A7EBF",
        style="italic",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("ΔSNR (dB)")
    ax.set_title("DAE Capacity Analysis vs MA-CANet")
    ax.set_ylim(3, 16)
    ax.grid(axis="both", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.85)

    # x 轴刻度标注
    ax.set_xticks([22_209, 51_329, 114_625, 320_000])
    ax.set_xticklabels(["22K", "51K", "115K", "320K"])

    fig.tight_layout()
    out_path = Path("outputs/figures/figure_appendix_dae_capacity.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"DAE 容量曲线图已保存：{out_path}")


if __name__ == "__main__":
    main()
