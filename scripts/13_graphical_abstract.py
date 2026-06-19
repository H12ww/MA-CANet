"""scripts/13_graphical_abstract.py
为 BSPC 投稿生成 Graphical Abstract。
尺寸：1328 × 531 px（Elsevier 推荐），300 DPI，输出 PDF。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import matplotlib.font_manager as fm

# ── 字体 ────────────────────────────────────────────────────────
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT  = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"
plt.rcParams.update({
    "font.family":  _FONT,
    "font.size":    10,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── 颜色 ────────────────────────────────────────────────────────
C = {
    "noisy":  "#7A7A7A",
    "clean":  "#1F77B4",
    "stem":   "#FF9800",
    "enc":    "#2196F3",
    "bn":     "#9C27B0",
    "dec":    "#F44336",
    "se":     "#FFC107",
    "metric": "#2E7D32",
}

# ── 合成展示数据 ────────────────────────────────────────────────
np.random.seed(42)
T = 512
t = np.linspace(0, 51.2, T)

# 干净 HRF（生理基线 + 慢变血流动力学）
clean = (0.6 * np.sin(2 * np.pi * 0.04 * t) +
         0.3 * np.sin(2 * np.pi * 0.08 * t + 0.5))

# 添加运动伪影：尖峰 + 基线漂移
artifact = np.zeros_like(t)
# 两个尖峰
for spike_t, sign in [(15, +1), (32, -1)]:
    artifact += sign * 2.5 * np.exp(-((t - spike_t) ** 2) / (2 * 0.3 ** 2))
# 一个基线漂移
artifact += 1.6 / (1 + np.exp(-(t - 22) / 1.5)) - 0.8

noisy = clean + artifact + 0.08 * np.random.randn(T)
denoised = clean + 0.05 * np.random.randn(T)  # 模拟 MA-CANet 输出

# ── 画布（宽:高 ≈ 2.5:1） ───────────────────────────────────────
FW, FH = 13.28, 5.31  # 英寸，对应 1328×531 @ 100dpi（300dpi 下保持比例）
fig = plt.figure(figsize=(FW, FH), dpi=300)
fig.patch.set_facecolor("white")

# 三栏：noisy | model | denoised
gs = fig.add_gridspec(
    nrows=2, ncols=3,
    width_ratios=[1.0, 0.85, 1.0],
    height_ratios=[1.0, 0.18],
    left=0.04, right=0.98, top=0.92, bottom=0.06,
    wspace=0.18, hspace=0.18,
)

# ─── 左栏：含噪信号 ────────────────────────────────────────────
ax_l = fig.add_subplot(gs[0, 0])
ax_l.plot(t, noisy, color=C["noisy"], lw=1.0, label="Noisy fNIRS")
ax_l.set_title("Motion-contaminated fNIRS", fontsize=11, pad=8,
               fontweight="bold")
ax_l.set_xlabel("Time (s)", fontsize=9)
ax_l.set_ylabel("ΔHbO (a.u.)", fontsize=9)
ax_l.tick_params(labelsize=8)
ax_l.grid(alpha=0.25, linestyle=":")
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)
# 标注伪影
ax_l.annotate("Spike", xy=(15, 1.6), xytext=(6, 1.4),
              fontsize=8.5, color="#C62828",
              arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8))
ax_l.annotate("Baseline shift", xy=(28, 1.4), xytext=(36, 1.2),
              fontsize=8.5, color="#C62828",
              arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8))

# ─── 中栏：MA-CANet 模型示意 ────────────────────────────────────
ax_m = fig.add_subplot(gs[0, 1])
ax_m.set_xlim(0, 10)
ax_m.set_ylim(0, 10)
ax_m.axis("off")
ax_m.set_title("MA-CANet", fontsize=12, pad=8, fontweight="bold")

# 模型主体框（圆角矩形）
def add_block(ax, x, y, w, h, color, label, fontsize=8.5, tc="white"):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=0.7, edgecolor="#444",
        facecolor=color, zorder=3,
    ))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight="bold", zorder=4)

# MS-Conv stem
add_block(ax_m, 1.6, 6.5, 2.0, 1.4, C["stem"], "MS-Conv\nk=3,7,15,31", 7.5)
# Encoder + SE
add_block(ax_m, 4.0, 6.5, 1.6, 1.4, C["enc"], "Encoder\n+ SE", 8.5)
# Bottleneck
add_block(ax_m, 6.2, 6.5, 1.4, 1.4, C["bn"], "Bottleneck", 8.5)
# Decoder + SE
add_block(ax_m, 8.4, 6.5, 1.6, 1.4, C["dec"], "Decoder\n+ SE", 8.5)

# 箭头连接
for x0, x1 in [(2.6, 3.2), (4.8, 5.5), (6.9, 7.6)]:
    ax_m.annotate("", xy=(x1, 6.5), xytext=(x0, 6.5),
                  arrowprops=dict(arrowstyle="-|>", color="#333",
                                  lw=1.0, mutation_scale=10), zorder=2)

# 模型下方 Hybrid Loss 与参数量
ax_m.text(5.0, 4.3, "Hybrid Loss",
          ha="center", fontsize=9, fontweight="bold", color="#333")
ax_m.text(5.0, 3.5, "MSE + 0.1·Spectral + 0.1·SSIM",
          ha="center", fontsize=8, style="italic", color="#555")
ax_m.text(5.0, 2.4, "320 K params  |  2.86 ms CPU",
          ha="center", fontsize=8.5, color="#1565C0",
          fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                    edgecolor="#1565C0", linewidth=0.7))

# ─── 右栏：去噪后信号 ──────────────────────────────────────────
ax_r = fig.add_subplot(gs[0, 2])
ax_r.plot(t, denoised, color=C["clean"], lw=1.2, label="MA-CANet output")
ax_r.plot(t, clean, color="#C62828", lw=0.8, linestyle="--",
          alpha=0.6, label="Ground truth")
ax_r.set_title("Denoised fNIRS", fontsize=11, pad=8, fontweight="bold")
ax_r.set_xlabel("Time (s)", fontsize=9)
ax_r.set_ylabel("ΔHbO (a.u.)", fontsize=9)
ax_r.tick_params(labelsize=8)
ax_r.grid(alpha=0.25, linestyle=":")
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)
ax_r.legend(loc="upper right", fontsize=7.5, frameon=False)

# ─── 底部一行：核心指标 ─────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, :])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 1)
ax_b.axis("off")

metrics = [
    (1.7, "ΔSNR", "13.98 ± 0.36 dB"),
    (5.0, "Pearson r",  "0.967 ± 0.004"),
    (8.3, "Cross-dataset", "0.846 → 0.911"),
]
for cx, label, val in metrics:
    ax_b.add_patch(FancyBboxPatch(
        (cx - 1.4, 0.20), 2.8, 0.65,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0.8, edgecolor=C["metric"],
        facecolor="#E8F5E9", zorder=2,
    ))
    ax_b.text(cx, 0.66, label, ha="center", va="center",
              fontsize=8.5, color="#1B5E20", fontweight="bold")
    ax_b.text(cx, 0.36, val, ha="center", va="center",
              fontsize=10, color=C["metric"], fontweight="bold")

# 大箭头：从左信号穿过模型到右信号
fig.text(0.345, 0.40, "→", ha="center", va="center",
         fontsize=28, color="#666", fontweight="bold")
fig.text(0.660, 0.40, "→", ha="center", va="center",
         fontsize=28, color="#666", fontweight="bold")

# ─── 保存 ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
for d in [ROOT / "outputs" / "figures", ROOT / "paper"]:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "graphical_abstract.pdf",
                dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(d / "graphical_abstract.png",
                dpi=300, bbox_inches="tight", format="png")
    print(f"Saved → {d / 'graphical_abstract.pdf'} / .png")

plt.close(fig)
print("Graphical abstract generated.")
