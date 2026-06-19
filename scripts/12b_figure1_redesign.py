"""scripts/12b_figure1_redesign.py
MA-CANet Figure 1 重新设计 — 横向 U 型布局。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import matplotlib.font_manager as fm

# ── 字体 & 样式 ───────────────────────────────────────────────────
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT  = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"
plt.rcParams.update({
    "font.family":  _FONT,
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── 颜色方案 ───────────────────────────────────────────────────────
C = {
    "io":   "#4CAF50",
    "stem": "#FF9800",
    "enc":  "#2196F3",
    "bn":   "#9C27B0",
    "dec":  "#F44336",
    "se":   "#FFC107",
    "skip": "#666666",
    "sub":  "#FFB74D",
}


# ── 绘图函数 ──────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, color, lines, fs=8.0, tc="white", bold=False):
    """绘制圆角矩形，中心 (cx, cy)。"""
    r = min(w, h) * 0.09
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=0.8, edgecolor="#444444",
        facecolor=color, zorder=3,
    ))
    if lines:
        text = "\n".join(lines) if isinstance(lines, list) else lines
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal",
                multialignment="center", zorder=4, linespacing=1.35)


def se_dot(ax, cx, cy):
    """SE 注意力小徽章（圆形，右上角）。"""
    ax.add_patch(plt.Circle((cx, cy), 0.135,
                             facecolor=C["se"], edgecolor="#AA8800",
                             linewidth=0.6, zorder=6))
    ax.text(cx, cy, "SE", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#333333", zorder=7)


def harrow(ax, x0, y, x1, lw=1.1, color="#333333"):
    """水平箭头（x0 → x1）。"""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9),
                zorder=5)


def skip_conn(ax, x, y_top, y_bot, label):
    """垂直虚线 skip 连接 + 标签。"""
    ax.plot([x, x], [y_top, y_bot], color=C["skip"],
            lw=0.85, linestyle=(0, (5, 3)), zorder=2)
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_bot + 0.20),
                arrowprops=dict(arrowstyle="-|>", color=C["skip"],
                                lw=0.85, mutation_scale=7), zorder=3)
    ax.text(x - 0.11, (y_top + y_bot) / 2, label,
            ha="right", va="center", fontsize=6.5,
            color=C["skip"], rotation=90)


# ── 坐标常量 ──────────────────────────────────────────────────────
FW, FH   = 14.5, 6.5
Y_T      = 4.85     # 顶排中心 y
Y_B      = 1.90     # 底排中心 y
BH       = 0.78     # 标准框高度
BH_MS    = 1.00     # MS-Conv 框高度
BH_BN    = 1.00     # Bottleneck 框高度
BW       = 1.15     # 标准框宽度
BW_IO    = 1.10     # Input/Output 宽度
BW_MS    = 2.10     # MS-Conv 宽度
BW_BN    = 1.50     # Bottleneck 宽度

X_IN     = 0.70
X_MS     = 2.30
X_ENC    = [4.00, 5.50, 7.00, 8.50]   # Enc1-4（从左到右）
X_BN     = 10.20
X_DEC    = [8.50, 7.00, 5.50, 4.00]   # Dec4-3-2-1（从右到左）
X_OUT    = 2.30

# ── 创建画布 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────
# 顶排：Input → MS-Conv → Enc1-4 → Bottleneck
# ─────────────────────────────────────────────────────────────────

# Input
rbox(ax, X_IN, Y_T, BW_IO, BH, C["io"],
     ["Input", "[B×1×512]"], fs=8.0, bold=True)

# MS-Conv 主框
rbox(ax, X_MS, Y_T, BW_MS, BH_MS, C["stem"], None)
ax.text(X_MS, Y_T + 0.31, "MS-Conv Stem",
        ha="center", va="center", fontsize=8.0,
        fontweight="bold", color="white", zorder=5)
# 4 个 k 子框（水平排列，位于主框中部偏下）
sub_xs  = [X_MS - 0.72, X_MS - 0.24, X_MS + 0.24, X_MS + 0.72]
sub_lbl = ["k = 3", "k = 7", "k = 15", "k = 31"]
for sx, sl in zip(sub_xs, sub_lbl):
    ax.add_patch(FancyBboxPatch(
        (sx - 0.20, Y_T - 0.26), 0.40, 0.29,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=0.7, edgecolor="white",
        facecolor=C["sub"], zorder=5,
    ))
    ax.text(sx, Y_T - 0.115, sl, ha="center", va="center",
            fontsize=5.5, color="#333333", fontweight="bold", zorder=6)
ax.text(X_MS, Y_T - 0.42, "Concat  →  1×1 Conv",
        ha="center", va="center", fontsize=6.0,
        color="white", style="italic", zorder=5)

# Enc1-4
enc_lbl = [
    ["Enc 1", "32 → 32"],
    ["Enc 2", "32 → 64"],
    ["Enc 3", "64 → 128"],
    ["Enc 4", "128 → 128"],
]
for xi, lb in zip(X_ENC, enc_lbl):
    rbox(ax, xi, Y_T, BW, BH, C["enc"], lb, fs=8.0)
    se_dot(ax, xi + BW / 2 - 0.165, Y_T + BH / 2 - 0.165)

# Bottleneck
rbox(ax, X_BN, Y_T, BW_BN, BH_BN, C["bn"],
     ["Bottleneck", "128 ch", "Dropout(0.3)"], fs=7.5, bold=True)

# ─────────────────────────────────────────────────────────────────
# 底排：Dec4-1 → Output
# ─────────────────────────────────────────────────────────────────
dec_lbl = [
    ["Dec 4", "256 → 128"],
    ["Dec 3", "256 → 64"],
    ["Dec 2", "128 → 32"],
    ["Dec 1", "64 → 32"],
]
for xi, lb in zip(X_DEC, dec_lbl):
    rbox(ax, xi, Y_B, BW, BH, C["dec"], lb, fs=8.0)
    se_dot(ax, xi + BW / 2 - 0.165, Y_B + BH / 2 - 0.165)

# Output
rbox(ax, X_OUT, Y_B, BW_IO, BH, C["io"],
     ["Output", "1×1 Conv", "[B×1×512]"], fs=7.5, bold=True)

# ─────────────────────────────────────────────────────────────────
# 顶排水平箭头
# ─────────────────────────────────────────────────────────────────
harrow(ax, X_IN  + BW_IO / 2,   Y_T, X_MS  - BW_MS / 2)
harrow(ax, X_MS  + BW_MS / 2,   Y_T, X_ENC[0] - BW / 2)
for i in range(3):
    harrow(ax, X_ENC[i] + BW / 2, Y_T, X_ENC[i + 1] - BW / 2)
harrow(ax, X_ENC[3] + BW / 2,   Y_T, X_BN  - BW_BN / 2)

# ─────────────────────────────────────────────────────────────────
# 底排水平箭头（向左）
# ─────────────────────────────────────────────────────────────────
for i in range(3):
    harrow(ax, X_DEC[i] - BW / 2, Y_B, X_DEC[i + 1] + BW / 2)
harrow(ax, X_DEC[3] - BW / 2, Y_B, X_OUT + BW_IO / 2)

# ─────────────────────────────────────────────────────────────────
# Bottleneck → Dec4（L 形连接）
# ─────────────────────────────────────────────────────────────────
lx      = X_BN
y_start = Y_T - BH_BN / 2         # BN 底边
y_end   = Y_B                      # 底排 y
x_end   = X_DEC[0] + BW / 2 + 0.05  # Dec4 右边缘

# 垂直段（从 BN 底部向下）
ax.plot([lx, lx], [y_start, y_end], color="#333333", lw=1.1, zorder=4)
# 水平段（向左至 Dec4）
harrow(ax, lx, y_end, x_end, color="#333333")

# ─────────────────────────────────────────────────────────────────
# Skip 连接（垂直虚线）
# ─────────────────────────────────────────────────────────────────
skip_lbl = ["skip 4", "skip 3", "skip 2", "skip 1"]
for xi, sl in zip(X_ENC, skip_lbl):
    skip_conn(ax, xi,
              Y_T - BH / 2 - 0.06,   # 编码器框下方
              Y_B + BH / 2 + 0.06,   # 解码器框上方
              sl)

# ─────────────────────────────────────────────────────────────────
# MaxPool / Upsample 小标注
# ─────────────────────────────────────────────────────────────────
mid_y = (Y_T - BH / 2 + Y_B + BH / 2) / 2   # 中间空白区域 y
for xi in X_ENC:
    ax.text(xi - 0.48, mid_y + 0.25, "↓pool", ha="center", va="center",
            fontsize=5.5, color="#888888", style="italic")
    ax.text(xi - 0.48, mid_y - 0.25, "↑upsample", ha="center", va="center",
            fontsize=5.5, color="#888888", style="italic")

# ─────────────────────────────────────────────────────────────────
# 路径标签
# ─────────────────────────────────────────────────────────────────
ax.text(5.95, FH - 0.22, "Encoder Path  →",
        ha="center", va="top", fontsize=9, style="italic", color="#555555")
ax.text(5.95, 0.22, "←  Decoder Path",
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555")

# ─────────────────────────────────────────────────────────────────
# 图例（右下角）
# ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C["io"],   edgecolor="#444", label="Input / Output"),
    mpatches.Patch(facecolor=C["stem"], edgecolor="#444", label="MS-Conv Stem"),
    mpatches.Patch(facecolor=C["enc"],  edgecolor="#444", label="Encoder Block"),
    mpatches.Patch(facecolor=C["bn"],   edgecolor="#444", label="Bottleneck"),
    mpatches.Patch(facecolor=C["dec"],  edgecolor="#444", label="Decoder Block"),
    mpatches.Patch(facecolor=C["se"],   edgecolor="#AA8800", label="SE Attention"),
    mpatches.Patch(facecolor="none",    edgecolor=C["skip"],
                   linestyle="--", label="Skip Connection"),
]
ax.legend(handles=legend_items, loc="lower right",
          bbox_to_anchor=(1.00, 0.00),
          fontsize=7.5, frameon=True, framealpha=0.93,
          edgecolor="#BBBBBB", ncol=2, columnspacing=0.8,
          handlelength=1.2)

# ─────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
for d in [ROOT / "outputs" / "figures", ROOT / "paper" / "figures"]:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "Figure1_architecture.pdf",
                dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved → {d / 'Figure1_architecture.pdf'}")

plt.close(fig)
print("Done.")
