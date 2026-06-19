"""生成论文发表级图表。

Figure 1 — MA-CANet 网络架构示意图（matplotlib 框图）
Figure 4 — 典型去噪效果波形对比（spike / shift / mixed 各一例）

输出目录：outputs/figures/
    Figure1_architecture.pdf
    Figure4_waveform_comparison.pdf

用法::

    python scripts/10_paper_figures.py \\
        [--data-dir  data/semi_synthetic] \\
        [--ckpt      outputs/checkpoints/ablation/A5_best.pth] \\
        [--config    configs/default.yaml] \\
        [--output-dir outputs/figures] \\
        [--seed      42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================
# 全局绘图参数
# ============================================================

def _setup_rc() -> None:
    """设置全局 matplotlib 参数，Times New Roman 字体，论文风格。"""
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in avail else "DejaVu Serif"
    plt.rcParams.update({
        "font.family":       family,
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "lines.linewidth":   1.2,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.4,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,   # TrueType 嵌入，兼容 Adobe
        "ps.fonttype":       42,
    })


# ============================================================
# Figure 1 — MA-CANet 架构示意图
# ============================================================

# 颜色方案
_C = {
    "io":         "#4CAF50",   # 输入/输出：绿色
    "ms":         "#FF9800",   # MS-Conv：橙色
    "enc":        "#2196F3",   # Encoder：蓝色
    "bn":         "#9C27B0",   # Bottleneck：紫色
    "dec":        "#F44336",   # Decoder：红色
    "se":         "#FFC107",   # SE Block 标签：金黄
    "skip":       "#607D8B",   # Skip connection 箭头：灰蓝
    "arrow":      "#333333",   # 普通箭头：深灰
    "text_dark":  "#111111",
    "text_light": "#FFFFFF",
}


def _box(ax, cx, cy, w, h, color, text, fontsize=8, text_color="#FFFFFF",
         radius=0.015, zorder=3, bold=False):
    """在 ax 上绘制带圆角的矩形盒子，中心位于 (cx, cy)。"""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0.8, edgecolor="#333333",
        facecolor=color, zorder=zorder,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            weight=weight, zorder=zorder + 1,
            multialignment="center")


def _arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.0,
           arrowstyle="-|>", dashed=False):
    """绘制带箭头的连线。"""
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )


def _se_badge(ax, cx, cy, r=0.018):
    """在指定位置画一个 SE block 小徽章（圆形）。"""
    circ = plt.Circle((cx, cy), r, color=_C["se"], linewidth=0.6,
                       edgecolor="#AA8800", zorder=6)
    ax.add_patch(circ)
    ax.text(cx, cy, "SE", ha="center", va="center",
            fontsize=6, color="#111111", weight="bold", zorder=7)


def draw_architecture(output_pdf: Path) -> None:
    """绘制 MA-CANet 网络架构示意图（Figure 1）。"""
    _setup_rc()

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 布局参数 ─────────────────────────────────────────────
    cx      = 0.50          # 主流中心 x
    bw      = 0.36          # 主流盒子宽度
    bh      = 0.042         # 盒子高度
    dy      = 0.068         # 垂直间距（盒中心间距）
    ms_bh   = 0.060         # MS-Conv 盒子稍高
    ms_bw   = 0.22          # MS-Conv 子框宽度

    # y 坐标（从上到下）
    y_input  = 0.955
    y_ms     = 0.870
    y_enc    = [0.790, 0.710, 0.630, 0.550]
    y_bn     = 0.460
    y_dec    = [0.370, 0.290, 0.210, 0.130]
    y_output = 0.050

    # ── 输入 ─────────────────────────────────────────────────
    _box(ax, cx, y_input, bw, bh, _C["io"],
         "Input  x(t)\n[B × 1 × 512]", fontsize=8.5, bold=True)

    # ── MS-Conv Stem ─────────────────────────────────────────
    _box(ax, cx, y_ms, bw + 0.06, ms_bh, _C["ms"],
         "Multi-Scale Conv (MS-Conv Stem)", fontsize=8.5, bold=True)

    # MS-Conv 内部 4 个并行子框
    sub_labels = ["k=3", "k=7", "k=15", "k=31"]
    sub_xs = [0.26, 0.38, 0.50, 0.62]
    for sx, sl in zip(sub_xs, sub_labels):
        _box(ax, sx, y_ms - 0.010, ms_bw * 0.6, bh * 0.58,
             "#FFB74D", sl, fontsize=6.5, text_color="#333333")

    ax.text(cx + 0.28, y_ms, "→ Concat → 1×1 Conv",
            ha="left", va="center", fontsize=7, color="#555555")

    # 主流箭头
    _arrow(ax, cx, y_input - bh / 2, cx, y_ms + ms_bh / 2)

    # ── Encoder × 4 ──────────────────────────────────────────
    enc_labels = [
        "Encoder 1\nConv1D→BN→ReLU → MaxPool(2)\n32 ch → 32 ch",
        "Encoder 2\nConv1D→BN→ReLU → MaxPool(2)\n32 ch → 64 ch",
        "Encoder 3\nConv1D→BN→ReLU → MaxPool(2)\n64 ch → 128 ch",
        "Encoder 4\nConv1D→BN→ReLU → MaxPool(2)\n128 ch → 128 ch",
    ]
    enc_right = cx + bw / 2    # 右边缘（skip connection 出发点）

    for i, (ye, lbl) in enumerate(zip(y_enc, enc_labels)):
        prev_y = y_ms if i == 0 else y_enc[i - 1]
        _arrow(ax, cx, prev_y - (ms_bh if i == 0 else bh) / 2,
               cx, ye + bh / 2)
        _box(ax, cx, ye, bw, bh, _C["enc"], lbl, fontsize=7)
        _se_badge(ax, cx + bw / 2 - 0.025, ye)

    # ── Bottleneck ───────────────────────────────────────────
    _arrow(ax, cx, y_enc[-1] - bh / 2, cx, y_bn + bh / 2)
    _box(ax, cx, y_bn, bw, bh, _C["bn"],
         "Bottleneck:  Conv1D → BN → ReLU → Dropout(0.3)\n128 ch",
         fontsize=7.5, bold=True)

    # ── Decoder × 4 ──────────────────────────────────────────
    dec_labels = [
        "Decoder 4\nUpsample(2) → Concat(skip4) → Conv1D→BN→ReLU\n256→128 ch",
        "Decoder 3\nUpsample(2) → Concat(skip3) → Conv1D→BN→ReLU\n256→64 ch",
        "Decoder 2\nUpsample(2) → Concat(skip2) → Conv1D→BN→ReLU\n128→32 ch",
        "Decoder 1\nUpsample(2) → Concat(skip1) → Conv1D→BN→ReLU\n64→32 ch",
    ]
    dec_left = cx - bw / 2    # 左边缘（skip connection 终点）

    for i, (yd, lbl) in enumerate(zip(y_dec, dec_labels)):
        prev_y = y_bn if i == 0 else y_dec[i - 1]
        _arrow(ax, cx, prev_y - bh / 2, cx, yd + bh / 2)
        _box(ax, cx, yd, bw, bh, _C["dec"], lbl, fontsize=7)
        _se_badge(ax, cx - bw / 2 + 0.025, yd)

    # ── 输出 ─────────────────────────────────────────────────
    _arrow(ax, cx, y_dec[-1] - bh / 2, cx, y_output + bh / 2)
    _box(ax, cx, y_output, bw, bh, _C["io"],
         "Output  1×1 Conv → ŷ(t)  [B × 1 × 512]",
         fontsize=8.5, bold=True)

    # ── Skip Connections（虚线，从 Encoder 右侧绕到 Decoder 左侧）─
    skip_x_right = cx + bw / 2 + 0.04    # Encoder 出发 x
    skip_x_left  = cx - bw / 2 - 0.04   # Decoder 终点 x

    skip_pairs = list(zip(y_enc, reversed(y_dec)))  # [(enc1,dec1),...,(enc4,dec4)]
    offsets    = [0.06, 0.10, 0.14, 0.18]           # 每条 skip 向右偏移量

    for idx, ((ye, yd), off) in enumerate(zip(skip_pairs, offsets)):
        xr = skip_x_right + off
        xl = skip_x_left  - off

        # Encoder 右边缘 → 右侧绕行点
        ax.annotate("", xy=(xr, ye),
                    xytext=(cx + bw / 2, ye),
                    arrowprops=dict(arrowstyle="-", color=_C["skip"],
                                    lw=0.8, linestyle=(0, (4, 2))))
        # 竖向连线（右侧）
        ax.plot([xr, xr], [ye, yd], color=_C["skip"],
                lw=0.8, linestyle=(0, (4, 2)), zorder=2)
        # 右侧 → Decoder 左边缘（带箭头）
        ax.annotate("", xy=(cx - bw / 2, yd),
                    xytext=(xr, yd),
                    arrowprops=dict(arrowstyle="-|>", color=_C["skip"],
                                    lw=0.8, linestyle=(0, (4, 2))))

        # skip 标签
        ax.text(xr + 0.005, (ye + yd) / 2,
                f"skip{4 - idx}", fontsize=6, color=_C["skip"],
                ha="left", va="center", rotation=90)

    # ── 图例 ─────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=_C["io"],  edgecolor="#333", label="Input / Output"),
        mpatches.Patch(facecolor=_C["ms"],  edgecolor="#333", label="MS-Conv Stem"),
        mpatches.Patch(facecolor=_C["enc"], edgecolor="#333", label="Encoder Block"),
        mpatches.Patch(facecolor=_C["bn"],  edgecolor="#333", label="Bottleneck"),
        mpatches.Patch(facecolor=_C["dec"], edgecolor="#333", label="Decoder Block"),
        mpatches.Patch(facecolor=_C["se"],  edgecolor="#AA8800", label="SE Attention"),
        mpatches.Patch(facecolor="none", edgecolor=_C["skip"],
                       linestyle="--", label="Skip Connection"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.01, 0.01),
              fontsize=7.5, frameon=True, framealpha=0.92,
              edgecolor="#AAAAAA", ncol=2)

    ax.set_title(
        "Figure 1 — MA-CANet Architecture\n"
        "(Multi-scale Attention-enhanced Convolutional Network for fNIRS Artifact Removal)",
        fontsize=10, pad=10, weight="bold",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Figure 1 已保存：%s", output_pdf)


# ============================================================
# Figure 4 — 典型去噪效果波形对比
# ============================================================

def _snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """计算 SNR (dB)；noise = signal - clean。"""
    ps = np.mean(signal ** 2)
    pn = np.mean(noise ** 2)
    if pn < 1e-12:
        return 0.0
    return 10.0 * np.log10(ps / pn)


def _delta_snr(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray) -> float:
    """计算 ΔSNR = SNR_after − SNR_before (dB)。"""
    snr_before = _snr(clean, noisy - clean)
    snr_after  = _snr(clean, denoised - clean)
    return snr_after - snr_before


def _select_samples(
    noisy: np.ndarray,
    clean: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    n_total: int = 400,
    target_dsnr: float = 13.98,
    top_k: int = 25,
) -> dict[str, int]:
    """
    从前 n_total 个样本中各挑 1 个最典型的 spike / shift / mixed 样本。

    先按形态学（峰度/均值偏移）确定每类 top_k 候选，再在候选集里
    运行模型，选取 ΔSNR 最接近 target_dsnr 的样本——保证代表性。
    """
    from scipy.stats import kurtosis

    n = min(n_total, len(noisy))
    arts   = noisy[:n] - clean[:n]
    kurt   = np.array([kurtosis(a, fisher=True) for a in arts])
    mu_abs = np.abs(arts.mean(axis=1))

    kurt_n = (kurt - kurt.min()) / (kurt.max() - kurt.min() + 1e-8)
    mu_n   = (mu_abs - mu_abs.min()) / (mu_abs.max() - mu_abs.min() + 1e-8)

    spike_score = kurt_n - mu_n
    shift_score = mu_n - 0.3 * kurt_n
    mixed_score = kurt_n * mu_n

    chosen: dict[str, int] = {}
    used: set[int] = set()

    for label, score in [("spike", spike_score),
                          ("shift", shift_score),
                          ("mixed", mixed_score)]:
        candidates = [i for i in np.argsort(-score)[:top_k] if i not in used]
        dsnrs = []
        for idx in candidates:
            den = _run_model(model, noisy[idx], device)
            dsnrs.append(_delta_snr(noisy[idx], den, clean[idx]))
        best_pos = int(np.argmin(np.abs(np.array(dsnrs) - target_dsnr)))
        best = candidates[best_pos]
        chosen[label] = best
        used.add(best)
        logger.info("%s  idx=%d  ΔSNR=%.2f dB  (target %.2f dB)",
                    label, best, dsnrs[best_pos], target_dsnr)

    return chosen


@torch.no_grad()
def _run_model(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """对单条 (1, 512) 信号运行模型，返回 (512,) numpy 数组。"""
    x = torch.from_numpy(noisy[None, None, :].astype(np.float32)).to(device)
    y = model(x).cpu().numpy()
    return y[0, 0]


def draw_waveform_comparison(
    data_dir: Path,
    ckpt_path: Path,
    cfg:       dict,
    output_pdf: Path,
    seed: int = 42,
) -> None:
    """绘制 Figure 4：spike / shift / mixed 三行波形对比。"""
    _setup_rc()

    # ── 加载数据 ──────────────────────────────────────────────
    noisy_all = np.load(data_dir / "test_noisy.npy")   # (N, 16, 512)
    clean_all = np.load(data_dir / "test_clean.npy")

    N, C, L = noisy_all.shape
    # 展平为 (N*C, 512)
    noisy_flat = noisy_all.reshape(N * C, L)
    clean_flat = clean_all.reshape(N * C, L)
    logger.info("测试集展平后：%d 个样本", len(noisy_flat))

    # ── 加载模型 ──────────────────────────────────────────────
    device = torch.device("cpu")
    from src.models.macanet import MACANetAblation
    model = MACANetAblation(
        ablation_id="A5",
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        ms_out_channels=cfg.get("model", {}).get("ms_out_channels", 32),
        encoder_channels=cfg.get("model", {}).get("encoder_channels", [32, 64, 128, 128]),
        se_reduction=cfg.get("model", {}).get("se_reduction", 8),
        dropout=cfg.get("model", {}).get("dropout", 0.3),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    logger.info("加载权重：%s（val_loss=%.5f，epoch=%d）",
                ckpt_path.name,
                state.get("val_loss", float("nan")),
                state.get("epoch", -1))

    # ── 选取代表样本（在形态学候选集里取最接近均值 ΔSNR 的样本）────
    chosen = _select_samples(
        noisy_flat, clean_flat, model, device,
        n_total=min(400, len(noisy_flat)),
        target_dsnr=13.98,
    )

    # ── 推理 ────────────────────────────────────────────────
    fs  = cfg.get("data", {}).get("sampling_rate", 10)
    t   = np.arange(L) / fs    # 时间轴（秒）

    results = {}
    for label, idx in chosen.items():
        noisy_s   = noisy_flat[idx]
        clean_s   = clean_flat[idx]
        denoised_s = _run_model(model, noisy_s, device)
        dsnr      = _delta_snr(noisy_s, denoised_s, clean_s)
        results[label] = {
            "noisy":    noisy_s,
            "clean":    clean_s,
            "denoised": denoised_s,
            "dsnr":     dsnr,
            "idx":      idx,
        }
        logger.info("%s  idx=%d  ΔSNR=%.2f dB", label, idx, dsnr)

    # ── 绘图 ─────────────────────────────────────────────────
    row_titles = {
        "spike": "(a) Spike Artifact",
        "shift": "(b) Baseline Shift Artifact",
        "mixed": "(c) Mixed Artifact",
    }

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), dpi=300,
                             sharex=False, constrained_layout=True)

    for ax, label in zip(axes, ["spike", "shift", "mixed"]):
        r = results[label]

        ax.plot(t, r["noisy"],    color="#AAAAAA", lw=0.9,
                label="Noisy (contaminated)", zorder=1, alpha=0.9)
        ax.plot(t, r["clean"],    color="#D32F2F", lw=1.1,
                linestyle="--", label="Clean (ground truth)", zorder=3)
        ax.plot(t, r["denoised"], color="#1565C0", lw=1.1,
                label="MA-CANet (denoised)", zorder=2)

        # ΔSNR 标注
        ymin, ymax = ax.get_ylim()
        text_y = ymax - 0.05 * (ymax - ymin) if ymax != ymin else 1.0
        ax.text(
            0.98, 0.96,
            f"$\\Delta$SNR = {r['dsnr']:+.2f} dB",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#AAAAAA", alpha=0.85),
        )

        ax.set_title(row_titles[label], fontsize=9.5, loc="left", pad=3)
        ax.set_ylabel("Amplitude (a.u.)", fontsize=8.5)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if label == list(row_titles)[-1]:
            ax.set_xlabel("Time (s)", fontsize=8.5)

    axes[0].legend(
        loc="upper left", fontsize=7.5,
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
        ncol=3,
    )

    fig.suptitle(
        "Figure 4 — Typical Denoising Results: MA-CANet vs. Ground Truth",
        fontsize=10.5, weight="bold",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Figure 4 已保存：%s", output_pdf)


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成论文级图表 Figure 1 & Figure 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   type=Path, default=Path("data/semi_synthetic"))
    p.add_argument("--ckpt",       type=Path,
                   default=Path("outputs/checkpoints/ablation/A5_best.pth"))
    p.add_argument("--config",     type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--figure",     choices=["1", "4", "all"], default="all",
                   help="指定生成哪张图（1=架构图, 4=波形对比, all=全部）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.figure in ("1", "all"):
        draw_architecture(args.output_dir / "Figure1_architecture.pdf")

    if args.figure in ("4", "all"):
        if not args.ckpt.exists():
            logger.error("找不到 checkpoint：%s\n请先完成 A5 训练。", args.ckpt)
            sys.exit(1)
        draw_waveform_comparison(
            data_dir=args.data_dir,
            ckpt_path=args.ckpt,
            cfg=cfg,
            output_pdf=args.output_dir / "Figure4_waveform_comparison.pdf",
            seed=args.seed,
        )

    logger.info("全部图表生成完毕。")


if __name__ == "__main__":
    main()
