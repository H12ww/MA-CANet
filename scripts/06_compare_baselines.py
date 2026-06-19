"""基线方法对比评估脚本。

在 data/semi_synthetic/test_*.npy 上运行 7 种方法，计算
ΔSNR / RMSE / Pearson_r / SSIM / η，输出：
  - outputs/Table_I.csv         方法对比表（含 Wilcoxon p-value）
  - outputs/figures/Figure_5.pdf 箱线图对比图（论文质量）
  - outputs/comparison_scores.csv 每样本原始分数（供后续分析）

用法::

    python scripts/06_compare_baselines.py [--config configs/default.yaml]
        [--test-noisy data/semi_synthetic/test_noisy.npy]
        [--test-clean data/semi_synthetic/test_clean.npy]
        [--ckpt-dir outputs/checkpoints]
        [--folds 0 1 2 3 4]
        [--output-dir outputs]
        [--device auto]
        [--batch-size 64]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 防止 Windows 多 OpenMP 库冲突

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 常量
# ===========================================================================

METHOD_ORDER = [
    "Bandpass", "Wavelet", "Spline", "TDDR",
    "PCA", "DAE", "MA-CANet",
]

METRIC_COLS = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]

METRIC_LABELS = {
    "delta_snr":  r"$\Delta$SNR (dB)",
    "rmse":       "RMSE",
    "pearson_r":  "Pearson r",
    "ssim":       "SSIM",
    "eta":        r"$\eta$ (%)",
}

# 指标方向：True = 越高越好，False = 越低越好
METRIC_HIGHER_BETTER = {
    "delta_snr": True,
    "rmse":      False,
    "pearson_r": True,
    "ssim":      True,
    "eta":       False,
}


# ===========================================================================
# MA-CANet 集成推理
# ===========================================================================

def _load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    """加载单个 MA-CANet 权重。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.macanet import MACANet
    model = MACANet.from_config(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    return model


def load_macanet_ensemble(
    ckpt_dir: Path,
    folds: list[int],
    cfg: dict,
    device: torch.device,
) -> list[torch.nn.Module]:
    """加载所有折的最优 checkpoint，返回模型列表。"""
    import glob
    models = []
    for fold in folds:
        pattern = str(ckpt_dir / f"fold_{fold}" / "checkpoints" / "best_*.pth")
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.warning("fold_%d: 未找到 checkpoint，跳过。", fold)
            continue
        m = _load_model(Path(paths[-1]), cfg, device)
        models.append(m)
        logger.info("fold_%d: 已加载 %s", fold, Path(paths[-1]).name)
    if not models:
        logger.error("没有可用的 MA-CANet checkpoint，请先训练。")
        sys.exit(1)
    return models


@torch.no_grad()
def run_macanet(
    models: list[torch.nn.Module],
    noisy: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """MA-CANet 集成推理（多折平均）。

    Args:
        models:     已加载的模型列表。
        noisy:      含噪数据，形状 (N, C, L)。
        device:     推理设备。
        batch_size: 每批通道数。

    Returns:
        去噪结果，形状 (N, C, L)。
    """
    N, C, L = noisy.shape
    # 展平为 (N*C, 1, L) 逐通道推理
    flat = noisy.reshape(N * C, 1, L).astype(np.float32)
    preds = np.zeros_like(flat)

    for start in range(0, len(flat), batch_size):
        x = torch.from_numpy(flat[start:start + batch_size]).to(device)
        batch_pred = sum(m(x) for m in models) / len(models)
        preds[start:start + batch_size] = batch_pred.cpu().numpy()

    return preds.reshape(N, C, L)


# ===========================================================================
# 各基线方法的批量推理封装
# ===========================================================================

def _run_baseline_batch(
    method_fn,
    noisy: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """对 (N, C, L) 数据逐样本调用方法函数，收集结果。

    Args:
        method_fn: 接受 (C, L) 并返回 (C, L) 的可调用对象。
        noisy:     (N, C, L) 数组。
        **kwargs:  传递给 method_fn 的额外参数。

    Returns:
        (N, C, L) 去噪结果。
    """
    out = np.empty_like(noisy)
    for i in range(len(noisy)):
        out[i] = method_fn(noisy[i], **kwargs)
    return out


def run_all_methods(
    test_noisy: np.ndarray,
    test_clean: np.ndarray,
    models: list[torch.nn.Module],
    cfg: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """运行全部 7 种方法，返回 {method_name: denoised (N,C,L)} 字典。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.baselines import (
        bandpass_filter, wavelet_denoise, spline_correction,
        tddr_correction, pca_denoise, dae_denoise, DAENet,
    )

    fs = float(cfg.get("data", {}).get("sampling_rate", 10.0))
    N, C, L = test_noisy.shape

    results: dict[str, np.ndarray] = {}

    # ── 1. Bandpass ────────────────────────────────────────────────────────
    t0 = time.time()
    results["Bandpass"] = _run_baseline_batch(bandpass_filter, test_noisy, fs=fs)
    logger.info("Bandpass     完成  %.1f s", time.time() - t0)

    # ── 2. Wavelet ─────────────────────────────────────────────────────────
    t0 = time.time()
    results["Wavelet"] = _run_baseline_batch(
        wavelet_denoise, test_noisy, wavelet="db4", level=4
    )
    logger.info("Wavelet      完成  %.1f s", time.time() - t0)

    # ── 3. Spline ──────────────────────────────────────────────────────────
    t0 = time.time()
    results["Spline"] = _run_baseline_batch(spline_correction, test_noisy, fs=fs)
    logger.info("Spline       完成  %.1f s", time.time() - t0)

    # ── 4. TDDR ────────────────────────────────────────────────────────────
    t0 = time.time()
    results["TDDR"] = _run_baseline_batch(tddr_correction, test_noisy, fs=fs)
    logger.info("TDDR         完成  %.1f s", time.time() - t0)

    # ── 5. PCA ─────────────────────────────────────────────────────────────
    t0 = time.time()
    results["PCA"] = _run_baseline_batch(pca_denoise, test_noisy, n_artifact_components=1)
    logger.info("PCA          完成  %.1f s", time.time() - t0)

    # ── 6. DAE ─────────────────────────────────────────────────────────────
    t0 = time.time()
    dae_model = DAENet(input_length=L)
    # 尝试加载 DAE checkpoint（fold_0 目录下同名文件，如不存在则用随机权重）
    dae_ckpt = Path(cfg.get("paths", {}).get("checkpoints", "outputs/checkpoints")) \
               / "dae" / "best_dae.pth"
    if dae_ckpt.exists():
        state = torch.load(dae_ckpt, map_location="cpu", weights_only=False)
        dae_model.load_state_dict(state.get("model_state_dict", state))
        logger.info("DAE 已加载预训练权重：%s", dae_ckpt)
    else:
        logger.warning(
            "DAE checkpoint 不存在（%s），使用随机权重 — "
            "建议先单独训练 DAE 以获得有效对比结果。", dae_ckpt,
        )
    dae_model.to(device).eval()

    flat = test_noisy.reshape(N * C, 1, L).astype(np.float32)
    dae_out = np.empty_like(flat)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            x = torch.from_numpy(flat[start:start + batch_size]).to(device)
            dae_out[start:start + batch_size] = dae_model(x).cpu().numpy()
    results["DAE"] = dae_out.reshape(N, C, L)
    logger.info("DAE          完成  %.1f s", time.time() - t0)

    # ── 7. MA-CANet ────────────────────────────────────────────────────────
    t0 = time.time()
    results["MA-CANet"] = run_macanet(models, test_noisy, device, batch_size)
    logger.info("MA-CANet     完成  %.1f s", time.time() - t0)

    return results


# ===========================================================================
# 逐样本指标计算
# ===========================================================================

def compute_per_sample_metrics(
    denoised_dict: dict[str, np.ndarray],
    test_noisy:    np.ndarray,
    test_clean:    np.ndarray,
) -> pd.DataFrame:
    """对每种方法计算每个测试样本的 5 项指标。

    Returns:
        长格式 DataFrame，列：method, sample_idx, delta_snr, rmse,
        pearson_r, ssim, eta。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation.metrics import compute_all_metrics

    records: list[dict] = []
    N = test_noisy.shape[0]

    for method, denoised in denoised_dict.items():
        logger.info("计算指标：%s …", method)
        for i in range(N):
            noisy_i    = test_noisy[i]     # (C, L)
            clean_i    = test_clean[i]
            denoised_i = denoised[i]
            m = compute_all_metrics(noisy_i, denoised_i, clean_i)
            records.append({
                "method":     method,
                "sample_idx": i,
                "delta_snr":  m["delta_snr"],
                "rmse":       m["rmse"],
                "pearson_r":  m["pearson_r"],
                "ssim":       m["ssim"],
                "eta":        m["eta"],
            })

    return pd.DataFrame(records)


# ===========================================================================
# 统计检验 & 汇总表
# ===========================================================================

def statistical_tests(scores_df: pd.DataFrame) -> pd.DataFrame:
    """对每种 baseline 方法与 MA-CANet 做配对 Wilcoxon 检验。

    Returns:
        DataFrame：method, metric, statistic, pvalue。
    """
    from scipy.stats import wilcoxon

    macanet_scores = scores_df[scores_df["method"] == "MA-CANet"]
    records = []

    for method in METHOD_ORDER:
        if method == "MA-CANet":
            continue
        method_scores = scores_df[scores_df["method"] == method]
        for metric in METRIC_COLS:
            a = macanet_scores[metric].values
            b = method_scores[metric].values
            # Wilcoxon 双侧检验（配对差）
            try:
                stat, pval = wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                stat, pval = float("nan"), float("nan")
            records.append({
                "method":    method,
                "metric":    metric,
                "statistic": stat,
                "pvalue":    pval,
            })

    return pd.DataFrame(records)


def make_table_i(
    scores_df:  pd.DataFrame,
    stats_df:   pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    """生成 Table I：方法 × 指标，格式 mean±std，附 p-value。

    最佳值加 * 标注，p<0.05 在 p-value 列标注 †。
    """
    rows = []
    for method in METHOD_ORDER:
        sub = scores_df[scores_df["method"] == method]
        row: dict = {"Method": method}
        for metric in METRIC_COLS:
            vals = sub[metric].dropna().values
            mean_ = vals.mean() if len(vals) else float("nan")
            std_  = vals.std()  if len(vals) else float("nan")
            row[metric] = mean_
            row[f"{metric}_std"] = std_
            row[f"{metric}_fmt"] = f"{mean_:.4f}±{std_:.4f}"
        # Wilcoxon p-values（相对 MA-CANet）
        if method != "MA-CANet":
            for metric in METRIC_COLS:
                sel = stats_df[
                    (stats_df["method"] == method) & (stats_df["metric"] == metric)
                ]
                pval = sel["pvalue"].values[0] if len(sel) else float("nan")
                row[f"{metric}_pval"] = pval
        rows.append(row)

    table = pd.DataFrame(rows)

    # 标注每列最佳值
    for metric in METRIC_COLS:
        vals = table[metric].values
        if METRIC_HIGHER_BETTER[metric]:
            best_idx = np.nanargmax(vals)
        else:
            best_idx = np.nanargmin(vals)
        table.at[best_idx, f"{metric}_fmt"] = (
            table.at[best_idx, f"{metric}_fmt"] + " *"
        )

    # 选出可读列输出
    fmt_cols = ["Method"] + [f"{m}_fmt" for m in METRIC_COLS]
    pval_cols = [f"{m}_pval" for m in METRIC_COLS]
    out_cols = fmt_cols + [c for c in pval_cols if c in table.columns]
    table[out_cols].to_csv(output_csv, index=False)
    logger.info("Table I 已保存：%s", output_csv)

    # 控制台打印
    print("\n" + "=" * 90)
    print("  TABLE I — 方法对比（test set，mean±std，* = 最佳值）")
    print("=" * 90)
    header = (
        f"  {'Method':<12}"
        + "".join(f"  {METRIC_LABELS[m]:>18}" for m in METRIC_COLS)
    )
    print(header)
    print("  " + "-" * 86)
    for _, r in table.iterrows():
        line = f"  {r['Method']:<12}"
        for metric in METRIC_COLS:
            val = r.get(f"{metric}_fmt", "")
            line += f"  {str(val):>18}"
        print(line)
    print("=" * 90 + "\n")

    return table


# ===========================================================================
# Figure 5 — 箱线图
# ===========================================================================

_PALETTE = [
    "#4878d0", "#ee854a", "#6acc65", "#d65f5f",
    "#956cb4", "#8c613c", "#dc7ec0",
]

def _setup_paper_style() -> None:
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update({
        "font.size":       9,
        "axes.titlesize": 10,
        "axes.labelsize":  9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth":  0.8,
        "lines.linewidth": 1.0,
        "grid.linewidth":  0.4,
        "grid.alpha":      0.5,
    })


def make_figure_5(
    scores_df:  pd.DataFrame,
    stats_df:   pd.DataFrame,
    output_pdf: Path,
) -> None:
    """生成 Figure 5：5 个子图 × 7 种方法的箱线图对比。

    显著性标注（MA-CANet vs baseline）：
      *** p < 0.001  ** p < 0.01  * p < 0.05
    """
    _setup_paper_style()

    nrow, ncol = 2, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 7.5), dpi=300)
    axes_flat = axes.flatten()

    method_positions = {m: i for i, m in enumerate(METHOD_ORDER)}
    xticks = list(range(len(METHOD_ORDER)))
    xlabels = METHOD_ORDER

    for ax_idx, metric in enumerate(METRIC_COLS):
        ax = axes_flat[ax_idx]
        data_per_method = [
            scores_df[scores_df["method"] == m][metric].dropna().values
            for m in METHOD_ORDER
        ]

        bp = ax.boxplot(
            data_per_method,
            positions=xticks,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.5),
            showfliers=True,
        )
        for patch, color in zip(bp["boxes"], _PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # MA-CANet 箱体加粗边框
        ma_idx = method_positions["MA-CANet"]
        bp["boxes"][ma_idx].set_linewidth(2.0)
        bp["boxes"][ma_idx].set_edgecolor("#c00000")

        # 显著性标注
        if metric in [s for _, s in [(r["method"], r["metric"]) for _, r in stats_df.iterrows()]]:
            y_max = max(
                np.percentile(d, 95) for d in data_per_method if len(d)
            )
            y_range = y_max - min(
                np.percentile(d, 5) for d in data_per_method if len(d)
            )
            for _, row in stats_df[stats_df["metric"] == metric].iterrows():
                pval = row["pvalue"]
                if np.isnan(pval) or pval >= 0.05:
                    continue
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
                baseline_pos = method_positions[row["method"]]
                y_ann = y_max + 0.04 * y_range
                ax.annotate(
                    "", xy=(ma_idx, y_ann), xytext=(baseline_pos, y_ann),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
                )
                ax.text(
                    (ma_idx + baseline_pos) / 2, y_ann + 0.01 * y_range,
                    sig, ha="center", va="bottom", fontsize=7, color="#888888",
                )

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=30, ha="right")
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
        "Figure 5 — Comparison of Motion Artifact Removal Methods (Test Set)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Figure 5 已保存：%s", output_pdf)


# ===========================================================================
# 命令行解析 & main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基线方法对比评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--test-noisy", type=Path, default=Path("data/semi_synthetic/test_noisy.npy"))
    p.add_argument("--test-clean", type=Path, default=Path("data/semi_synthetic/test_clean.npy"))
    p.add_argument("--ckpt-dir",   type=Path, default=Path("outputs/checkpoints"))
    p.add_argument("--folds",      type=int,  nargs="+", default=list(range(5)))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--device",     type=str,  default="auto",
                   help="auto / cpu / cuda")
    p.add_argument("--batch-size", type=int,  default=64)
    p.add_argument("--skip-methods", type=str, nargs="*", default=[],
                   help="跳过指定方法（调试用），如 --skip-methods DAE Spline")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("推理设备：%s", device)

    # 加载测试数据
    logger.info("加载测试数据 …")
    test_noisy = np.load(args.test_noisy)   # (N, C, L)
    test_clean  = np.load(args.test_clean)
    N, C, L = test_noisy.shape
    logger.info("  test_noisy=%s  test_clean=%s", test_noisy.shape, test_clean.shape)

    # 加载 MA-CANet 集成
    logger.info("加载 MA-CANet（%d 折集成）…", len(args.folds))
    models = load_macanet_ensemble(args.ckpt_dir, args.folds, cfg, device)

    # 运行所有方法
    logger.info("─" * 50)
    logger.info("运行 7 种方法 …")
    denoised_dict = run_all_methods(
        test_noisy, test_clean, models, cfg, device, args.batch_size
    )

    # 跳过指定方法
    for skip in args.skip_methods:
        if skip in denoised_dict:
            del denoised_dict[skip]
            logger.info("已跳过方法：%s", skip)

    # 计算每样本指标
    logger.info("─" * 50)
    logger.info("计算 %d × %d 样本指标 …", len(denoised_dict), N)
    scores_df = compute_per_sample_metrics(denoised_dict, test_noisy, test_clean)

    # 保存原始分数
    raw_csv = args.output_dir / "comparison_scores.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(raw_csv, index=False)
    logger.info("原始分数已保存：%s  (%d 行)", raw_csv, len(scores_df))

    # 统计检验
    logger.info("─" * 50)
    logger.info("Wilcoxon 配对检验 …")
    stats_df = statistical_tests(scores_df)
    stats_csv = args.output_dir / "wilcoxon_tests.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wilcoxon 结果已保存：%s", stats_csv)

    # Table I
    table_csv = args.output_dir / "Table_I.csv"
    make_table_i(scores_df, stats_df, table_csv)

    # Figure 5
    fig_pdf = args.output_dir / "figures" / "Figure_5.pdf"
    make_figure_5(scores_df, stats_df, fig_pdf)

    logger.info("=" * 50)
    logger.info("评估完成。")


if __name__ == "__main__":
    main()
