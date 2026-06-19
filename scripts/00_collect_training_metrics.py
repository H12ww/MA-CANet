"""从 TensorBoard 事件文件中提取训练指标，生成 CSV 和论文质量曲线图。

用法::

    python scripts/00_collect_training_metrics.py [--logdir outputs/checkpoints] \\
        [--output-csv outputs/training_history.csv] \\
        [--output-fig outputs/figures/training_curves.pdf] \\
        [--folds 0 1 2 3 4]

功能：
  1. 遍历每折最新的 TensorBoard 日志目录
  2. 提取 train_loss / val_loss / RMSE / Pearson_r / ΔSNR / lr 逐 epoch 值
  3. 保存长格式 CSV：列 fold, epoch, metric, value
  4. 生成 2×2 子图 PDF（train_loss, val_loss, ΔSNR, Pearson_r）
     论文标准：300 DPI、Times New Roman、半透明置信带（若有多折）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# TensorBoard tag → 内部列名
_TAG_MAP: dict[str, str] = {
    "Loss/train":    "train_loss",
    "Loss/val":      "val_loss",
    "LR":            "lr",
    "Val/rmse":      "rmse",
    "Val/pearson_r": "pearson_r",
    "Val/delta_snr": "delta_snr",
}


# ===========================================================================
# 数据提取
# ===========================================================================

def _latest_log_dir(fold_dir: Path) -> Path | None:
    """返回该折下最新的 TensorBoard 日志子目录（按目录名升序取最后一个）。"""
    log_root = fold_dir / "logs"
    if not log_root.exists():
        return None
    subdirs = sorted(
        [d for d in log_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return subdirs[-1] if subdirs else None


def _load_fold(fold_dir: Path, fold_idx: int) -> pd.DataFrame:
    """从一折的最新日志中加载所有 scalar，返回长格式 DataFrame。"""
    log_dir = _latest_log_dir(fold_dir)
    if log_dir is None:
        logger.warning("fold_%d: 未找到日志目录，跳过。", fold_idx)
        return pd.DataFrame()

    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    available_tags = set(ea.Tags().get("scalars", []))
    records: list[dict] = []

    for tag, col in _TAG_MAP.items():
        if tag not in available_tags:
            logger.debug("fold_%d: tag '%s' 不存在，跳过。", fold_idx, tag)
            continue
        for ev in ea.Scalars(tag):
            records.append({
                "fold":   fold_idx,
                "epoch":  int(ev.step),
                "metric": col,
                "value":  float(ev.value),
            })

    n_epochs = len(ea.Scalars("Loss/train")) if "Loss/train" in available_tags else 0
    logger.info(
        "fold_%d: %d epochs  dir=%s", fold_idx, n_epochs, log_dir.name
    )
    return pd.DataFrame(records)


def collect_all(logdir: Path, folds: list[int]) -> pd.DataFrame:
    """汇总所有折的训练指标，返回长格式 DataFrame。"""
    frames: list[pd.DataFrame] = []
    for fold_idx in folds:
        fold_dir = logdir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            logger.warning("目录 %s 不存在，跳过。", fold_dir)
            continue
        df = _load_fold(fold_dir, fold_idx)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("未提取到任何数据，请检查日志目录。")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# 论文图
# ===========================================================================

# 每个子图的配置：(metric列名, y轴标签, 是否翻转（越低越好）, y最小截断)
_SUBPLOT_CFG: list[tuple[str, str, bool, float | None]] = [
    ("train_loss", "Training Loss",       False, 0.0),
    ("val_loss",   "Validation Loss",     False, 0.0),
    ("delta_snr",  r"$\Delta$SNR (dB)",   False, None),
    ("pearson_r",  "Pearson r",           False, 0.0),
]

_FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_FOLD_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def _setup_fonts() -> None:
    """配置 Times New Roman 字体（若不可用则回退 serif）。"""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available:
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
        logger.debug("Times New Roman 不可用，使用 serif 回退。")

    plt.rcParams.update({
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "legend.fontsize":   8,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.2,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.4,
    })


def _draw_subplot(
    ax: plt.Axes,
    wide: pd.DataFrame,
    metric: str,
    ylabel: str,
    folds: list[int],
    y_min: float | None,
) -> None:
    """在单个 Axes 上绘制所有折的曲线。"""
    for i, fold_idx in enumerate(folds):
        sub = wide[(wide["fold"] == fold_idx) & (wide["metric"] == metric)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("epoch")
        ax.plot(
            sub["epoch"],
            sub["value"],
            color=_FOLD_COLORS[i % len(_FOLD_COLORS)],
            linestyle=_FOLD_STYLES[i % len(_FOLD_STYLES)],
            label=f"Fold {fold_idx}",
            linewidth=1.3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.grid(True, which="both", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if y_min is not None:
        cur_ylim = ax.get_ylim()
        ax.set_ylim(bottom=max(y_min, cur_ylim[0]))


def plot_training_curves(
    df_long: pd.DataFrame,
    folds: list[int],
    output_path: Path,
) -> None:
    """生成 2×2 训练曲线 PDF，论文质量。"""
    _setup_fonts()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    axes_flat = axes.flatten()

    for ax, (metric, ylabel, _, y_min) in zip(axes_flat, _SUBPLOT_CFG):
        _draw_subplot(ax, df_long, metric, ylabel, folds, y_min)

    # 统一图例（取第一个有图例的子图）
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=len(folds),
            frameon=True,
            framealpha=0.9,
            edgecolor="#aaaaaa",
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle("MA-CANet Training Curves (5-Fold Cross-Validation)", fontsize=12, y=1.01)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("论文图已保存：%s", output_path)


# ===========================================================================
# 命令行入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="提取 TensorBoard 日志 → CSV + 论文训练曲线图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--logdir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="包含 fold_0 ~ fold_N 子目录的根目录。",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/training_history.csv"),
        help="输出 CSV 路径（长格式）。",
    )
    p.add_argument(
        "--output-fig",
        type=Path,
        default=Path("outputs/figures/training_curves.pdf"),
        help="输出 PDF 路径。",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(5)),
        help="要处理的折编号。",
    )
    return p.parse_args()


def _print_summary(df_long: pd.DataFrame, folds: list[int]) -> None:
    """打印各折末 epoch 的关键指标摘要。"""
    wide = df_long.pivot_table(
        index=["fold", "epoch"],
        columns="metric",
        values="value",
    ).reset_index()

    print("\n" + "=" * 68)
    print("  各折末 epoch 指标摘要")
    print("=" * 68)
    print(f"  {'Fold':>4}  {'Epoch':>5}  {'val_loss':>9}  "
          f"{'RMSE':>7}  {'Pearson_r':>9}  {'ΔSNR(dB)':>9}")
    print("  " + "-" * 64)

    for fold_idx in folds:
        sub = wide[wide["fold"] == fold_idx]
        if sub.empty:
            continue
        last = sub.loc[sub["epoch"].idxmax()]
        print(
            f"  {fold_idx:>4}  {int(last['epoch']):>5}  "
            f"{last.get('val_loss', float('nan')):>9.4f}  "
            f"{last.get('rmse', float('nan')):>7.4f}  "
            f"{last.get('pearson_r', float('nan')):>9.4f}  "
            f"{last.get('delta_snr', float('nan')):>9.2f}"
        )

    # 末 epoch 均值
    rows_last = []
    for fold_idx in folds:
        sub = wide[wide["fold"] == fold_idx]
        if sub.empty:
            continue
        rows_last.append(sub.loc[sub["epoch"].idxmax()])

    if rows_last:
        last_df = pd.DataFrame(rows_last)
        print("  " + "-" * 64)
        for agg_name, func in [("均值", "mean"), ("标准差", "std")]:
            row = last_df[["val_loss", "rmse", "pearson_r", "delta_snr"]].agg(func)
            print(
                f"  {agg_name:>4}  {'':>5}  "
                f"{row.get('val_loss', float('nan')):>9.4f}  "
                f"{row.get('rmse', float('nan')):>7.4f}  "
                f"{row.get('pearson_r', float('nan')):>9.4f}  "
                f"{row.get('delta_snr', float('nan')):>9.2f}"
            )

    print("=" * 68 + "\n")


def main() -> None:
    args = parse_args()

    logger.info("扫描日志目录：%s", args.logdir.resolve())
    df_long = collect_all(args.logdir, args.folds)

    # 保存 CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(args.output_csv, index=False)
    logger.info(
        "CSV 已保存：%s  (%d 行)", args.output_csv, len(df_long)
    )

    # 打印摘要
    _print_summary(df_long, args.folds)

    # 生成图
    plot_training_curves(df_long, args.folds, args.output_fig)

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
