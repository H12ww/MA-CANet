#!/usr/bin/env python3
"""MA-CANet 训练脚本 — 5 折交叉验证。

流程
----
1. 从 data/semi_synthetic/ 加载 train + val 数据合并为全量集
2. KFold(n_splits=5) 按 pair 维度划分（保证被试不跨折）
3. 每折：实例化 MACANet、HybridLoss、Trainer → fit()
4. 保存每折最优权重到 outputs/checkpoints/fold_N/
5. 全部折完成后打印 5 折平均指标 ± 标准差

用法
----
    python scripts/04_train.py
    python scripts/04_train.py --config configs/default.yaml --gpu 0
    python scripts/04_train.py --fold 2          # 只训练第 2 折（调试用）
    python scripts/04_train.py --no-cv           # 仅用官方 train/val 划分训练一次
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import Augmenter
from src.models.macanet import MACANet
from src.training.losses import HybridLoss
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 内联 Dataset（直接包装 numpy 数组，避免依赖 .npy 文件路径）
# ===========================================================================

class _ArrayPairDataset(Dataset):
    """从 numpy 数组创建 (noisy, clean) Dataset，__getitem__ 返回 (1, L) 张量。"""

    def __init__(
        self,
        noisy:      np.ndarray,        # (N, C, L)
        clean:      np.ndarray,        # (N, C, L)
        aug_config: Optional[dict] = None,
        augment:    bool           = False,
    ) -> None:
        assert noisy.shape == clean.shape
        assert noisy.ndim == 3
        self._noisy    = noisy
        self._clean    = clean
        self._n_pairs  = noisy.shape[0]
        self._n_ch     = noisy.shape[1]

        _aug_cfg = dict(aug_config or {})
        _aug_cfg["enabled"] = augment
        self._augmenter = Augmenter(_aug_cfg)

    def __len__(self) -> int:
        return self._n_pairs * self._n_ch

    def __getitem__(self, idx: int):
        pair = idx // self._n_ch
        ch   = idx  % self._n_ch
        noisy = self._noisy[pair, ch : ch + 1, :].copy()
        clean = self._clean[pair, ch : ch + 1, :].copy()
        noisy, clean = self._augmenter(noisy, clean)
        return (
            torch.from_numpy(noisy.astype(np.float32)),
            torch.from_numpy(clean.astype(np.float32)),
        )


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MA-CANet 5 折交叉验证训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="配置文件路径",
    )
    p.add_argument(
        "--gpu", type=int, default=None,
        help="使用的 GPU 编号；None = 自动（有 CUDA 则用 cuda:0）",
    )
    p.add_argument(
        "--fold", type=int, default=None,
        choices=[0, 1, 2, 3, 4],
        help="只训练指定折（0-4）；None = 全部 5 折",
    )
    p.add_argument(
        "--no-cv", action="store_true",
        help="不做 CV，直接用 train/val .npy 划分训练一次",
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs",
        help="输出根目录",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="交叉验证折数",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    p.add_argument(
        "--debug", action="store_true",
        help=(
            "调试模式：epochs=3, 关闭早停, batch_size=16, num_workers=0, "
            "训练只用前 100 个 pairs、验证只用前 20 个 pairs，"
            "强制 fold=0，启用 verbose Pearson r 日志。"
        ),
    )
    return p.parse_args()


# ===========================================================================
# 数据加载
# ===========================================================================

def load_semi_synthetic(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """加载 train + val 的 noisy/clean 数组并拼接。

    Returns:
        (noisy_all, clean_all)，形状均为 (N_total, 16, 512)。
    """
    data_dir = PROJECT_ROOT / cfg["paths"]["semi_synthetic"]

    splits_to_load = ["train", "val"]
    noisy_parts, clean_parts = [], []

    for split in splits_to_load:
        np_path = data_dir / f"{split}_noisy.npy"
        cp_path = data_dir / f"{split}_clean.npy"
        if not np_path.exists():
            raise FileNotFoundError(
                f"找不到 {np_path}\n请先运行 scripts/03_generate_semi_synthetic.py"
            )
        noisy_parts.append(np.load(np_path))
        clean_parts.append(np.load(cp_path))
        logger.info("加载 %s split: %s", split, np.load(np_path).shape)

    noisy_all = np.concatenate(noisy_parts, axis=0)
    clean_all = np.concatenate(clean_parts, axis=0)
    logger.info("合并后总 pairs：%d", noisy_all.shape[0])
    return noisy_all, clean_all


def load_split(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """加载单个 split 的 .npy 文件。"""
    data_dir = PROJECT_ROOT / cfg["paths"]["semi_synthetic"]
    noisy = np.load(data_dir / f"{split}_noisy.npy")
    clean = np.load(data_dir / f"{split}_clean.npy")
    return noisy, clean


# ===========================================================================
# 工具
# ===========================================================================

def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_loaders(
    train_noisy: np.ndarray,
    train_clean: np.ndarray,
    val_noisy:   np.ndarray,
    val_clean:   np.ndarray,
    cfg:         dict,
) -> tuple[DataLoader, DataLoader]:
    tcfg       = cfg.get("training", {})
    aug_cfg    = cfg.get("augmentation", {})
    batch_size = tcfg.get("batch_size", 32)
    n_workers  = tcfg.get("num_workers", 0)   # Windows 下多进程 DataLoader 需 spawn
    pin_memory = tcfg.get("pin_memory", False) and torch.cuda.is_available()

    train_ds = _ArrayPairDataset(train_noisy, train_clean, aug_config=aug_cfg, augment=True)
    val_ds   = _ArrayPairDataset(val_noisy,   val_clean,   aug_config=aug_cfg, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_one_fold(
    fold_idx:    int,
    train_noisy: np.ndarray,
    train_clean: np.ndarray,
    val_noisy:   np.ndarray,
    val_clean:   np.ndarray,
    cfg:         dict,
    output_dir:  Path,
    device:      torch.device,
    verbose:     bool = False,
) -> Dict[str, float]:
    """训练一折，返回该折最优验证指标。"""
    fold_name = f"fold_{fold_idx}"
    fold_dir  = output_dir / "checkpoints" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[%s] train pairs=%d  val pairs=%d",
        fold_name, len(train_noisy), len(val_noisy),
    )

    train_loader, val_loader = make_loaders(
        train_noisy, train_clean, val_noisy, val_clean, cfg
    )

    model     = MACANet.from_config(cfg)
    criterion = HybridLoss.from_config(cfg)

    # 将早停 checkpoint 写到折专属目录
    fold_cfg = {**cfg}
    fold_cfg["_fold_output"] = str(fold_dir)   # Trainer 使用 output_dir 参数

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        config       = cfg,
        output_dir   = fold_dir,
        device       = device,
        verbose      = verbose,
    )

    t0      = time.time()
    history = trainer.fit()
    elapsed = time.time() - t0
    logger.info("[%s] 训练完成，耗时 %.1f s", fold_name, elapsed)

    # 加载最优权重后计算最终验证指标
    try:
        trainer.load_best_checkpoint()
    except FileNotFoundError:
        pass  # 若早停从未触发（epoch 不足）则使用当前权重

    val_loss, val_metrics = trainer.validate_epoch()
    val_metrics["val_loss"] = val_loss
    val_metrics["elapsed"]  = elapsed

    print(
        f"\n[{fold_name}] 最优 val_loss={val_loss:.4f}"
        f"  RMSE={val_metrics.get('rmse', float('nan')):.4f}"
        f"  Pearson_r={val_metrics.get('pearson_r', float('nan')):.4f}"
        f"  ΔSNR={val_metrics.get('delta_snr', float('nan')):.2f} dB"
    )
    return val_metrics


# ===========================================================================
# 汇总打印
# ===========================================================================

def print_cv_summary(fold_results: List[Dict[str, float]]) -> None:
    keys = ["val_loss", "rmse", "pearson_r", "delta_snr"]
    labels = {
        "val_loss":  "val_loss",
        "rmse":      "RMSE",
        "pearson_r": "Pearson_r",
        "delta_snr": "ΔSNR(dB)",
    }
    W = 72
    print("\n" + "=" * W)
    print("  5 折交叉验证汇总")
    print("=" * W)
    print(f"  {'Fold':>4}  {'val_loss':>10}  {'RMSE':>8}  {'Pearson_r':>9}  {'ΔSNR(dB)':>9}")
    print("  " + "-" * (W - 2))

    for i, res in enumerate(fold_results):
        print(
            f"  {i:>4}  {res.get('val_loss', float('nan')):>10.4f}"
            f"  {res.get('rmse', float('nan')):>8.4f}"
            f"  {res.get('pearson_r', float('nan')):>9.4f}"
            f"  {res.get('delta_snr', float('nan')):>9.2f}"
        )

    print("  " + "-" * (W - 2))
    for agg, label in [("均值", np.mean), ("标准差", np.std)]:
        row = {}
        for k in keys:
            vals = [r[k] for r in fold_results if k in r and not np.isnan(r[k])]
            row[k] = label(vals) if vals else float("nan")
        print(
            f"  {agg:>4}  {row.get('val_loss', float('nan')):>10.4f}"
            f"  {row.get('rmse', float('nan')):>8.4f}"
            f"  {row.get('pearson_r', float('nan')):>9.4f}"
            f"  {row.get('delta_snr', float('nan')):>9.2f}"
        )

    print("=" * W)

    # 参数量提示
    model_tmp = MACANet()
    n_params  = model_tmp.count_parameters()
    print(f"\n  模型参数量：{n_params:,}  ({n_params/1e3:.1f} K)")
    print()


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    args = parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 覆盖 seed
    cfg["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Debug 模式：极速验证管线，不关注指标数值 ────────────────────
    if args.debug:
        logger.warning(
            "DEBUG 模式：epochs=3, 早停关闭, "
            "train_pairs=100, val_pairs=20, num_workers=0"
        )
        cfg["training"]["epochs"]                  = 3
        # 将 patience 设为远大于 epochs 的值，等效于关闭早停
        cfg["training"]["early_stopping_patience"] = 10_000
        cfg["training"]["batch_size"]              = 16
        cfg["training"]["num_workers"]             = 0
        cfg["training"]["pin_memory"]              = False
        # 若未指定 fold，debug 时只跑 fold=0
        if args.fold is None:
            args.fold = 0
    verbose = args.debug

    device = resolve_device(args)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 55}")
    print("  MA-CANet 训练 — 5 折交叉验证")
    print(f"{'=' * 55}")
    print(f"  配置文件  : {args.config}")
    print(f"  设备      : {device}")
    print(f"  输出目录  : {output_dir}")
    print(f"  fold      : {args.fold if args.fold is not None else '全部 (0-4)'}")
    print(f"  no-cv     : {args.no_cv}")
    print(f"  seed      : {args.seed}")
    print(f"{'=' * 55}\n")

    # debug 时的样本截断限制（pair 维度）
    _debug_train_limit = 100 if args.debug else None
    _debug_val_limit   =  20 if args.debug else None

    # ── 非 CV 模式：直接用官方 train/val 划分训练一次 ─────────────────
    if args.no_cv:
        logger.info("非 CV 模式：直接使用 train/val split 训练")
        train_noisy, train_clean = load_split(cfg, "train")
        val_noisy,   val_clean   = load_split(cfg, "val")
        if _debug_train_limit:
            train_noisy = train_noisy[:_debug_train_limit]
            train_clean = train_clean[:_debug_train_limit]
        if _debug_val_limit:
            val_noisy = val_noisy[:_debug_val_limit]
            val_clean = val_clean[:_debug_val_limit]
        result = run_one_fold(
            fold_idx    = 0,
            train_noisy = train_noisy,
            train_clean = train_clean,
            val_noisy   = val_noisy,
            val_clean   = val_clean,
            cfg         = cfg,
            output_dir  = output_dir,
            device      = device,
            verbose     = verbose,
        )
        print_cv_summary([result])
        return

    # ── CV 模式 ───────────────────────────────────────────────────────
    from sklearn.model_selection import KFold

    noisy_all, clean_all = load_semi_synthetic(cfg)
    n_pairs = noisy_all.shape[0]
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pair_idx = np.arange(n_pairs)

    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pair_idx)):
        # --fold 指定时跳过其余折
        if args.fold is not None and fold_idx != args.fold:
            continue

        print(f"\n{'─' * 55}")
        print(f"  FOLD {fold_idx} / {args.n_folds - 1}")
        print(f"{'─' * 55}")

        # debug 模式截断：只用指定数量的 pairs
        t_idx = train_idx[:_debug_train_limit] if _debug_train_limit else train_idx
        v_idx = val_idx[:_debug_val_limit]     if _debug_val_limit   else val_idx

        result = run_one_fold(
            fold_idx    = fold_idx,
            train_noisy = noisy_all[t_idx],
            train_clean = clean_all[t_idx],
            val_noisy   = noisy_all[v_idx],
            val_clean   = clean_all[v_idx],
            cfg         = cfg,
            output_dir  = output_dir,
            device      = device,
            verbose     = verbose,
        )
        fold_results.append(result)

    if fold_results:
        print_cv_summary(fold_results)
    else:
        logger.warning("没有训练任何折次，请检查 --fold 参数。")


if __name__ == "__main__":
    main()
