"""训练 EnhancedDAE（增强版去噪自编码器基线）。

训练配置与 scripts/05_train_dae.py 完全一致，唯一变量是模型架构：
DAENet（22K 参数）→ EnhancedDAE（322K 参数）。

目的：在参数量对齐 MA-CANet 的条件下，验证 MA-CANet 各组件的必要性。

用法::

    python scripts/05b_train_enhanced_dae.py
    python scripts/05b_train_enhanced_dae.py --quick-test   # 只跑 2 epoch × 50 batch
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 随机种子
# ===========================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 数据加载（与 05_train_dae.py 完全一致）
# ===========================================================================

def load_split(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """加载一个数据分割并将 16 通道展开为独立样本。

    Args:
        data_dir: 半合成数据目录（含 {split}_noisy.npy, {split}_clean.npy）。
        split:    'train' / 'val' / 'test'。

    Returns:
        Tuple (noisy, clean)，每个形状 (N*16, 1, 512)，float32 Tensor。
    """
    noisy_path = data_dir / f"{split}_noisy.npy"
    clean_path = data_dir / f"{split}_clean.npy"

    if not noisy_path.exists() or not clean_path.exists():
        raise FileNotFoundError(
            f"未找到 {split} 数据：{noisy_path} 或 {clean_path}\n"
            "请先运行 scripts/03_generate_semi_synthetic.py"
        )

    noisy = np.load(noisy_path)   # (N, 16, 512)
    clean = np.load(clean_path)   # (N, 16, 512)

    N, C, L = noisy.shape
    noisy_flat = noisy.reshape(N * C, L)[:, np.newaxis, :]
    clean_flat = clean.reshape(N * C, L)[:, np.newaxis, :]

    noisy_t = torch.from_numpy(noisy_flat.astype(np.float32))
    clean_t = torch.from_numpy(clean_flat.astype(np.float32))

    logger.info("%s: %d 对 x 16 通道 = %d 样本  (L=%d)", split, N, N * C, L)
    return noisy_t, clean_t


# ===========================================================================
# 训练主循环
# ===========================================================================

def train(
    model:       nn.Module,
    loader_tr:   DataLoader,
    loader_val:  DataLoader,
    device:      torch.device,
    epochs:      int,
    patience:    int,
    output_dir:  Path,
    log_dir:     Path,
    quick_test:  bool = False,
    max_batches: int  = 50,
) -> dict:
    """训练 EnhancedDAE，返回训练历史字典。

    Args:
        model:       EnhancedDAE 实例。
        loader_tr:   训练 DataLoader。
        loader_val:  验证 DataLoader。
        device:      训练设备。
        epochs:      最大训练轮数。
        patience:    早停耐心值。
        output_dir:  checkpoint 保存目录。
        log_dir:     TensorBoard 日志目录。
        quick_test:  True 时只跑 2 epoch 且每 epoch 限制 max_batches 个 batch。
        max_batches: quick_test 时每 epoch 最多使用的 batch 数。

    Returns:
        history dict 含 'train_loss', 'val_loss' 列表。
    """
    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard 日志目录：%s", log_dir)
    except Exception:
        writer = None
        logger.warning("TensorBoard 不可用，跳过日志记录")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_cnt  = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_enhanced_dae.pth"

    max_epochs = 2 if quick_test else epochs

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── 训练 ───────────────────────────────────────────────────────────
        model.train()
        tr_loss  = 0.0
        n_tr     = 0
        for batch_idx, (noisy_b, clean_b) in enumerate(loader_tr):
            if quick_test and batch_idx >= max_batches:
                break
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            pred    = model(noisy_b)
            loss    = criterion(pred, clean_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(noisy_b)
            n_tr    += len(noisy_b)
        tr_loss /= n_tr

        # ── 验证 ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch_idx, (noisy_b, clean_b) in enumerate(loader_val):
                if quick_test and batch_idx >= max_batches:
                    break
                noisy_b = noisy_b.to(device)
                clean_b = clean_b.to(device)
                pred    = model(noisy_b)
                loss    = criterion(pred, clean_b)
                val_loss += loss.item() * len(noisy_b)
                n_val    += len(noisy_b)
        val_loss /= n_val

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        elapsed  = time.time() - t0
        is_best  = val_loss < best_val_loss
        print(
            f"Epoch {epoch:3d} | train_loss={tr_loss:.6f} | "
            f"val_loss={val_loss:.6f} | LR={current_lr:.2e} | "
            f"time={elapsed:.1f}s" + ("  [best]" if is_best else "")
        )

        if writer is not None:
            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss, epoch)
            writer.add_scalar("LR",         current_lr, epoch)

        # ── 早停 + 保存 ────────────────────────────────────────────────────
        if is_best:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         val_loss,
                },
                ckpt_path,
            )
        else:
            patience_cnt += 1
            if not quick_test and patience_cnt >= patience:
                logger.info(
                    "Early stopping: %d epochs 无提升 (best val=%.6f)",
                    patience, best_val_loss,
                )
                break

    if writer is not None:
        writer.close()

    logger.info("最佳验证损失：%.6f  权重已保存：%s", best_val_loss, ckpt_path)
    return history


# ===========================================================================
# 命令行入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练 EnhancedDAE 增强版基线（参数量对齐 MA-CANet）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    type=Path, default=Path("data/semi_synthetic"),
                   help="半合成数据目录。")
    p.add_argument("--output-dir",  type=Path, default=Path("outputs/checkpoints/enhanced_dae"),
                   help="checkpoint 保存目录。")
    p.add_argument("--log-dir",     type=Path, default=Path("outputs/logs/enhanced_dae"),
                   help="TensorBoard 日志目录。")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--quick-test",  action="store_true",
                   help="快速验证：只跑 2 epoch x 50 batch，不保存最终权重。")
    return p.parse_args()


def _print_summary(history: dict, ckpt_path: Path) -> None:
    tr   = history["train_loss"]
    val  = history["val_loss"]
    best_e = int(np.argmin(val)) + 1
    print("\n" + "=" * 56)
    print("  EnhancedDAE 训练完成")
    print("=" * 56)
    print(f"  总 epoch：       {len(tr)}")
    print(f"  最佳 epoch：     {best_e}  (val_loss={val[best_e-1]:.6f})")
    print(f"  末 epoch train： {tr[-1]:.6f}")
    print(f"  末 epoch val：   {val[-1]:.6f}")
    print(f"  权重路径：       {ckpt_path}")
    print("=" * 56 + "\n")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    logger.info("随机种子：%d", args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("训练设备：%s", device)

    # 加载模型
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.baselines import EnhancedDAE

    model = EnhancedDAE().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("EnhancedDAE 参数量：%d", n_params)

    # 加载数据
    noisy_tr,  clean_tr  = load_split(args.data_dir, "train")
    noisy_val, clean_val = load_split(args.data_dir, "val")

    loader_tr = DataLoader(
        TensorDataset(noisy_tr, clean_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    loader_val = DataLoader(
        TensorDataset(noisy_val, clean_val),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(
        "数据集规模：train=%d  val=%d  batch=%d",
        len(loader_tr.dataset), len(loader_val.dataset), args.batch_size,
    )

    if args.quick_test:
        logger.info("=== Quick-test 模式：2 epoch x 50 batch ===")

    # 训练
    history = train(
        model       = model,
        loader_tr   = loader_tr,
        loader_val  = loader_val,
        device      = device,
        epochs      = args.epochs,
        patience    = args.patience,
        output_dir  = args.output_dir,
        log_dir     = args.log_dir,
        quick_test  = args.quick_test,
        max_batches = 50,
    )

    if not args.quick_test:
        _print_summary(history, args.output_dir / "best_enhanced_dae.pth")
    else:
        logger.info("Quick-test 完成，loss 曲线：%s", history)


if __name__ == "__main__":
    main()
