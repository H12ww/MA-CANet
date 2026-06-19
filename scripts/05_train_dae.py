"""训练 DAENet（去噪自编码器基线）。

参考 Gao et al. 2022 训练配置：
  - Adam, lr=1e-3, MSE loss
  - batch_size=64, epochs=100, early_stopping patience=15

DAE 以单通道 (B, 1, 512) 格式处理 fNIRS 信号，与 MA-CANet 一致。
训练数据从半合成 (noisy, clean) 对中将 16 通道展开为独立样本。

输出：
  outputs/checkpoints/dae/best_dae.pth   — 最佳验证损失权重

用法::

    python scripts/05_train_dae.py \\
        [--data-dir data/semi_synthetic] \\
        [--output-dir outputs/checkpoints/dae] \\
        [--epochs 100] \\
        [--batch-size 64] \\
        [--lr 1e-3] \\
        [--patience 15] \\
        [--device auto]
"""

from __future__ import annotations

import argparse
import logging
import os
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
# 数据加载
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
    # (N, 16, 512) → (N*16, 512) → (N*16, 1, 512)
    noisy_flat = noisy.reshape(N * C, L)[:, np.newaxis, :]
    clean_flat = clean.reshape(N * C, L)[:, np.newaxis, :]

    noisy_t = torch.from_numpy(noisy_flat.astype(np.float32))
    clean_t = torch.from_numpy(clean_flat.astype(np.float32))

    logger.info(
        "%s: %d 对 × 16 通道 = %d 样本  (L=%d)",
        split, N, N * C, L,
    )
    return noisy_t, clean_t


# ===========================================================================
# 训练主循环
# ===========================================================================

def train(
    model:      nn.Module,
    loader_tr:  DataLoader,
    loader_val: DataLoader,
    device:     torch.device,
    epochs:     int,
    patience:   int,
    output_dir: Path,
) -> dict:
    """训练 DAENet，返回训练历史字典。

    Args:
        model:      DAENet 实例。
        loader_tr:  训练 DataLoader。
        loader_val: 验证 DataLoader。
        device:     训练设备。
        epochs:     最大训练轮数。
        patience:   早停耐心值。
        output_dir: checkpoint 保存目录。

    Returns:
        history dict 含 'train_loss', 'val_loss' 列表。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss   = float("inf")
    patience_cnt    = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_dae.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── 训练 ───────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for noisy_b, clean_b in loader_tr:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            pred    = model(noisy_b)
            loss    = criterion(pred, clean_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(noisy_b)
        tr_loss /= len(loader_tr.dataset)

        # ── 验证 ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_b, clean_b in loader_val:
                noisy_b = noisy_b.to(device)
                clean_b = clean_b.to(device)
                pred    = model(noisy_b)
                loss    = criterion(pred, clean_b)
                val_loss += loss.item() * len(noisy_b)
        val_loss /= len(loader_val.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  train=%.6f  val=%.6f  %.1fs%s",
            epoch, epochs, tr_loss, val_loss, elapsed,
            "  ★" if val_loss < best_val_loss else "",
        )

        # ── 早停 + 保存 ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
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
            if patience_cnt >= patience:
                logger.info(
                    "Early stopping: %d epochs without improvement (best val=%.6f)",
                    patience, best_val_loss,
                )
                break

    logger.info("最佳验证损失：%.6f  权重已保存：%s", best_val_loss, ckpt_path)
    return history


# ===========================================================================
# 命令行入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练 DAENet 去噪自编码器基线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    type=Path, default=Path("data/semi_synthetic"),
                   help="半合成数据目录。")
    p.add_argument("--output-dir",  type=Path, default=Path("outputs/checkpoints/dae"),
                   help="checkpoint 保存目录。")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--num-workers", type=int,   default=0,
                   help="DataLoader worker 数（Windows 建议 0）。")
    return p.parse_args()


def _print_summary(history: dict) -> None:
    tr  = history["train_loss"]
    val = history["val_loss"]
    best_e = int(np.argmin(val)) + 1
    print("\n" + "=" * 52)
    print("  DAENet 训练完成")
    print("=" * 52)
    print(f"  总 epoch：{len(tr)}")
    print(f"  最佳 epoch：{best_e}（val_loss={val[best_e-1]:.6f}）")
    print(f"  末 epoch train_loss：{tr[-1]:.6f}")
    print(f"  末 epoch val_loss：  {val[-1]:.6f}")
    print("=" * 52 + "\n")


def main() -> None:
    args = parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("训练设备：%s", device)

    # 加载模型
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.baselines import DAENet

    model = DAENet(input_length=512).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("DAENet 参数量：%d", n_params)

    # 加载数据
    noisy_tr,  clean_tr  = load_split(args.data_dir, "train")
    noisy_val, clean_val = load_split(args.data_dir, "val")

    loader_tr = DataLoader(
        TensorDataset(noisy_tr, clean_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_val = DataLoader(
        TensorDataset(noisy_val, clean_val),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        "数据集规模：train=%d  val=%d  batch=%d",
        len(loader_tr.dataset), len(loader_val.dataset), args.batch_size,
    )

    # 训练
    history = train(
        model, loader_tr, loader_val,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    _print_summary(history)


if __name__ == "__main__":
    main()
