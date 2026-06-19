"""消融实验：按 CLAUDE.md 定义的 A1–A5 五种配置训练并评估。

| ID | MS-Conv | SE Attn | 损失函数    | 说明                      |
|----|---------|---------|-------------|---------------------------|
| A1 | ✗       | ✗       | MSE         | 基础 U-Net                |
| A2 | ✓       | ✗       | MSE         | + Multi-Scale Conv        |
| A3 | ✓       | ✓       | MSE         | + SE Attention            |
| A4 | ✓       | ✓       | Hybrid      | + Hybrid Loss (freq+SSIM) |
| A5 | ✓       | ✓       | Hybrid      | 完整 MA-CANet（从头训练）  |

A1–A5 均使用完全相同的训练条件（相同数据/超参/早停策略），
确保消融对比的公平性。

输出：
  outputs/Table_II_ablation.csv     — 含五行的消融对比表
  outputs/figures/Figure_ablation.pdf — 折线图（5 指标趋势）

用法::

    python scripts/08_ablation_study.py \\
        [--data-dir data/semi_synthetic] \\
        [--ckpt-dir outputs/checkpoints] \\
        [--output-dir outputs] \\
        [--epochs 200] \\
        [--patience 20] \\
        [--batch-size 32] \\
        [--device auto]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 消融配置元数据
ABLATION_CONFIGS = [
    ("A1", "Base U-Net (no MS-Conv, no SE)",    False, False, "mse"),
    ("A2", "+ MS-Conv",                          True,  False, "mse"),
    ("A3", "+ SE Attention",                     True,  True,  "mse"),
    ("A4", "+ Hybrid Loss",                      True,  True,  "hybrid"),
    ("A5", "Full MA-CANet (scratch train)",        True,  True,  "hybrid"),
]


# ===========================================================================
# 数据加载
# ===========================================================================

def _load_tensor(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """加载 (noisy, clean) 对，展平为 (N*16, 1, 512)。"""
    n = np.load(data_dir / f"{split}_noisy.npy")   # (N, 16, 512)
    c = np.load(data_dir / f"{split}_clean.npy")
    N, C, L = n.shape
    nt = torch.from_numpy(n.reshape(N * C, 1, L).astype(np.float32))
    ct = torch.from_numpy(c.reshape(N * C, 1, L).astype(np.float32))
    logger.info("%s: %d 对 × 16 ch = %d 样本", split, N, N * C)
    return nt, ct


def build_loaders(
    data_dir: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """构建 train/val DataLoader 和 test 张量（用于直接评估）。"""
    nt, ct = _load_tensor(data_dir, "train")
    nv, cv = _load_tensor(data_dir, "val")
    n_test, c_test = _load_tensor(data_dir, "test")

    loader_tr = DataLoader(TensorDataset(nt, ct), batch_size=batch_size,
                           shuffle=True, num_workers=0)
    loader_val = DataLoader(TensorDataset(nv, cv), batch_size=batch_size * 2,
                            shuffle=False, num_workers=0)
    return loader_tr, loader_val, n_test, c_test


# ===========================================================================
# 模型 & 损失构建
# ===========================================================================

def build_model(ablation_id: str, cfg: dict) -> nn.Module:
    """根据消融 ID 实例化对应模型。

    A1–A5 均使用 MACANetAblation，通过 ablation_id 控制启用的模块，
    保证训练条件完全一致。
    """
    from src.models.macanet import MACANetAblation

    return MACANetAblation(
        ablation_id=ablation_id,
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        ms_out_channels=cfg.get("model", {}).get("ms_out_channels", 32),
        encoder_channels=cfg.get("model", {}).get("encoder_channels", [32, 64, 128, 128]),
        se_reduction=cfg.get("model", {}).get("se_reduction", 8),
        dropout=cfg.get("model", {}).get("dropout", 0.3),
    )


def build_loss(loss_type: str, cfg: dict) -> nn.Module:
    """构建损失函数；hybrid 返回 HybridLoss，其余返回 MSELoss。"""
    if loss_type == "hybrid":
        from src.training.losses import HybridLoss
        return HybridLoss.from_config(cfg)
    return nn.MSELoss()


def _compute_loss(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """统一调用接口：HybridLoss 返回 (total, dict)，MSELoss 返回 tensor。"""
    result = loss_fn(pred, target)
    return result[0] if isinstance(result, tuple) else result


# ===========================================================================
# 训练循环
# ===========================================================================

def train_one_config(
    ablation_id: str,
    model: nn.Module,
    loss_fn: nn.Module,
    loader_tr: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    save_path: Path,
) -> dict:
    """训练单个消融配置，返回 history dict。"""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val = float("inf")
    wait      = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── 训练 ─────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = _compute_loss(loss_fn, model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(loader_tr.dataset)
        scheduler.step()

        # ── 验证 ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += _compute_loss(loss_fn, model(xb), yb).item() * len(xb)
        val_loss /= len(loader_val.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        star = "  ★" if val_loss < best_val else ""
        logger.info(
            "%s  Epoch %3d/%d  train=%.5f  val=%.5f  %.1fs%s",
            ablation_id, epoch, epochs, tr_loss, val_loss,
            time.time() - t0, star,
        )

        # ── 早停 & 保存 ───────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "epoch": epoch}, save_path)
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "%s  早停：%d epoch 无改善（best val=%.5f）",
                    ablation_id, patience, best_val,
                )
                break

    return history


# ===========================================================================
# 评估
# ===========================================================================

@torch.no_grad()
def evaluate_model(
    model:    nn.Module,
    n_test:   torch.Tensor,
    c_test:   torch.Tensor,
    device:   torch.device,
    batch:    int = 256,
) -> dict[str, float]:
    """在测试集上计算 ΔSNR / RMSE / Pearson_r / SSIM / η。

    Args:
        model:  已加载权重的模型（已 .eval()）。
        n_test: (N*16, 1, 512) 含噪测试张量。
        c_test: (N*16, 1, 512) 干净测试张量。
        device: 推理设备。
        batch:  推理批大小。

    Returns:
        dict 含各指标均值。
    """
    from src.evaluation.metrics import compute_all_metrics

    model.eval()
    preds = []
    for i in range(0, len(n_test), batch):
        xb = n_test[i:i + batch].to(device)
        preds.append(model(xb).cpu())
    pred_all = torch.cat(preds, dim=0).squeeze(1).numpy()   # (N*16, 512)

    noisy_np = n_test.squeeze(1).numpy()
    clean_np = c_test.squeeze(1).numpy()

    metrics = compute_all_metrics(noisy_np, pred_all, clean_np)
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ===========================================================================
# 图表
# ===========================================================================

def _setup_rc() -> None:
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in avail else "serif"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.2, "axes.linewidth": 0.8,
        "grid.linewidth": 0.4, "grid.alpha": 0.4,
    })


def plot_ablation_figure(df: pd.DataFrame, output_pdf: Path) -> None:
    """Figure: A1→A5 各指标折线趋势图（3 行 × 2 列，共 5 个指标）。"""
    _setup_rc()

    metrics = [
        ("delta_snr",  r"$\Delta$SNR (dB)",    True),
        ("rmse",       "RMSE",                  False),
        ("pearson_r",  "Pearson r",              True),
        ("ssim",       "SSIM",                   True),
        ("eta",        r"$\eta$ (%)",            False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), dpi=300)
    axes_flat = axes.flatten()

    ids     = df["ablation_id"].tolist()
    x_ticks = range(len(ids))
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, (col, label, higher_better) in zip(axes_flat, metrics):
        vals = df[col].tolist()
        for xi, (yi, c) in enumerate(zip(vals, colors)):
            ax.scatter(xi, yi, color=c, s=60, zorder=3)
        ax.plot(x_ticks, vals, color="#555555", linewidth=1.0, zorder=2)

        # 标注数值
        for xi, yi in enumerate(vals):
            ax.annotate(
                f"{yi:.3f}", (xi, yi),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(ids)
        ax.set_ylabel(label)
        ax.set_title(f"{'↑ better' if higher_better else '↓ better'}")
        ax.grid(True, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 关掉多余子图
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    # 统一图例
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8,
                   label=f"{aid} — {desc[:28]}")
        for (aid, desc, *_), c in zip(ABLATION_CONFIGS, colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=1,
        bbox_to_anchor=(0.78, 0.10),
        fontsize=7, frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        "Figure 6 — Ablation Study: Incremental Contribution of Each Component",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("消融图已保存：%s", output_pdf)


# ===========================================================================
# 命令行入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="消融实验：A1–A5 配置训练与评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   type=Path, default=Path("data/semi_synthetic"))
    p.add_argument("--config",     type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt-dir",   type=Path, default=Path("outputs/checkpoints"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--epochs",     type=int,  default=200)
    p.add_argument("--patience",   type=int,  default=20)
    p.add_argument("--batch-size", type=int,  default=32)
    p.add_argument("--device",     type=str,  default="auto")
    p.add_argument("--skip-train", action="store_true",
                   help="跳过 A1–A4 训练，直接从已有 checkpoint 加载（调试用）。")
    return p.parse_args()


def _print_table(df: pd.DataFrame) -> None:
    """在控制台打印 Table II。"""
    cols = ["ablation_id", "description", "n_params",
            "delta_snr", "rmse", "pearson_r", "ssim", "eta"]
    w = [6, 38, 10, 12, 10, 11, 10, 12]

    print("\n" + "=" * 100)
    print("  TABLE II — Ablation Study (fold 0, test set)")
    print("=" * 100)

    hdr = "  " + "  ".join(f"{c:>{w[i]}}" for i, c in enumerate(cols))
    print(hdr)
    print("  " + "-" * 96)

    for _, row in df.iterrows():
        best_col = {
            "delta_snr": df["delta_snr"].idxmax(),
            "rmse":      df["rmse"].idxmin(),
            "pearson_r": df["pearson_r"].idxmax(),
            "ssim":      df["ssim"].idxmax(),
            "eta":       df["eta"].idxmin(),
        }
        line = (
            f"  {str(row['ablation_id']):>{w[0]}}  "
            f"{str(row['description'])[:w[1]]:>{w[1]}}  "
            f"{int(row['n_params']):>{w[2]},}  "
            f"{row['delta_snr']:>{w[3]}.4f}  "
            f"{row['rmse']:>{w[4]}.4f}  "
            f"{row['pearson_r']:>{w[5]}.4f}  "
            f"{row['ssim']:>{w[6]}.4f}  "
            f"{row['eta']:>{w[7]}.4f}"
        )
        print(line)

    print("=" * 100 + "\n")


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设备
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    logger.info("训练/推理设备：%s", device)

    # 数据
    logger.info("加载数据集…")
    loader_tr, loader_val, n_test, c_test = build_loaders(
        args.data_dir, args.batch_size
    )

    records: list[dict] = []

    for ablation_id, description, use_ms, use_se, loss_type in ABLATION_CONFIGS:
        logger.info("=" * 60)
        logger.info("配置 %s：%s", ablation_id, description)
        logger.info("=" * 60)

        model = build_model(ablation_id, cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("参数量：%d", n_params)

        save_path = args.ckpt_dir / "ablation" / f"{ablation_id}_best.pth"

        # ── A1–A5 统一训练路径：skip_train=True 且 checkpoint 存在时跳过 ──
        loss_fn = build_loss(loss_type, cfg).to(device)

        if args.skip_train and save_path.exists():
            logger.info("%s: skip_train=True，从 %s 加载", ablation_id, save_path)
            state = torch.load(save_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"])
        else:
            train_one_config(
                ablation_id, model, loss_fn,
                loader_tr, loader_val, device,
                args.epochs, args.patience, save_path,
            )
            state = torch.load(save_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            logger.info(
                "%s: 训练完成（best val_loss=%.5f，epoch=%d）",
                ablation_id, state["val_loss"], state["epoch"],
            )

        # ── 评估 ─────────────────────────────────────────────────────────
        model.eval()
        metrics = evaluate_model(model, n_test, c_test, device)

        logger.info(
            "%s 测试结果：ΔSNR=%.2f dB  RMSE=%.4f  Pearson_r=%.4f  "
            "SSIM=%.4f  eta=%.2f%%",
            ablation_id,
            metrics.get("delta_snr", float("nan")),
            metrics.get("rmse",      float("nan")),
            metrics.get("pearson_r", float("nan")),
            metrics.get("ssim",      float("nan")),
            metrics.get("eta",       float("nan")),
        )

        records.append({
            "ablation_id": ablation_id,
            "description": description,
            "use_ms_conv": use_ms,
            "use_se":      use_se,
            "loss_type":   loss_type,
            "n_params":    n_params,
            **metrics,
        })

    # ── 保存 CSV & 打印 ────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = args.output_dir / "Table_II_ablation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Table II 已保存：%s", csv_path)

    _print_table(df)

    # ── 生成消融趋势图 ────────────────────────────────────────────────────
    plot_ablation_figure(
        df,
        output_pdf=args.output_dir / "figures" / "Figure_ablation.pdf",
    )

    logger.info("消融实验全部完成。")


if __name__ == "__main__":
    main()
