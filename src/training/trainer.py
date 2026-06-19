"""MA-CANet 训练管线。

包含：
- EarlyStopping：监控验证损失，保存最优权重
- Trainer：完整训练循环，支持 AdamW + CosineAnnealingLR + TensorBoard + checkpoint
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ===========================================================================
# EarlyStopping
# ===========================================================================

class EarlyStopping:
    """监控验证损失，触发早停并保存最优权重。

    Args:
        patience:        无改善时等待的 epoch 数，默认 20。
        min_delta:       视为改善的最小绝对变化量。
        checkpoint_path: 最优权重保存路径；None 则不保存。
    """

    def __init__(
        self,
        patience:        int            = 20,
        min_delta:       float          = 1e-6,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss: float = float("inf")
        self.counter:   int   = 0
        self.should_stop:bool = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """根据当前验证损失更新状态。

        Args:
            val_loss: 当前 epoch 验证损失。
            model:    有改善时保存其权重。

        Returns:
            True = 应当停止训练；False = 继续训练。
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
                logger.debug("最优权重已保存：%s", self.checkpoint_path)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """MA-CANet（及消融变体）的完整训练管线。

    职责：
    - 训练循环（train_epoch / validate_epoch）
    - CosineAnnealingLR 调度与梯度裁剪
    - EarlyStopping + 最优 checkpoint 保存
    - TensorBoard 损失曲线与学习率日志
    - 每 epoch 打印 train_loss / val_loss / val_metrics（RMSE, Pearson r, ΔSNR）

    Args:
        model:        要训练的 PyTorch 模型。
        train_loader: 训练集 DataLoader。
        val_loader:   验证集 DataLoader。
        criterion:    损失函数（HybridLoss 返回 (total, components_dict)，
                      也可是普通 nn.Module 返回标量）。
        config:       完整配置字典（来自 configs/default.yaml）。
        output_dir:   根输出目录；checkpoints/logs 保存在其子目录中。
        device:       'auto'、'cuda' 或 'cpu'；'auto' 时自动检测 CUDA。
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        criterion:    nn.Module,
        config:       dict,
        output_dir:   Path | str = "outputs",
        device:       str | torch.device = "auto",
        verbose:      bool = False,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.config       = config
        self.output_dir   = Path(output_dir)
        self._verbose     = verbose   # 若 True，在验证时打印被跳过的常数通道数

        # 设备
        if isinstance(device, str) and device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info("训练设备：%s", self.device)
        self.model.to(self.device)

        # 训练超参数
        tcfg = config.get("training", {})
        self._epochs       = tcfg.get("epochs", 200)
        self._grad_clip    = tcfg.get("gradient_clip", 1.0)
        self._patience     = tcfg.get("early_stopping_patience", 20)

        # 时间戳（用于文件命名）
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_early_stopping()
        self._setup_logging()

    # ------------------------------------------------------------------
    # 内部初始化
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        tcfg = self.config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = tcfg.get("lr", 1e-3),
            weight_decay = tcfg.get("weight_decay", 1e-4),
        )

    def _setup_scheduler(self) -> None:
        tcfg  = self.config.get("training", {})
        T_max = tcfg.get("scheduler_T_max", self._epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=1e-6
        )

    def _setup_early_stopping(self) -> None:
        ckpt_dir  = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / f"best_{self._run_id}.pth"
        self.early_stopping = EarlyStopping(
            patience        = self._patience,
            checkpoint_path = best_path,
        )

    def _setup_logging(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.output_dir / "logs" / self._run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer: Optional[object] = SummaryWriter(log_dir=str(log_dir))
            logger.info("TensorBoard 日志目录：%s", log_dir)
        except ImportError:
            logger.warning("tensorboard 未安装，跳过 TensorBoard 日志。")
            self.writer = None

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------
    # 损失计算辅助
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """统一处理 HybridLoss（返回 tuple）和普通 loss（返回标量）。"""
        out = self.criterion(pred, target)
        if isinstance(out, tuple):
            total, components = out
        else:
            total      = out
            components = {"mse": total.item()}
        return total, components

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """运行一个训练 epoch，返回平均训练损失。"""
        self.model.train()
        total_loss = 0.0

        for noisy, clean in self.train_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(noisy)
            loss, _ = self._compute_loss(pred, clean)
            loss.backward()

            if self._grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """运行一个验证 epoch，返回 (val_loss, metrics_dict)。

        metrics_dict 包含：rmse, pearson_r, delta_snr（均在 CPU numpy 上计算）。
        """
        self.model.eval()
        total_loss = 0.0

        all_pred:   List[np.ndarray] = []
        all_clean:  List[np.ndarray] = []
        all_noisy:  List[np.ndarray] = []

        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                pred  = self.model(noisy)

                loss, _ = self._compute_loss(pred, clean)
                total_loss += loss.item()

                all_pred.append(pred.cpu().numpy())
                all_clean.append(clean.cpu().numpy())
                all_noisy.append(noisy.cpu().numpy())

        val_loss = total_loss / max(1, len(self.val_loader))

        # --- 批量指标计算（在 numpy 上进行）---
        P = np.concatenate(all_pred,  axis=0)   # (N, 1, 512)
        C = np.concatenate(all_clean, axis=0)
        N = np.concatenate(all_noisy, axis=0)

        metrics = _compute_quick_metrics(P, C, N, verbose=self._verbose)
        return val_loss, metrics

    def fit(self) -> Dict[str, list]:
        """运行完整训练循环直到完成或触发早停。

        Returns:
            history 字典，键为 'train_loss'、'val_loss'，值为每 epoch 的列表。
        """
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        logger.info("开始训练，共 %d epochs，设备：%s", self._epochs, self.device)

        header = (
            f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>10}"
            f"  {'RMSE':>7}  {'Pearson_r':>9}  {'dSNR':>6}  {'lr':>9}"
        )
        print(header)
        print("-" * len(header))

        for epoch in range(1, self._epochs + 1):
            t0 = time.time()

            train_loss            = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]

            # TensorBoard
            self._log_scalar("Loss/train", train_loss, epoch)
            self._log_scalar("Loss/val",   val_loss,   epoch)
            self._log_scalar("LR",         lr,         epoch)
            for k, v in val_metrics.items():
                self._log_scalar(f"Val/{k}", v, epoch)

            # 控制台打印
            elapsed = time.time() - t0
            print(
                f"{epoch:>6}/{self._epochs}"
                f"  {train_loss:>10.4f}  {val_loss:>10.4f}"
                f"  {val_metrics.get('rmse', float('nan')):>7.4f}"
                f"  {val_metrics.get('pearson_r', float('nan')):>9.4f}"
                f"  {val_metrics.get('delta_snr', float('nan')):>6.2f}"
                f"  {lr:>9.2e}"
                f"  [{elapsed:.1f}s]"
            )

            # 早停
            if self.early_stopping.step(val_loss, self.model):
                logger.info(
                    "早停触发（epoch %d，patience=%d）", epoch, self._patience
                )
                print(f"\n[早停] epoch {epoch}，最优验证损失 = {self.early_stopping.best_loss:.4f}")
                break

        if self.writer is not None:
            self.writer.close()

        logger.info("训练结束，最优验证损失：%.4f", self.early_stopping.best_loss)
        return history

    def load_best_checkpoint(self) -> None:
        """从早停保存的 checkpoint 加载最优权重。

        Raises:
            FileNotFoundError: 若 checkpoint 尚未保存。
        """
        ckpt = self.early_stopping.checkpoint_path
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(
                f"找不到最优 checkpoint：{ckpt}\n请先调用 fit() 完成训练。"
            )
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        logger.info("已加载最优权重：%s", ckpt)

    def save_checkpoint(
        self,
        path:  Optional[Path] = None,
        epoch: Optional[int]  = None,
    ) -> Path:
        """保存当前模型权重（含优化器和调度器状态）。

        Args:
            path:  目标路径；None 时自动按时间戳生成。
            epoch: 写入文件名的 epoch 编号。

        Returns:
            实际保存的路径。
        """
        if path is None:
            ckpt_dir = self.output_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_ep{epoch}" if epoch is not None else ""
            path   = ckpt_dir / f"ckpt_{self._run_id}{suffix}.pth"

        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    self.model.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict(),
                "scheduler_state_dict":self.scheduler.state_dict(),
                "best_val_loss":       self.early_stopping.best_loss,
            },
            path,
        )
        logger.info("Checkpoint 已保存：%s", path)
        return path

    @classmethod
    def from_config(
        cls,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        criterion:    nn.Module,
        config:       dict,
        output_dir:   Path | str = "outputs",
    ) -> "Trainer":
        """从 default.yaml 配置字典创建 Trainer。

        device 由 config['device'] 确定（'auto'/'cuda'/'cpu'）。
        """
        device = config.get("device", "auto")
        return cls(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            criterion    = criterion,
            config       = config,
            output_dir   = output_dir,
            device       = device,
        )


# ===========================================================================
# 训练时快速指标（委托给 metrics.py 统一实现）
# ===========================================================================

def _compute_quick_metrics(
    pred:    np.ndarray,        # (N, 1, L)
    clean:   np.ndarray,        # (N, 1, L)
    noisy:   np.ndarray,        # (N, 1, L)
    verbose: bool = False,
) -> Dict[str, float]:
    """计算训练时验证集快速指标：RMSE、Pearson r、ΔSNR。

    将 (N, 1, L) 张量 reshape 为 (N, L)，每条样本视作独立通道，
    委托给 metrics.py 的健壮实现（含常数信号跳过 + skip 计数日志）。
    """
    from src.evaluation.metrics import (
        delta_snr as _delta_snr,
        rmse      as _rmse,
        pearson_r as _pearson_r,
    )

    P = pred[:, 0, :]     # (N, L)
    C = clean[:, 0, :]    # (N, L)
    N = noisy[:, 0, :]    # (N, L)

    return {
        "rmse":      _rmse(P, C),
        "pearson_r": _pearson_r(P, C, verbose=verbose),
        "delta_snr": _delta_snr(N, P, C),
    }


# ===========================================================================
# 单元测试（不依赖真实数据）
# ===========================================================================

if __name__ == "__main__":
    import sys
    import yaml
    from pathlib import Path
    from torch.utils.data import TensorDataset

    print("=" * 60)
    print("  trainer.py 单元测试（小规模合成数据）")
    print("=" * 60)

    # 加载配置
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 快速测试：缩短 epochs，减小 patience
    cfg["training"]["epochs"]                   = 5
    cfg["training"]["early_stopping_patience"]  = 3

    # 合成数据集
    torch.manual_seed(0)
    N, C, L = 64, 1, 512
    noisy_t = torch.randn(N, C, L)
    clean_t = torch.randn(N, C, L)
    ds      = TensorDataset(noisy_t, clean_t)
    loader  = DataLoader(ds, batch_size=16, shuffle=False)

    # 构造简单模型和损失
    from src.models.macanet import MACANet
    from src.training.losses import HybridLoss

    model     = MACANet.from_config(cfg)
    criterion = HybridLoss.from_config(cfg)

    trainer = Trainer(
        model        = model,
        train_loader = loader,
        val_loader   = loader,
        criterion    = criterion,
        config       = cfg,
        output_dir   = Path("outputs") / "test_run",
        device       = "cpu",
    )

    history = trainer.fit()
    assert len(history["train_loss"]) > 0
    assert len(history["val_loss"])   == len(history["train_loss"])
    print(f"\n[PASS] 训练完成，共 {len(history['train_loss'])} epochs")

    # save_checkpoint
    ckpt_path = trainer.save_checkpoint(epoch=len(history["train_loss"]))
    assert ckpt_path.exists()
    print(f"[PASS] save_checkpoint -> {ckpt_path.name}")

    # load_best_checkpoint
    trainer.load_best_checkpoint()
    print("[PASS] load_best_checkpoint")

    print("=" * 60)
    print("  全部测试通过")
    print("=" * 60)
    sys.exit(0)
