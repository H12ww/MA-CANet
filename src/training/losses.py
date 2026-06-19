"""混合损失函数：MSE + 频域损失 + SSIM 损失。

Loss = λ1·MSE + λ2·FrequencyLoss + λ3·SSIMLoss1D
默认权重：λ1=1.0, λ2=0.1, λ3=0.1（来自 configs/default.yaml）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLoss(nn.Module):
    """频域 L1 损失：在 FFT 幅度谱上计算差异。

    保证模型在去除伪影的同时保留血流动力学频带（0.01–0.1 Hz），
    对高频成分通过 log 压缩降低权重。

    Args:
        log_scale: 若为 True，在 log 幅度谱上计算损失（默认 True）。
    """

    def __init__(self, log_scale: bool = True) -> None:
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频域 L1 损失。

        Args:
            pred:   预测信号，形状 (B, C, L)。
            target: 干净参考信号，形状 (B, C, L)。

        Returns:
            标量频域损失。
        """
        pred_fft   = torch.fft.rfft(pred,   dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)

        pred_mag   = pred_fft.abs()
        target_mag = target_fft.abs()

        if self.log_scale:
            pred_mag   = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)

        return F.l1_loss(pred_mag, target_mag)


class SSIMLoss(nn.Module):
    """一维结构相似性（SSIM）损失。

    用高斯滑动窗口沿时间轴计算局部 SSIM，返回 1 - mean_SSIM，
    最小化此损失等价于最大化 SSIM。

    Args:
        window_size: 滑动窗口大小（采样点数），默认 51。
        sigma:       高斯核标准差，默认 5.0。
        data_range:  信号数值范围，归一化信号为 1.0。
        k1:          SSIM 稳定常数 k1。
        k2:          SSIM 稳定常数 k2。
    """

    def __init__(
        self,
        window_size: int = 51,
        sigma: float = 5.0,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma       = sigma
        self.data_range  = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

    def _gaussian_kernel(self, device: torch.device) -> torch.Tensor:
        """构建归一化 1D 高斯核，形状 (1, 1, window_size)。"""
        half = self.window_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)   # (1, 1, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 1 - SSIM 损失。

        Args:
            pred:   预测信号，形状 (B, C, L)。
            target: 干净参考信号，形状 (B, C, L)。

        Returns:
            标量 SSIM 损失（1 - mean_SSIM）。
        """
        B, C, L = pred.shape
        kernel = self._gaussian_kernel(pred.device)
        pad    = self.window_size // 2

        # 合并 B×C 维度统一做卷积
        p = pred.view(B * C, 1, L)
        t = target.view(B * C, 1, L)

        mu1 = F.conv1d(p, kernel, padding=pad)
        mu2 = F.conv1d(t, kernel, padding=pad)

        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv1d(p * p, kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv1d(t * t, kernel, padding=pad) - mu2_sq
        sigma12   = F.conv1d(p * t, kernel, padding=pad) - mu1_mu2

        # 防止负方差（数值误差）
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)

        numerator   = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)

        ssim_map = numerator / denominator.clamp(min=1e-8)
        return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    """加权混合损失：MSE + 频域损失 + SSIM 损失。

    Loss = λ1·MSE(ŷ,y) + λ2·FreqLoss(ŷ,y) + λ3·SSIMLoss(ŷ,y)

    Args:
        mse_weight:  MSE 权重 λ1，默认 1.0。
        freq_weight: 频域损失权重 λ2，默认 0.1。
        ssim_weight: SSIM 损失权重 λ3，默认 0.1。
        log_freq:    是否在 log 幅度谱上计算频域损失。
    """

    def __init__(
        self,
        mse_weight:  float = 1.0,
        freq_weight: float = 0.1,
        ssim_weight: float = 0.1,
        log_freq:    bool  = True,
    ) -> None:
        super().__init__()
        self.mse_weight  = mse_weight
        self.freq_weight = freq_weight
        self.ssim_weight = ssim_weight
        self.freq_loss   = FrequencyLoss(log_scale=log_freq)
        self.ssim_loss   = SSIMLoss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算混合损失并返回各分量。

        Args:
            pred:   预测信号，形状 (B, C, L)。
            target: 干净参考信号，形状 (B, C, L)。

        Returns:
            total_loss: 标量总损失。
            components: 各损失分量字典，键为 'mse', 'freq', 'ssim'（供 TensorBoard 记录）。
        """
        mse  = F.mse_loss(pred, target)
        freq = self.freq_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total = (
            self.mse_weight  * mse
            + self.freq_weight * freq
            + self.ssim_weight * ssim
        )
        components = {
            "mse":  mse.item(),
            "freq": freq.item(),
            "ssim": ssim.item(),
        }
        return total, components

    @classmethod
    def from_config(cls, config: dict) -> "HybridLoss":
        """从配置字典实例化（键来自 configs/default.yaml 的 loss 节）。

        Args:
            config: 完整配置字典或仅 loss 子字典。

        Returns:
            配置好的 HybridLoss 实例。
        """
        lcfg = config.get("loss", config)
        return cls(
            mse_weight  = lcfg.get("mse_weight",  1.0),
            freq_weight = lcfg.get("freq_weight",  0.1),
            ssim_weight = lcfg.get("ssim_weight",  0.1),
        )


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  losses.py 单元测试")
    print("=" * 55)

    B, C, L = 4, 1, 512
    pred   = torch.randn(B, C, L)
    target = torch.randn(B, C, L)
    same   = pred.clone()

    # ── FrequencyLoss ─────────────────────────────────────────
    fl = FrequencyLoss(log_scale=True)
    loss_diff = fl(pred, target)
    loss_same = fl(pred, same)
    assert loss_same.item() < 1e-6, f"相同信号频域损失应为 0，得 {loss_same.item()}"
    assert loss_diff.item() > 0
    print(f"[PASS] FrequencyLoss  不同={loss_diff.item():.4f}  相同={loss_same.item():.2e}")

    # ── SSIMLoss ──────────────────────────────────────────────
    sl = SSIMLoss()
    loss_diff = sl(pred, target)
    loss_same = sl(pred, same)
    assert loss_same.item() < 1e-5, f"相同信号 SSIM 损失应接近 0，得 {loss_same.item()}"
    assert 0 <= loss_diff.item() <= 2.0
    print(f"[PASS] SSIMLoss       不同={loss_diff.item():.4f}  相同={loss_same.item():.2e}")

    # ── HybridLoss ────────────────────────────────────────────
    hl = HybridLoss(mse_weight=1.0, freq_weight=0.1, ssim_weight=0.1)
    total, comps = hl(pred, target)
    assert total.item() > 0
    assert set(comps.keys()) == {"mse", "freq", "ssim"}
    print(f"[PASS] HybridLoss     total={total.item():.4f}  {comps}")

    # ── from_config ───────────────────────────────────────────
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hl_cfg = HybridLoss.from_config(cfg)
    total2, _ = hl_cfg(pred, target)
    assert total2.item() > 0
    print(f"[PASS] from_config    total={total2.item():.4f}")

    print("=" * 55)
    print("  全部测试通过")
    print("=" * 55)
    sys.exit(0)
