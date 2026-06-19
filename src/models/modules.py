"""MA-CANet 核心构建模块。

包含：
- MSConvBlock  — 多尺度并行 1D 卷积
- SEBlock      — Squeeze-and-Excitation 通道注意力
- EncoderBlock — 编码器单元（Conv + BN + ReLU + SE + Pool）
- DecoderBlock — 解码器单元（Upsample + Skip + Conv + BN + ReLU + SE）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSConvBlock(nn.Module):
    """多尺度并行 1D 卷积块。

    4 个并行分支，kernel_size ∈ [3, 7, 15, 31]，各接 BN + ReLU，
    Concatenate 后用 1×1 Conv 将通道降回 out_channels。

    Args:
        in_channels:  输入通道数。
        out_channels: 输出通道数（1×1 Conv 后）。
        kernels:      并行卷积核大小列表，默认 [3, 7, 15, 31]。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernels is None:
            kernels = [3, 7, 15, 31]

        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k,
                              padding=k // 2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 1×1 Conv 将拼接后的 len(kernels)×out_channels → out_channels
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * len(kernels), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力块。

    Squeeze : Global Average Pooling → (B, C, 1)
    Excite  : FC → ReLU → FC → Sigmoid → (B, C, 1)
    Scale   : element-wise multiply

    Args:
        channels:        输入/输出通道数。
        reduction_ratio: FC 瓶颈压缩比，默认 8。
    """

    def __init__(self, channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        reduced = max(1, channels // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        scale = x.mean(dim=-1)           # (B, C)
        scale = self.excitation(scale)   # (B, C)
        return x * scale.unsqueeze(-1)  # (B, C, L)


class EncoderBlock(nn.Module):
    """编码器单元：Conv1D → BN → ReLU → SEBlock → MaxPool1D(2)。

    Args:
        in_channels:     输入通道数。
        out_channels:    输出通道数（池化前）。
        reduction_ratio: SEBlock 压缩比，默认 8。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels, reduction_ratio)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Returns:
            pooled: 下采样后的特征，(B, out_channels, L/2)。
            skip:   池化前的特征（跳跃连接），(B, out_channels, L)。
        """
        feat = self.conv(x)
        feat = self.se(feat)
        return self.pool(feat), feat


class DecoderBlock(nn.Module):
    """解码器单元：Upsample(2) → Concat(skip) → Conv1D → BN → ReLU → SEBlock。

    Args:
        in_channels:     上采样输入通道数。
        skip_channels:   跳跃连接通道数（来自对应编码器层）。
        out_channels:    输出通道数。
        reduction_ratio: SEBlock 压缩比，默认 8。
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels, reduction_ratio)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x:    上一层特征，(B, in_channels, L)。
            skip: 跳跃连接特征，(B, skip_channels, 2L)。

        Returns:
            输出特征，(B, out_channels, 2L)。
        """
        x = self.up(x)
        # 处理因池化导致的长度差（差 ≤ 1 个时间步）
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.se(x)


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    B, L = 4, 512   # batch=4, 时间长度=512

    print("=" * 55)
    print("  modules.py 单元测试")
    print("=" * 55)

    # ── MSConvBlock ──────────────────────────────────────────
    ms = MSConvBlock(in_channels=1, out_channels=64)
    x_in = torch.randn(B, 1, L)
    out = ms(x_in)
    assert out.shape == (B, 64, L), f"MSConvBlock shape 错误：{out.shape}"
    print(f"[PASS] MSConvBlock  输入 {tuple(x_in.shape)} → 输出 {tuple(out.shape)}")

    ms2 = MSConvBlock(in_channels=64, out_channels=128, kernels=[3, 7])
    out2 = ms2(out)
    assert out2.shape == (B, 128, L)
    print(f"[PASS] MSConvBlock(kernels=[3,7])  输出 {tuple(out2.shape)}")

    # ── SEBlock ──────────────────────────────────────────────
    se = SEBlock(channels=64, reduction_ratio=8)
    x_se = torch.randn(B, 64, L)
    out_se = se(x_se)
    assert out_se.shape == x_se.shape, f"SEBlock shape 错误：{out_se.shape}"
    assert not torch.allclose(out_se, x_se)
    print(f"[PASS] SEBlock      输入 {tuple(x_se.shape)} → 输出 {tuple(out_se.shape)}")

    # ── EncoderBlock ─────────────────────────────────────────
    enc = EncoderBlock(in_channels=1, out_channels=64)
    x_enc = torch.randn(B, 1, L)
    pooled, skip = enc(x_enc)
    assert pooled.shape == (B, 64, L // 2), f"EncoderBlock pooled shape 错误：{pooled.shape}"
    assert skip.shape  == (B, 64, L),       f"EncoderBlock skip shape 错误：{skip.shape}"
    print(f"[PASS] EncoderBlock 输入 {tuple(x_enc.shape)} → pooled {tuple(pooled.shape)}, skip {tuple(skip.shape)}")

    # ── DecoderBlock ─────────────────────────────────────────
    dec = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
    out_dec = dec(pooled, skip)
    assert out_dec.shape == (B, 32, L), f"DecoderBlock shape 错误：{out_dec.shape}"
    print(f"[PASS] DecoderBlock 输入 {tuple(pooled.shape)} + skip {tuple(skip.shape)} → 输出 {tuple(out_dec.shape)}")

    # ── 奇数长度鲁棒性测试 ───────────────────────────────────
    L_odd = 511
    enc2 = EncoderBlock(in_channels=1, out_channels=32)
    pooled_odd, skip_odd = enc2(torch.randn(B, 1, L_odd))
    dec2 = DecoderBlock(in_channels=32, skip_channels=32, out_channels=16)
    out_odd = dec2(pooled_odd, skip_odd)
    assert out_odd.shape[-1] == skip_odd.shape[-1], \
        f"奇数长度对齐失败：{out_odd.shape} vs skip {skip_odd.shape}"
    print(f"[PASS] 奇数长度(L={L_odd}) DecoderBlock 对齐通过")

    print("=" * 55)
    print("  全部测试通过")
    print("=" * 55)
    sys.exit(0)
