"""MA-CANet: Multi-scale Attention-enhanced Convolutional Network for fNIRS artifact removal.

架构（输入长度 L=512）：
    MS-Conv stem (kernels [3,7,15,31])
    → Encoder × 4 层（MaxPool×2 下采样）
    → Bottleneck（Conv + Dropout）
    → Decoder × 4 层（Upsample + 跳跃连接 + SE）
    → 1×1 Conv 输出

目标：参数量 < 500 K，CPU 推理 < 5 ms。
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

from src.models.modules import DecoderBlock, EncoderBlock, MSConvBlock, SEBlock

logger = logging.getLogger(__name__)


class MACANet(nn.Module):
    """Multi-scale Attention-enhanced Convolutional Autoencoder Network.

    Args:
        in_channels:      输入通道数，默认 1（单通道处理）。
        ms_kernels:       MS-Conv 并行卷积核大小列表。
        ms_out_channels:  MS-Conv stem 输出通道数。
        encoder_channels: 各编码器层的输出通道数（共 4 层）。
        se_reduction:     SEBlock 压缩比，默认 8。
        dropout:          Bottleneck Dropout 概率，默认 0.3。

    Example::

        model = MACANet()
        x = torch.randn(4, 1, 512)
        y = model(x)   # (4, 1, 512)
    """

    def __init__(
        self,
        in_channels: int = 1,
        ms_kernels: List[int] = (3, 7, 15, 31),
        ms_out_channels: int = 32,
        encoder_channels: List[int] = (32, 64, 128, 128),
        se_reduction: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        enc_ch = list(encoder_channels)
        n_levels = len(enc_ch)

        # ── MS-Conv stem ─────────────────────────────────────────────────
        self.stem = MSConvBlock(in_channels, ms_out_channels, list(ms_kernels))

        # ── Encoder ──────────────────────────────────────────────────────
        enc_in_ch = [ms_out_channels] + enc_ch[:-1]
        self.encoders = nn.ModuleList([
            EncoderBlock(enc_in_ch[i], enc_ch[i], se_reduction)
            for i in range(n_levels)
        ])

        # ── Bottleneck ───────────────────────────────────────────────────
        bn_ch = enc_ch[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bn_ch, bn_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bn_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Decoder ──────────────────────────────────────────────────────
        # Dec[i]: in = rev[i], skip = rev[i], out = rev[i+1] (or ms_out_ch)
        rev = list(reversed(enc_ch))                       # [128,128,64,32]
        dec_out_ch = rev[1:] + [ms_out_channels]           # [128,64,32,32]
        self.decoders = nn.ModuleList([
            DecoderBlock(rev[i], rev[i], dec_out_ch[i], se_reduction)
            for i in range(n_levels)
        ])

        # ── Output ───────────────────────────────────────────────────────
        self.output_conv = nn.Conv1d(ms_out_channels, in_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 含伪影的 fNIRS 信号，形状 (B, in_channels, L)。

        Returns:
            重建的干净信号 ŷ，形状 (B, in_channels, L)。
        """
        # Stem
        x = self.stem(x)

        # Encode — 保存跳跃连接
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode — 反向使用跳跃连接
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.output_conv(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """返回可训练参数总量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "MACANet":
        """从配置字典实例化模型（键来自 configs/default.yaml 的 model 节）。

        Args:
            config: 完整配置字典或仅 model 子字典。

        Returns:
            配置好的 MACANet 实例。
        """
        mcfg = config.get("model", config)
        return cls(
            in_channels=mcfg.get("in_channels", 1),
            ms_kernels=mcfg.get("ms_kernels", [3, 7, 15, 31]),
            ms_out_channels=mcfg.get("ms_out_channels", 32),
            encoder_channels=mcfg.get("encoder_channels", [32, 64, 128, 128]),
            se_reduction=mcfg.get("se_reduction", 8),
            dropout=mcfg.get("dropout", 0.3),
        )


# ---------------------------------------------------------------------------
# 消融实验变体
# ---------------------------------------------------------------------------

class _PlainEncoderBlock(nn.Module):
    """不含 SE 的简单编码器块（消融用）。"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return self.pool(feat), feat


class _PlainDecoderBlock(nn.Module):
    """不含 SE 的简单解码器块（消融用）。"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return self.conv(torch.cat([x, skip], dim=1))


class MACANetAblation(nn.Module):
    """MA-CANet 消融变体，支持 A1~A5 五种配置。

    | ID | MS-Conv | SE  | 说明                     |
    |----|---------|-----|--------------------------|
    | A1 | ✗       | ✗   | 基础编码器-解码器         |
    | A2 | ✓       | ✗   | + 多尺度卷积              |
    | A3 | ✓       | ✓   | + SE 注意力               |
    | A4 | ✓       | ✓   | + 混合损失（架构同 A3）   |
    | A5 | ✓       | ✓   | 完整 MA-CANet             |

    Args:
        ablation_id:      'A1'~'A5'。
        in_channels:      输入通道数。
        ms_out_channels:  MS-Conv stem 输出通道数。
        encoder_channels: 各编码器层输出通道数。
        se_reduction:     SEBlock 压缩比。
        dropout:          Bottleneck Dropout 概率。
    """

    def __init__(
        self,
        ablation_id: str,
        in_channels: int = 1,
        ms_out_channels: int = 32,
        encoder_channels: List[int] = (32, 64, 128, 128),
        se_reduction: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if ablation_id not in ("A1", "A2", "A3", "A4", "A5"):
            raise ValueError(f"ablation_id 必须为 A1~A5，收到 '{ablation_id}'")

        self.ablation_id = ablation_id
        use_ms = ablation_id != "A1"
        use_se = ablation_id not in ("A1", "A2")

        enc_ch = list(encoder_channels)
        n = len(enc_ch)

        # Stem
        if use_ms:
            self.stem = MSConvBlock(in_channels, ms_out_channels)
            stem_out = ms_out_channels
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, ms_out_channels, 3, padding=1, bias=False),
                nn.BatchNorm1d(ms_out_channels),
                nn.ReLU(inplace=True),
            )
            stem_out = ms_out_channels

        # Encoder
        enc_in = [stem_out] + enc_ch[:-1]
        if use_se:
            self.encoders = nn.ModuleList([
                EncoderBlock(enc_in[i], enc_ch[i], se_reduction) for i in range(n)
            ])
        else:
            self.encoders = nn.ModuleList([
                _PlainEncoderBlock(enc_in[i], enc_ch[i]) for i in range(n)
            ])

        # Bottleneck
        bn_ch = enc_ch[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bn_ch, bn_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(bn_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Decoder
        rev = list(reversed(enc_ch))
        dec_out = rev[1:] + [ms_out_channels]
        if use_se:
            self.decoders = nn.ModuleList([
                DecoderBlock(rev[i], rev[i], dec_out[i], se_reduction) for i in range(n)
            ])
        else:
            self.decoders = nn.ModuleList([
                _PlainDecoderBlock(rev[i], rev[i], dec_out[i]) for i in range(n)
            ])

        self.output_conv = nn.Conv1d(ms_out_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x) if not isinstance(self.stem, nn.Sequential) else self.stem(x)

        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.output_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 入口测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml
    from pathlib import Path

    print("=" * 60)
    print("  MACANet 验证测试")
    print("=" * 60)

    # ── 1. 从 default.yaml 加载并实例化 ─────────────────────────────
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MACANet.from_config(cfg)
    model.eval()

    # ── 2. 前向传播 ──────────────────────────────────────────────────
    x = torch.randn(1, 1, 512)
    with torch.no_grad():
        y = model(x)

    print(f"输入形状  : {tuple(x.shape)}")
    print(f"输出形状  : {tuple(y.shape)}")
    assert y.shape == (1, 1, 512), f"输出形状错误：{y.shape}"
    print("[PASS] 输出形状 (1, 1, 512)")

    # ── 3. 参数量 ────────────────────────────────────────────────────
    n_params = model.count_parameters()
    print(f"\n总参数量  : {n_params:,}")
    assert n_params < 500_000, f"参数量超过 500K！({n_params:,})"
    print(f"[PASS] 参数量 {n_params/1000:.1f}K < 500K")

    # ── 4. 消融变体测试 ──────────────────────────────────────────────
    print("\n消融变体参数量：")
    for aid in ("A1", "A2", "A3", "A4", "A5"):
        abl = MACANetAblation(ablation_id=aid)
        n = abl.count_parameters()
        out = abl(x)
        assert out.shape == (1, 1, 512)
        print(f"  {aid}: {n:>7,} 参数  输出 {tuple(out.shape)}")

    print("\n" + "=" * 60)
    print("  全部测试通过")
    print("=" * 60)
    sys.exit(0)
