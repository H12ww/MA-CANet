"""fNIRS 运动伪影去除基线方法。

每个方法均以函数或类的形式暴露，输入含噪信号、输出去噪信号。
所有方法的信号格式统一为 numpy 数组 (n_channels, n_timepoints)，
或对 DAE 使用 PyTorch 张量 (B, 1, L)。

方法列表：
1. BandpassFilter     — 0.01–0.1 Hz Butterworth 零相位滤波
2. WaveletThreshold   — db4 小波软阈值（Brigadoi 2014）
3. SplineInterpolation — 检测伪影 + 三次样条插值（Homer3 风格）
4. TDDR               — 时间导数分布修复（Fishburn 2019）
5. PCAMethod          — 主成分分析去除伪影成分
6. DAENet             — 8 层卷积去噪自编码器（Gao 2022 移植）
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Bandpass Filter
# ===========================================================================

class BandpassFilter:
    """Butterworth 带通滤波基线（0.01–0.1 Hz）。

    Args:
        fs:      采样频率（Hz），默认 10.0。
        low_hz:  低截止频率，默认 0.01 Hz。
        high_hz: 高截止频率，默认 0.1 Hz。
        order:   滤波器阶数，默认 4。
    """

    def __init__(
        self,
        fs:      float = 10.0,
        low_hz:  float = 0.01,
        high_hz: float = 0.1,
        order:   int   = 4,
    ) -> None:
        from scipy.signal import butter
        nyq  = fs / 2.0
        low  = low_hz  / nyq
        high = high_hz / nyq
        # 夹紧到 (0, 1) 避免数值问题
        low  = float(np.clip(low,  1e-6, 1 - 1e-6))
        high = float(np.clip(high, 1e-6, 1 - 1e-6))
        self._sos = butter(order, [low, high], btype="band", output="sos")

    def process(self, signal: np.ndarray) -> np.ndarray:
        """对每个通道施加零相位带通滤波。

        Args:
            signal: 输入信号，形状 (n_channels, n_timepoints)。

        Returns:
            滤波后信号，形状相同。
        """
        from scipy.signal import sosfiltfilt
        out = np.empty_like(signal)
        for ch in range(signal.shape[0]):
            out[ch] = sosfiltfilt(self._sos, signal[ch])
        return out


def bandpass_filter(
    signal: np.ndarray,
    fs:     float = 10.0,
    low:    float = 0.01,
    high:   float = 0.1,
) -> np.ndarray:
    """函数式接口：带通滤波。

    Args:
        signal: (n_channels, n_timepoints) 或 (n_timepoints,)。
        fs:     采样频率。
        low:    低截止频率（Hz）。
        high:   高截止频率（Hz）。

    Returns:
        滤波后信号，形状与输入相同。
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]
    out = BandpassFilter(fs, low, high).process(signal)
    return out[0] if squeeze else out


# ===========================================================================
# 2. Wavelet Threshold
# ===========================================================================

class WaveletThreshold:
    """小波软阈值去噪（db4，Brigadoi 2014）。

    Args:
        wavelet:        PyWavelets 小波名，默认 'db4'。
        level:          分解层数；None 则自动确定。
        threshold_mode: 'soft' 或 'hard'，默认 'soft'。
    """

    def __init__(
        self,
        wavelet:        str           = "db4",
        level:          Optional[int] = None,
        threshold_mode: str           = "soft",
    ) -> None:
        self.wavelet        = wavelet
        self.level          = level
        self.threshold_mode = threshold_mode

    def process(self, signal: np.ndarray) -> np.ndarray:
        """对每个通道应用小波阈值去噪。

        Args:
            signal: (n_channels, n_timepoints)。

        Returns:
            去噪后信号，形状相同。
        """
        import pywt
        out = np.empty_like(signal)
        for ch in range(signal.shape[0]):
            coeffs = pywt.wavedec(signal[ch], self.wavelet, level=self.level)
            # 用最细尺度估计噪声标准差（MAD）
            detail = coeffs[-1]
            sigma  = np.median(np.abs(detail)) / 0.6745
            thr    = sigma * np.sqrt(2 * np.log(len(signal[ch])))
            coeffs_thr = [coeffs[0]] + [
                pywt.threshold(c, thr, mode=self.threshold_mode)
                for c in coeffs[1:]
            ]
            out[ch] = pywt.waverec(coeffs_thr, self.wavelet)[: signal.shape[1]]
        return out


def wavelet_denoise(
    signal:  np.ndarray,
    wavelet: str           = "db4",
    level:   Optional[int] = 4,
) -> np.ndarray:
    """函数式接口：小波去噪。"""
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]
    out = WaveletThreshold(wavelet, level).process(signal)
    return out[0] if squeeze else out


# ===========================================================================
# 3. Spline Interpolation
# ===========================================================================

class SplineInterpolation:
    """样条插值基线（Homer3 风格）。

    检测到伪影区域后，用三次样条插值替换。

    Args:
        fs:            采样频率（Hz）。
        std_threshold: 伪影判定阈值（标准差倍数），默认 3.0。
        spline_order:  样条阶数，默认 3（cubic）。
    """

    def __init__(
        self,
        fs:            float = 10.0,
        std_threshold: float = 3.0,
        spline_order:  int   = 3,
    ) -> None:
        self.fs            = fs
        self.std_threshold = std_threshold
        self.spline_order  = spline_order

    def detect_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """返回伪影布尔掩码（True = 伪影）。

        使用 MAD（中位绝对偏差）估计噪声标准差，对异常值更鲁棒
        （Scholkmann et al. 2010 / Homer3 实现风格）。

        Args:
            signal: (n_channels, n_timepoints)。

        Returns:
            布尔掩码，形状相同。
        """
        # 时间一阶差分（对 spike 和 shift onset 均敏感）
        diff = np.diff(signal, axis=1, prepend=signal[:, :1])
        # MAD 估计鲁棒标准差：σ̂ = 1.4826 · MAD
        med  = np.median(diff, axis=1, keepdims=True)
        mad  = np.median(np.abs(diff - med), axis=1, keepdims=True)
        sigma_hat = 1.4826 * mad
        return np.abs(diff - med) > self.std_threshold * (sigma_hat + 1e-10)

    def process(self, signal: np.ndarray) -> np.ndarray:
        """检测伪影并用样条插值修复。

        Args:
            signal: (n_channels, n_timepoints)。

        Returns:
            修复后信号，形状相同。
        """
        from scipy.interpolate import make_interp_spline

        mask = self.detect_artifacts(signal)   # (C, T)
        out  = signal.copy()
        t    = np.arange(signal.shape[1])

        for ch in range(signal.shape[0]):
            art = mask[ch]
            clean_idx = np.where(~art)[0]
            # 干净点少于 4 个时跳过（无法拟合三次样条）
            if len(clean_idx) < max(4, self.spline_order + 1):
                continue
            try:
                spl = make_interp_spline(
                    clean_idx, signal[ch, clean_idx],
                    k=self.spline_order,
                )
                art_idx = np.where(art)[0]
                # 只插值在干净点范围内的伪影
                valid = (art_idx >= clean_idx[0]) & (art_idx <= clean_idx[-1])
                if valid.any():
                    out[ch, art_idx[valid]] = spl(art_idx[valid])
            except Exception:
                pass
        return out


def spline_correction(
    signal: np.ndarray,
    fs:     float = 10.0,
) -> np.ndarray:
    """函数式接口：样条插值去伪影。"""
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]
    out = SplineInterpolation(fs=fs).process(signal)
    return out[0] if squeeze else out


# ===========================================================================
# 4. TDDR — Temporal Derivative Distribution Repair
# ===========================================================================

class TDDR:
    """时间导数分布修复（Fishburn et al. 2019）。

    Reference: Fishburn FA, Ludlum RS, Vaidya CJ, Medvedev AV (2019).
        Temporal Derivative Distribution Repair (TDDR): A motion correction
        method for fNIRS. NeuroImage, 184, 171-179.

    Args:
        fs:         采样频率（Hz）。
        n_iter:     迭代次数，默认 50。
        tuning_k:   Tukey biweight 调参常数，默认 4.685。
    """

    def __init__(
        self,
        fs:       float = 10.0,
        n_iter:   int   = 50,
        tuning_k: float = 4.685,
    ) -> None:
        self.fs       = fs
        self.n_iter   = n_iter
        self.tuning_k = tuning_k

    def _tddr_channel(self, y: np.ndarray) -> np.ndarray:
        """对单通道信号应用 TDDR。"""
        mu = y.mean()
        y  = y - mu

        d = np.diff(y)                # 时间差分 (T-1,)
        w = np.ones_like(d)           # 初始权重

        for _ in range(self.n_iter):
            mu_d    = np.sum(w * d) / (np.sum(w) + 1e-10)
            sigma_d = np.sqrt(np.sum(w * (d - mu_d) ** 2) / (np.sum(w) + 1e-10))
            u       = (d - mu_d) / (self.tuning_k * sigma_d + 1e-10)
            # 限幅到 [-1, 1] 后再平方，避免 u>>1 时 u² 溢出
            u_safe  = np.where(np.abs(u) < 1.0, u, np.sign(u))
            w       = ((1 - u_safe ** 2) ** 2) * (np.abs(u) < 1.0)

        # 用加权差分重建信号
        corrected = np.concatenate([[0.0], np.cumsum(w * d)])
        return corrected + mu

    def process(self, signal: np.ndarray) -> np.ndarray:
        """对每个通道应用 TDDR。

        Args:
            signal: 光密度信号，形状 (n_channels, n_timepoints)。

        Returns:
            修复后信号，形状相同。
        """
        out = np.empty_like(signal)
        for ch in range(signal.shape[0]):
            out[ch] = self._tddr_channel(signal[ch])
        return out


def tddr_correction(
    signal: np.ndarray,
    fs:     float = 10.0,
) -> np.ndarray:
    """函数式接口：TDDR 运动修正。"""
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]
    out = TDDR(fs=fs).process(signal)
    return out[0] if squeeze else out


# ===========================================================================
# 5. PCA Method
# ===========================================================================

class PCAMethod:
    """靶向 PCA 去除伪影成分。

    将信号投影到主成分空间，清零与运动相关的头部成分后反投影。

    Args:
        n_artifact_components: 假设捕获伪影方差的 PC 数量，默认 1。
    """

    def __init__(self, n_artifact_components: int = 1) -> None:
        self.n_artifact_components = n_artifact_components

    def process(self, signal: np.ndarray) -> np.ndarray:
        """去除伪影主成分。

        Args:
            signal: (n_channels, n_timepoints)。

        Returns:
            修复后信号，形状相同。
        """
        # 中心化
        mean = signal.mean(axis=1, keepdims=True)
        X    = signal - mean                          # (C, T)

        # SVD：X = U @ diag(s) @ Vt，主成分在 Vt 的行
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        # 清零头部 n_artifact_components 个成分
        s_masked = s.copy()
        s_masked[: self.n_artifact_components] = 0.0

        out = (U * s_masked) @ Vt + mean
        return out


def pca_denoise(
    signals:               np.ndarray,
    n_artifact_components: int = 1,
) -> np.ndarray:
    """函数式接口：PCA 去噪（多通道输入）。

    Args:
        signals:               (n_channels, n_timepoints)。
        n_artifact_components: 去除的主成分数量。

    Returns:
        去噪信号，形状相同。
    """
    return PCAMethod(n_artifact_components).process(signals)


# ===========================================================================
# 6. DAE — Denoising Autoencoder (Gao 2022 Net_8layers 移植)
# ===========================================================================

class DAENet(nn.Module):
    """8 层卷积去噪自编码器，移植自 Gao et al. 2022 (Net_8layers)。

    原始实现将 HbO+HbR（共 1024 点）拼接后输入，此处适配为
    单通道 (B, 1, 512) 输入/输出，与项目数据格式一致。

    架构：
        Conv(1→32, k=11) → Pool × 4
        → Conv(32→32, k=3) bottleneck
        → Upsample × 4 → Conv(32→1, k=3)

    Args:
        input_length: 输入时间长度，默认 512。
    """

    def __init__(self, input_length: int = 512) -> None:
        super().__init__()
        # 编码器卷积
        self.conv1 = nn.Conv1d(1,  32, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        # 瓶颈
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        # 解码器卷积
        self.conv6 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.conv7 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.conv8 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        # 输出
        self.conv9 = nn.Conv1d(32, 1,  kernel_size=3,  padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 含噪信号，形状 (B, 1, L)。

        Returns:
            去噪信号，形状 (B, 1, L)。
        """
        # 编码
        x = self.pool(F.relu(self.conv1(x)))   # L/2
        x = self.pool(F.relu(self.conv2(x)))   # L/4
        x = self.pool(F.relu(self.conv3(x)))   # L/8
        x = self.pool(F.relu(self.conv4(x)))   # L/16
        # 瓶颈
        x = F.relu(self.conv5(x))
        # 解码
        x = self.up(F.relu(self.conv6(x)))     # L/8
        x = self.up(F.relu(self.conv7(x)))     # L/4
        x = self.up(F.relu(self.conv8(x)))     # L/2
        x = self.up(self.conv9(x))             # L
        return x


# 向后兼容别名
DAE = DAENet


def dae_denoise(
    signal:  np.ndarray,
    model:   Optional["DAENet"] = None,
    device:  str = "cpu",
) -> np.ndarray:
    """函数式接口：DAE 去噪。

    Args:
        signal: (n_channels, n_timepoints) 或 (n_timepoints,)。
        model:  已实例化的 DAENet；为 None 时自动创建（随机权重，仅用于形状测试）。
        device: 推理设备（'cpu' 或 'cuda'）。

    Returns:
        去噪信号，形状与输入相同。
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]

    if model is None:
        model = DAENet(input_length=signal.shape[1])
        model.eval()

    model = model.to(device)
    out = np.empty_like(signal)
    with torch.no_grad():
        for ch in range(signal.shape[0]):
            x = torch.from_numpy(
                signal[ch:ch+1, :].astype(np.float32)
            ).unsqueeze(0).to(device)   # (1, 1, L)
            y = model(x)
            out[ch] = y.squeeze().cpu().numpy()

    return out[0] if squeeze else out


# ===========================================================================
# 7. EnhancedDAE — 参数量对齐版 DAE（~300K，与 MA-CANet 对齐）
# ===========================================================================

class EnhancedDAE(nn.Module):
    """增强版去噪自编码器，参数量 ~300K，与 MA-CANet 对齐。

    架构特点（与 MA-CANet 对比的关键差异）：
    - 无跳跃连接（vs MA-CANet 有 4 个 skip connection）
    - 无 SE 注意力（vs MA-CANet 每个 block 有 SE）
    - 无多尺度卷积（vs MA-CANet 有 MS-Conv stem k=3,7,15,31）
    - 单尺度对称编解码

    输入：(B, 1, 512)
    输出：(B, 1, 512)
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder：1 -> 22 -> 44 -> 88 -> 176（通道数校准至 ~320K 与 MA-CANet 对齐）
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 22, kernel_size=15, padding=7),
            nn.BatchNorm1d(22), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(22, 44, kernel_size=9, padding=4),
            nn.BatchNorm1d(44), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(44, 88, kernel_size=7, padding=3),
            nn.BatchNorm1d(88), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(88, 176, kernel_size=5, padding=2),
            nn.BatchNorm1d(176), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(176, 176, kernel_size=3, padding=1),
            nn.BatchNorm1d(176), nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # Decoder：对称还原，无 skip 连接
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(176, 88, kernel_size=5, padding=2),
            nn.BatchNorm1d(88), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(88, 44, kernel_size=7, padding=3),
            nn.BatchNorm1d(44), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(44, 22, kernel_size=9, padding=4),
            nn.BatchNorm1d(22), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(22, 1, kernel_size=15, padding=7)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        return self.dec1(x)


# ===========================================================================
# 8. SmallDAE — 小型 DAE，可配置通道数（附录消融用）
# ===========================================================================

class SmallDAE(nn.Module):
    """小型 DAE，可配置 base_channels 控制参数量。

    用于附录"DAE 容量饱和"消融实验。

    Args:
        base_channels: 基础通道数。
            16 时约 60K 参数，24 时约 120K 参数。

    输入：(B, 1, 512)
    输出：(B, 1, 512)
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, 15, padding=7), nn.BatchNorm1d(c), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c, c * 2, 9, padding=4), nn.BatchNorm1d(c * 2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(c * 2, c * 4, 7, padding=3), nn.BatchNorm1d(c * 4), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(c * 4, c * 4, 3, padding=1), nn.BatchNorm1d(c * 4), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(c * 4, c * 2, 7, padding=3), nn.BatchNorm1d(c * 2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(c * 2, c, 9, padding=4), nn.BatchNorm1d(c), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(c, 1, 15, padding=7)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# ===========================================================================
# 9. CNNwP — Huang et al. 2024 复现（1D CNN with Penalty Network）
# ===========================================================================

class CNNwP(nn.Module):
    """Huang et al. 2024 — 1D CNN with Penalty Network。

    Reference: Frontiers in Neuroscience, 10.3389/fnins.2024.1432138

    架构：
    - 主干 1D CNN：7 层卷积（4×Conv+MaxPool + 3×Conv+UpSample）+ FCL(256) + output
    - Penalty 分支：Flatten + FCL(128) + output
    - 融合：CNN_output × Penalty_output（element-wise）+ 末端 FCL

    Args:
        window_size: 输入/输出时间长度，默认 512。

    输入：(B, 1, 512)
    输出：(B, 1, 512)
    """

    def __init__(self, window_size: int = 512) -> None:
        super().__init__()
        self.window_size = window_size

        # 主干 CNN：前 4 层 Conv+MaxPool
        self.conv_pool1 = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2)
        )  # 512 -> 256
        self.conv_pool2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2)
        )  # 256 -> 128
        self.conv_pool3 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2)
        )  # 128 -> 64
        self.conv_pool4 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2)
        )  # 64 -> 32

        # 主干 CNN：后 3 层 Conv+UpSample
        self.conv_up1 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        )  # 32 -> 64
        self.conv_up2 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        )  # 64 -> 128
        self.conv_up3 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        )  # 128 -> 256

        # AdaptiveAvgPool 压缩后 flatten：(B,16,256) → (B,16,16) → flat=256
        # 避免 Linear(4096,256) 引入 1M+ 参数
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.cnn_flatten_size = 16 * 16
        self.cnn_fcl = nn.Linear(self.cnn_flatten_size, 128)
        self.cnn_out = nn.Linear(128, window_size)

        # Penalty 分支：Flatten + FCL(128) + output
        self.penalty_fcl = nn.Linear(window_size, 128)
        self.penalty_out = nn.Linear(128, window_size)

        # 融合层：1x1 Conv（替代 Linear(512,512) 的 262K 参数）
        self.fusion_conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干 CNN
        c = self.conv_pool1(x)
        c = self.conv_pool2(c)
        c = self.conv_pool3(c)
        c = self.conv_pool4(c)
        c = self.conv_up1(c)
        c = self.conv_up2(c)
        c = self.conv_up3(c)                              # (B, 16, 256)
        c = self.adaptive_pool(c)                         # (B, 16, 16)
        c = c.flatten(1)                                  # (B, 256)
        c = torch.relu(self.cnn_fcl(c))                   # (B, 128)
        cnn_output = self.cnn_out(c)                      # (B, 512)

        # Penalty 分支
        x_flat = x.flatten(1)                             # (B, 512)
        p = torch.relu(self.penalty_fcl(x_flat))          # (B, 128)
        penalty_output = torch.sigmoid(self.penalty_out(p))  # (B, 512)

        # Element-wise 相乘 + 1x1 Conv 融合
        fused = (cnn_output * penalty_output).unsqueeze(1)  # (B, 1, 512)
        return self.fusion_conv(fused)                    # (B, 1, 512)


# ===========================================================================
# 10. LSTMAutoencoder — Yang et al. 2025 复现（LSTM-AE 三阶段架构）
# ===========================================================================

class LSTMAutoencoder(nn.Module):
    """Yang et al. 2025 — LSTM-Autoencoder。

    Reference: European Journal of Neuroscience, 10.1111/ejn.16679

    三阶段架构：
    1. Encoder (Conv1D)：提取形态学特征
    2. LSTM module：捕捉样本间时序相关性
    3. Decoder (Conv1D)：从隐空间重建

    Args:
        hidden_dim:      LSTM 隐藏维度，默认 64。
        num_lstm_layers: LSTM 层数，默认 2。

    输入：(B, 1, 512)
    输出：(B, 1, 512)
    """

    def __init__(self, hidden_dim: int = 80, num_lstm_layers: int = 2) -> None:
        super().__init__()
        # Encoder：Conv1D 形态学特征提取，输出 (B, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 512 -> 256
            nn.Conv1d(16, 32, kernel_size=9, padding=4), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 256 -> 128
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 128 -> 64
        )

        # LSTM 中间层：捕捉时序相关性
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0
        )
        self.lstm_proj = nn.Linear(hidden_dim * 2, 64)

        # Decoder：Conv1D 重建
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=9, padding=4), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(16, 1, kernel_size=15, padding=7)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)              # (B, 64, 64)

        feat_seq = feat.permute(0, 2, 1)    # (B, 64, 64)
        lstm_out, _ = self.lstm(feat_seq)   # (B, 64, hidden_dim*2)
        lstm_out = self.lstm_proj(lstm_out) # (B, 64, 64)
        lstm_out = lstm_out.permute(0, 2, 1)# (B, 64, 64)

        feat = feat + lstm_out              # 残差连接增强稳定性
        out = self.decoder(feat)            # (B, 1, 512)
        return out


# ===========================================================================
# 单元测试
# ===========================================================================

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)
    C, T = 4, 512   # 4 通道，512 时间点

    print("=" * 60)
    print("  baselines.py 单元测试")
    print("=" * 60)

    # 生成含 spike 伪影的测试信号
    clean  = rng.standard_normal((C, T)).astype(np.float32)
    spike  = np.zeros((C, T), dtype=np.float32)
    spike[:, 100] = 10.0
    noisy  = clean + spike

    # ── 1. BandpassFilter ────────────────────────────────────
    out = bandpass_filter(noisy, fs=10.0)
    assert out.shape == noisy.shape
    print(f"[PASS] bandpass_filter     {noisy.shape} -> {out.shape}")

    # ── 2. WaveletThreshold ──────────────────────────────────
    out = wavelet_denoise(noisy, wavelet="db4", level=4)
    assert out.shape == noisy.shape
    print(f"[PASS] wavelet_denoise     {noisy.shape} -> {out.shape}")

    # ── 3. SplineInterpolation ───────────────────────────────
    out = spline_correction(noisy, fs=10.0)
    assert out.shape == noisy.shape
    print(f"[PASS] spline_correction   {noisy.shape} -> {out.shape}")

    # ── 4. TDDR ──────────────────────────────────────────────
    out = tddr_correction(noisy, fs=10.0)
    assert out.shape == noisy.shape
    print(f"[PASS] tddr_correction     {noisy.shape} -> {out.shape}")

    # ── 5. PCA ───────────────────────────────────────────────
    out = pca_denoise(noisy, n_artifact_components=1)
    assert out.shape == noisy.shape
    print(f"[PASS] pca_denoise         {noisy.shape} -> {out.shape}")

    # ── 6. DAENet ────────────────────────────────────────────
    model = DAENet(input_length=T)
    model.eval()
    x = torch.from_numpy(noisy[:1]).unsqueeze(1)   # (1, 1, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape, f"DAENet shape: {y.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PASS] DAENet              {tuple(x.shape)} -> {tuple(y.shape)}  params={n_params:,}")

    # 单通道函数式接口
    out_1d = bandpass_filter(noisy[0])
    assert out_1d.ndim == 1
    print("[PASS] 单通道 bandpass_filter 接口")

    print("=" * 60)
    print("  全部测试通过")
    print("=" * 60)
    sys.exit(0)
