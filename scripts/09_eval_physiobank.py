"""PhysioBank 数据集上的 MA-CANet 评估（跨数据集泛化验证）。

数据集：motion-artifact-contaminated-fnirs-and-eeg-data-1.0.0
  - 9 条 fNIRS 记录（fnirs_1 ~ fnirs_9）
  - 采样率 200 Hz，4 通道（690 nm × 2 + 830 nm × 2）
  - trigger 标注：F = 伪影开始，R = 伪影结束

处理流程：
  1. wfdb 读取 → NaN 插值 → 200 Hz 降采样至 10 Hz
  2. MA-CANet 5 折集成推理（重叠加法重建）
  3. 无参考指标评估：
     (a) MAD 伪影检测率（去噪前后及削减率）
     (b) 伪影区段与安静区段的频谱相似度（去噪前后）
     (c) 高频成分比（伪影带 0.5-5 Hz / 血流动力学带 0.01-0.5 Hz）

输出：
  outputs/Table_III_physiobank.csv       — 9 条记录的指标汇总表
  outputs/figures/Figure_physiobank.pdf  — 可视化图（波形 + PSD + 指标箱线图）

用法::

    python scripts/09_eval_physiobank.py \\
        [--data-dir data/raw/motion-artifact-contaminated-fnirs-and-eeg-data-1.0.0] \\
        [--ckpt-dir outputs/checkpoints] \\
        [--output-dir outputs] \\
        [--device auto]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================
# 常量
# ============================================================

SRC_FS    = 200       # PhysioBank 采样频率（Hz）
TGT_FS    = 10        # 模型训练时采样频率（Hz）
DS_FACTOR = SRC_FS // TGT_FS   # 降采样比 = 20

SEG_LEN   = 512       # 模型输入长度（10 Hz 下约 51.2 s）
HOP       = SEG_LEN // 2       # 50% 重叠步长

# 频率带（10 Hz 系）
HMD_LO, HMD_HI = 0.01, 0.5    # 血流动力学带
ART_LO,  ART_HI = 0.5,  5.0   # 伪影带（高频）

# MAD 伪影检测阈值
MAD_K = 1.4826
MAD_THR_SIGMA = 3.5            # σ 倍数

# 安静区段：取每个 F 开始前 60 s（最少需要 30 s）
REST_MARGIN_S  = 60.0
REST_MIN_LEN_S = 30.0


# ============================================================
# 数据读取与预处理
# ============================================================

def load_fnirs_record(
    data_dir: Path,
    rec_id: int,
) -> tuple[np.ndarray, list[tuple[int, str]], int]:
    """用 wfdb 读取 fNIRS 记录并降采样至 10 Hz。

    Args:
        data_dir: PhysioBank 数据目录。
        rec_id:   记录编号（1~9）。

    Returns:
        (signal, annotations, fs_out) 元组。
        signal:      形状 (n_samples_10Hz, n_ch) 的 float64 数组。
        annotations: [(sample_10hz, symbol), ...] 列表。
        fs_out:      输出采样频率（= TGT_FS = 10）。
    """
    import wfdb
    from scipy.signal import resample_poly

    rec_name = str(data_dir / f"fnirs_{rec_id}")
    rec      = wfdb.rdrecord(rec_name)

    # 仅提取 fNIRS 通道（排除 ACC 和 trigger）
    fnirs_idx = [
        j for j, n in enumerate(rec.sig_name)
        if "fNIRS" in n and "tr" not in n.lower()
    ]
    sig_raw = rec.p_signal[:, fnirs_idx].copy()   # (T, C)

    # 线性插值补全 NaN
    for ch in range(sig_raw.shape[1]):
        col = sig_raw[:, ch]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            idx = np.where(~nan_mask)[0]
            if len(idx) >= 2:
                sig_raw[:, ch] = np.interp(np.arange(len(col)), idx, col[idx])
            else:
                sig_raw[:, ch] = 0.0

    # 200 Hz → 10 Hz 降采样
    sig_10 = resample_poly(sig_raw, 1, DS_FACTOR, axis=0)

    # trigger 标注 → 换算为 10 Hz 采样点
    annotations: list[tuple[int, str]] = []
    try:
        ann = wfdb.rdann(rec_name, "trigger")
        for samp, sym in zip(ann.sample, ann.symbol):
            annotations.append((int(samp // DS_FACTOR), sym))
    except Exception:
        logger.warning("fnirs_%d: trigger 标注读取失败", rec_id)

    logger.info(
        "fnirs_%d: %d 样本 @ %d Hz  %d 通道  %d 个标注",
        rec_id, sig_10.shape[0], TGT_FS, sig_10.shape[1], len(annotations),
    )
    return sig_10.astype(np.float64), annotations, TGT_FS


def parse_intervals(
    annotations: list[tuple[int, str]],
    sig_len: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """从 F/R 标注中提取伪影区段和安静区段。

    Args:
        annotations: [(sample, 'F'|'R'), ...] 列表（10 Hz 换算后）。
        sig_len:     信号长度（采样点数）。

    Returns:
        (artifact_intervals, rest_intervals) 元组，
        各列表格式为 [(start, end), ...]。
    """
    f_samples = sorted(s for s, sym in annotations if sym == "F")
    r_samples = sorted(s for s, sym in annotations if sym == "R")

    artifact_ivs: list[tuple[int, int]] = []
    for f in f_samples:
        # 找紧随其后的 R
        rs_after = [r for r in r_samples if r > f]
        end = rs_after[0] if rs_after else sig_len
        artifact_ivs.append((f, end))

    rest_ivs: list[tuple[int, int]] = []
    rest_min = int(REST_MIN_LEN_S * TGT_FS)
    for f in f_samples:
        margin = int(REST_MARGIN_S * TGT_FS)
        start  = max(0, f - margin)
        end    = f
        if end - start >= rest_min:
            rest_ivs.append((start, end))

    return artifact_ivs, rest_ivs


# ============================================================
# 5 折集成推理
# ============================================================

def load_ensemble(
    ckpt_dir: Path,
    cfg: dict,
    device: torch.device,
) -> list[torch.nn.Module]:
    """加载 5 折模型并返回集成列表。

    Args:
        ckpt_dir: checkpoints 根目录。
        cfg:      YAML 配置字典。
        device:   推理设备。

    Returns:
        5 个处于 eval 模式的 MACANet 模型列表。
    """
    import glob
    from src.models.macanet import MACANet

    models: list[torch.nn.Module] = []
    for fold in range(5):
        pattern = str(ckpt_dir / f"fold_{fold}" / "checkpoints" / "best_*.pth")
        files   = sorted(glob.glob(pattern))
        if not files:
            logger.warning("fold_%d: 未找到 checkpoint，跳过", fold)
            continue

        ckpt_path = files[-1]
        model = MACANet.from_config(cfg).to(device)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # fold checkpoint 直接保存 state_dict
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        models.append(model)
        logger.info("fold_%d: 加载 %s", fold, Path(ckpt_path).name)

    if not models:
        raise RuntimeError("未找到任何有效的 fold checkpoint。")
    logger.info("集成：共加载 %d 个模型", len(models))
    return models


@torch.no_grad()
def ensemble_denoise_channel(
    signal_1d: np.ndarray,
    models: list[torch.nn.Module],
    device: torch.device,
) -> np.ndarray:
    """对单通道信号进行 5 折集成去噪。

    Args:
        signal_1d: (T,) 一维信号。
        models:    集成模型列表。
        device:    推理设备。

    Returns:
        (T,) 去噪后信号。
    """
    T = len(signal_1d)

    # 通道级 z-score 归一化（与训练数据保持相同量纲）
    mu  = np.mean(signal_1d)
    std = np.std(signal_1d) + 1e-8
    sig_norm = (signal_1d - mu) / std

    # 汉宁窗重叠加法缓冲区
    out_sum = np.zeros(T, dtype=np.float64)
    win_sum = np.zeros(T, dtype=np.float64)
    window  = np.hanning(SEG_LEN)

    # 所有窗口起始位置（含末尾不完整窗口）
    starts = list(range(0, T, HOP))

    for start in starts:
        end = start + SEG_LEN
        if end > T:
            # 末尾补零（reflect 模式）
            seg = sig_norm[start:]
            pad = SEG_LEN - len(seg)
            seg_padded = np.pad(seg, (0, pad), mode="reflect")
            valid_len  = len(seg)
        else:
            seg_padded = sig_norm[start:end]
            valid_len  = SEG_LEN

        # 转为张量 (1, 1, 512)
        x = torch.from_numpy(seg_padded.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = x.to(device)

        # 集成平均
        preds = [m(x).squeeze().cpu().numpy() for m in models]
        pred  = np.mean(preds, axis=0)   # (512,)

        # 重叠加法
        actual_end = min(start + valid_len, T)
        seg_len    = actual_end - start
        out_sum[start:actual_end] += pred[:seg_len] * window[:seg_len]
        win_sum[start:actual_end] += window[:seg_len]

    # 归一化后反归一化还原量纲
    win_sum = np.maximum(win_sum, 1e-12)
    denoised_norm = out_sum / win_sum
    return denoised_norm * std + mu


def denoise_recording(
    sig: np.ndarray,
    models: list[torch.nn.Module],
    device: torch.device,
) -> np.ndarray:
    """对所有通道逐一去噪。

    Args:
        sig:    原始信号 (T, C)。
        models: 集成模型列表。
        device: 推理设备。

    Returns:
        去噪后信号 (T, C)。
    """
    T, C = sig.shape
    denoised = np.zeros_like(sig)
    for ch in range(C):
        denoised[:, ch] = ensemble_denoise_channel(sig[:, ch], models, device)
    return denoised


# ============================================================
# 无参考评估指标
# ============================================================

def _mad_artifact_rate(signal_1d: np.ndarray) -> float:
    """基于 MAD 的伪影检测率（事件数/秒）。

    Args:
        signal_1d: (T,) 一维信号（10 Hz）。

    Returns:
        伪影率：事件数/秒。
    """
    if len(signal_1d) < 3:
        return 0.0
    med  = np.median(signal_1d)
    mad  = np.median(np.abs(signal_1d - med))
    thr  = MAD_K * mad * MAD_THR_SIGMA
    events = np.sum(np.abs(signal_1d - med) > thr)
    duration_s = len(signal_1d) / TGT_FS
    return float(events / max(duration_s, 1e-6))


def artifact_rate_all_ch(sig: np.ndarray) -> float:
    """所有通道的平均 MAD 伪影检测率。

    Args:
        sig: (T, C) 信号数组。

    Returns:
        全通道平均伪影率（事件/秒）。
    """
    rates = [_mad_artifact_rate(sig[:, ch]) for ch in range(sig.shape[1])]
    return float(np.mean(rates))


def _psd_in_band(
    signal_1d: np.ndarray,
    lo: float,
    hi: float,
    fs: float = TGT_FS,
) -> tuple[np.ndarray, np.ndarray]:
    """计算指定频带内的 Welch PSD，返回 (f, Pxx)。

    Args:
        signal_1d: (T,) 信号。
        lo, hi:    频带下限和上限（Hz）。
        fs:        采样频率。

    Returns:
        (f_band, Pxx_band) 元组。
    """
    from scipy.signal import welch
    nperseg = min(256, len(signal_1d))
    f, Pxx  = welch(signal_1d, fs=fs, nperseg=nperseg)
    mask    = (f >= lo) & (f <= hi)
    return f[mask], Pxx[mask]


def spectral_similarity_to_rest(
    sig_art_chs: np.ndarray,
    sig_rest_chs: np.ndarray,
) -> float:
    """血流动力学频带内伪影区段与安静区段的 PSD 相关性（Pearson r）。

    Args:
        sig_art_chs:  伪影区段信号 (T_art, C)。
        sig_rest_chs: 安静区段信号 (T_rest, C)。

    Returns:
        全通道平均 Pearson r。
    """
    from scipy.stats import pearsonr

    rs: list[float] = []
    for ch in range(sig_art_chs.shape[1]):
        _, p_art  = _psd_in_band(sig_art_chs[:, ch],  HMD_LO, HMD_HI)
        _, p_rest = _psd_in_band(sig_rest_chs[:, ch], HMD_LO, HMD_HI)

        # 取较短长度对齐
        n = min(len(p_art), len(p_rest))
        if n < 3:
            continue
        r, _ = pearsonr(p_art[:n], p_rest[:n])
        if np.isfinite(r):
            rs.append(float(r))

    return float(np.mean(rs)) if rs else float("nan")


def hf_power_ratio(sig: np.ndarray) -> float:
    """高频带（0.5-5 Hz）与血流动力学带（0.01-0.5 Hz）功率比的全通道平均。

    Args:
        sig: (T, C) 信号。

    Returns:
        HF/LF 功率比（全通道均值）。
    """
    ratios: list[float] = []
    for ch in range(sig.shape[1]):
        _, p_hmd = _psd_in_band(sig[:, ch], HMD_LO, HMD_HI)
        _, p_art = _psd_in_band(sig[:, ch], ART_LO,  ART_HI)
        lf = float(np.mean(p_hmd)) + 1e-30
        hf = float(np.mean(p_art))
        ratios.append(hf / lf)
    return float(np.mean(ratios))


# ============================================================
# 单条记录评估
# ============================================================

def evaluate_one_record(
    sig_raw: np.ndarray,
    sig_den: np.ndarray,
    artifact_ivs: list[tuple[int, int]],
    rest_ivs: list[tuple[int, int]],
) -> dict:
    """计算单条记录的无参考评估指标。

    Args:
        sig_raw:      原始信号 (T, C)。
        sig_den:      去噪后信号 (T, C)。
        artifact_ivs: 伪影区段列表 [(start, end), ...]（10 Hz 采样点）。
        rest_ivs:     安静区段列表 [(start, end), ...]（10 Hz 采样点）。

    Returns:
        指标字典。
    """
    T = sig_raw.shape[0]

    # 合并伪影区段为掩码
    art_mask = np.zeros(T, dtype=bool)
    for s, e in artifact_ivs:
        art_mask[s:e] = True

    # 合并安静区段为掩码
    rest_mask = np.zeros(T, dtype=bool)
    for s, e in rest_ivs:
        rest_mask[s:e] = True

    # MAD 伪影率（仅在伪影区段计算）
    if art_mask.sum() > 0:
        art_rate_before = artifact_rate_all_ch(sig_raw[art_mask])
        art_rate_after  = artifact_rate_all_ch(sig_den[art_mask])
    else:
        art_rate_before = artifact_rate_all_ch(sig_raw)
        art_rate_after  = artifact_rate_all_ch(sig_den)
    art_reduction = (
        (art_rate_before - art_rate_after) / max(art_rate_before, 1e-10) * 100
    )

    # 频谱相似度（伪影区段 vs 安静区段）
    if art_mask.sum() > 0 and rest_mask.sum() > TGT_FS * 5:
        sim_before = spectral_similarity_to_rest(
            sig_raw[art_mask], sig_raw[rest_mask]
        )
        sim_after = spectral_similarity_to_rest(
            sig_den[art_mask], sig_raw[rest_mask]
        )
    else:
        sim_before = sim_after = float("nan")

    # 高频功率比（伪影区段）
    if art_mask.sum() > 0:
        hf_ratio_before = hf_power_ratio(sig_raw[art_mask])
        hf_ratio_after  = hf_power_ratio(sig_den[art_mask])
    else:
        hf_ratio_before = hf_power_ratio(sig_raw)
        hf_ratio_after  = hf_power_ratio(sig_den)

    hf_reduction = (
        (hf_ratio_before - hf_ratio_after) / max(hf_ratio_before, 1e-10) * 100
    )

    return {
        "art_rate_before":    art_rate_before,
        "art_rate_after":     art_rate_after,
        "art_rate_reduction": art_reduction,
        "sim_before":         sim_before,
        "sim_after":          sim_after,
        "sim_improvement":    float(sim_after - sim_before) if np.isfinite(sim_before) else float("nan"),
        "hf_ratio_before":    hf_ratio_before,
        "hf_ratio_after":     hf_ratio_after,
        "hf_reduction":       hf_reduction,
    }


# ============================================================
# 可视化
# ============================================================

def _setup_rc() -> None:
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in avail else "serif"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.0, "axes.linewidth": 0.8,
        "grid.linewidth": 0.4, "grid.alpha": 0.4,
    })


def plot_waveform_psd(
    sig_raw: np.ndarray,
    sig_den: np.ndarray,
    artifact_ivs: list[tuple[int, int]],
    rec_id: int,
    ax_wave: "plt.Axes",
    ax_psd:  "plt.Axes",
) -> None:
    """绘制单条记录的波形对比和 PSD 对比（以第 0 通道为代表）。

    Args:
        sig_raw:      原始信号 (T, C)。
        sig_den:      去噪后信号 (T, C)。
        artifact_ivs: 伪影区段列表。
        rec_id:       记录编号。
        ax_wave:      波形绘制目标 Axes。
        ax_psd:       PSD 绘制目标 Axes。
    """
    t = np.arange(sig_raw.shape[0]) / TGT_FS
    ch = 0   # 代表通道

    # 波形对比
    ax_wave.plot(t, sig_raw[:, ch], color="#aaaaaa", alpha=0.6, linewidth=0.6, label="before")
    ax_wave.plot(t, sig_den[:, ch], color="#1f77b4", linewidth=0.8, label="after")
    for s, e in artifact_ivs:
        ax_wave.axvspan(s / TGT_FS, e / TGT_FS, alpha=0.15, color="#d62728")
    ax_wave.set_title(f"fnirs_{rec_id}  ch0  (red=artifact)")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.legend(fontsize=7, loc="upper right")
    ax_wave.grid(True, linestyle="--")
    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)

    # PSD 对比
    from scipy.signal import welch
    nperseg = min(256, sig_raw.shape[0])
    f_b, p_b = welch(sig_raw[:, ch], fs=TGT_FS, nperseg=nperseg)
    f_a, p_a = welch(sig_den[:, ch], fs=TGT_FS, nperseg=nperseg)

    ax_psd.semilogy(f_b, p_b, color="#aaaaaa", alpha=0.7, linewidth=0.7, label="before")
    ax_psd.semilogy(f_a, p_a, color="#1f77b4", linewidth=0.9, label="after")
    ax_psd.axvline(HMD_HI, color="#d62728", linestyle=":", linewidth=0.7, label=f"{HMD_HI} Hz")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title(f"fnirs_{rec_id} PSD")
    ax_psd.legend(fontsize=7, loc="upper right")
    ax_psd.grid(True, linestyle="--")
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)


def plot_physiobank_figure(
    df: pd.DataFrame,
    waveform_cache: dict,
    output_pdf: Path,
) -> None:
    """生成包含代表波形和指标箱线图的汇总图，输出为 PDF。

    布局：
      第 0 行：fnirs_1 波形 | fnirs_1 PSD
      第 1 行：fnirs_5 波形 | fnirs_5 PSD
      第 2 行：伪影率箱线图 (before/after) | 频谱相似度箱线图 (before/after)
      第 3 行：各记录三项改善指标柱状图（跨全宽）

    Args:
        df:             9 条记录的结果 DataFrame。
        waveform_cache: {rec_id: (sig_raw, sig_den, artifact_ivs)} 字典。
        output_pdf:     输出 PDF 路径。
    """
    _setup_rc()

    fig = plt.figure(figsize=(14, 14), dpi=150)
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

    # 代表波形 / PSD（fnirs_1, fnirs_5）
    for row_idx, rep_id in enumerate([1, 5]):
        if rep_id in waveform_cache:
            sig_r, sig_d, art_ivs = waveform_cache[rep_id]
            ax_w = fig.add_subplot(gs[row_idx, 0])
            ax_p = fig.add_subplot(gs[row_idx, 1])
            plot_waveform_psd(sig_r, sig_d, art_ivs, rep_id, ax_w, ax_p)

    # 箱线图：伪影检测率
    ax_art = fig.add_subplot(gs[2, 0])
    bp_data = [df["art_rate_before"].dropna().values,
               df["art_rate_after"].dropna().values]
    bp = ax_art.boxplot(bp_data, tick_labels=["Before", "After"],
                        patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#ffaaaa")
    bp["boxes"][1].set_facecolor("#aaddff")
    ax_art.set_ylabel("Artifact rate (events/s)")
    ax_art.set_title("MAD Artifact Detection Rate")
    ax_art.grid(True, linestyle="--", axis="y")
    ax_art.spines["top"].set_visible(False)
    ax_art.spines["right"].set_visible(False)

    # 箱线图：频谱相似度
    ax_sim = fig.add_subplot(gs[2, 1])
    sim_data = [
        df["sim_before"].dropna().values,
        df["sim_after"].dropna().values,
    ]
    if all(len(d) > 0 for d in sim_data):
        bp2 = ax_sim.boxplot(sim_data, tick_labels=["Before", "After"],
                             patch_artist=True, widths=0.5)
        bp2["boxes"][0].set_facecolor("#ffaaaa")
        bp2["boxes"][1].set_facecolor("#aaddff")
    ax_sim.set_ylabel("Spectral similarity (Pearson r)")
    ax_sim.set_title("Spectral Similarity to Rest Segment")
    ax_sim.grid(True, linestyle="--", axis="y")
    ax_sim.spines["top"].set_visible(False)
    ax_sim.spines["right"].set_visible(False)

    # 柱状图：三项改善指标（跨全宽）
    ax_sum = fig.add_subplot(gs[3, :])
    x_labels = [f"fnirs_{i}" for i in df["rec_id"].tolist()]
    x = np.arange(len(df))
    w = 0.25

    for offset, col, label, color in [
        (-w, "art_rate_reduction", "Artifact rate reduction (%)", "#d62728"),
        ( 0, "sim_improvement",    "Spectral sim. improvement",   "#2ca02c"),
        (+w, "hf_reduction",       "HF power reduction (%)",      "#1f77b4"),
    ]:
        vals = df[col].values
        ax_sum.bar(x + offset, vals, width=w * 0.9, label=label, color=color, alpha=0.75)

    ax_sum.axhline(0, color="#333333", linewidth=0.8)
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(x_labels, rotation=30, ha="right")
    ax_sum.set_ylabel("Improvement")
    ax_sum.set_title("Cross-Dataset Improvement per Recording (PhysioBank)")
    ax_sum.set_ylim(-100, 50)
    ax_sum.legend(fontsize=7, loc="upper right")
    ax_sum.grid(True, linestyle="--", axis="y")
    ax_sum.spines["top"].set_visible(False)
    ax_sum.spines["right"].set_visible(False)

    fig.suptitle(
        "Figure 8 — MA-CANet Cross-Dataset Generalization on PhysioBank fNIRS",
        fontsize=11, y=1.01,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("可视化图已保存：%s", output_pdf)


# ============================================================
# 结果汇总打印
# ============================================================

def _print_summary(df: pd.DataFrame) -> None:
    """在控制台打印 Table III 汇总表。"""
    print("\n" + "=" * 110)
    print("  TABLE III — PhysioBank Cross-Dataset Evaluation (MA-CANet 5-fold ensemble)")
    print("=" * 110)
    hdr = (
        f"  {'rec':>7}  {'art_rate_b':>11}  {'art_rate_a':>11}  {'art_reduc%':>11}"
        f"  {'sim_b':>8}  {'sim_a':>8}  {'sim_imp':>8}"
        f"  {'hf_b':>8}  {'hf_a':>8}  {'hf_red%':>9}"
    )
    print(hdr)
    print("  " + "-" * 106)

    for _, row in df.iterrows():
        def _f(v: float, fmt: str = ".4f") -> str:
            return f"{v:{fmt}}" if np.isfinite(v) else "  nan  "

        print(
            f"  {'fnirs_' + str(int(row['rec_id'])):>7}  "
            f"{_f(row['art_rate_before']):>11}  "
            f"{_f(row['art_rate_after']):>11}  "
            f"{_f(row['art_rate_reduction'], '.2f'):>11}  "
            f"{_f(row['sim_before']):>8}  "
            f"{_f(row['sim_after']):>8}  "
            f"{_f(row['sim_improvement']):>8}  "
            f"{_f(row['hf_ratio_before']):>8}  "
            f"{_f(row['hf_ratio_after']):>8}  "
            f"{_f(row['hf_reduction'], '.2f'):>9}"
        )

    print("  " + "-" * 106)

    def _agg(col: str, func: str) -> str:
        v = getattr(df[col].dropna(), func)()
        return f"{v:.4f}" if np.isfinite(v) else "  nan  "

    for label, func in [("mean", "mean"), ("median", "median")]:
        print(
            f"  {label:>7}  "
            f"{_agg('art_rate_before', func):>11}  "
            f"{_agg('art_rate_after', func):>11}  "
            f"{_agg('art_rate_reduction', func):>11}  "
            f"{_agg('sim_before', func):>8}  "
            f"{_agg('sim_after', func):>8}  "
            f"{_agg('sim_improvement', func):>8}  "
            f"{_agg('hf_ratio_before', func):>8}  "
            f"{_agg('hf_ratio_after', func):>8}  "
            f"{_agg('hf_reduction', func):>9}"
        )

    print("=" * 110 + "\n")


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhysioBank fNIRS 评估（跨数据集泛化验证）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=Path("data/raw/motion-artifact-contaminated-fnirs-and-eeg-data-1.0.0"),
        help="PhysioBank 数据目录",
    )
    p.add_argument("--config",     type=Path, default=Path("configs/default.yaml"),
                   help="YAML 配置文件路径")
    p.add_argument("--ckpt-dir",   type=Path, default=Path("outputs/checkpoints"),
                   help="checkpoints 根目录")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="结果输出目录")
    p.add_argument("--device",     type=str,  default="auto",
                   help="推理设备（auto/cpu/cuda）")
    p.add_argument(
        "--records", type=int, nargs="+", default=list(range(1, 10)),
        help="评估的记录编号（默认：1~9）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    logger.info("推理设备：%s", device)

    # 加载 5 折集成模型
    models = load_ensemble(args.ckpt_dir, cfg, device)

    records: list[dict]  = []
    waveform_cache: dict = {}

    for rec_id in args.records:
        logger.info("─── 处理 fnirs_%d ─────────────────────────────────────", rec_id)

        # 读取数据
        sig_raw, annotations, _ = load_fnirs_record(args.data_dir, rec_id)
        T, C = sig_raw.shape

        # 解析伪影区段和安静区段
        artifact_ivs, rest_ivs = parse_intervals(annotations, T)
        logger.info(
            "fnirs_%d: 伪影区段 %d 个，安静区段 %d 个",
            rec_id, len(artifact_ivs), len(rest_ivs),
        )

        # 集成去噪
        sig_den = denoise_recording(sig_raw, models, device)

        # 计算指标
        metrics = evaluate_one_record(sig_raw, sig_den, artifact_ivs, rest_ivs)

        logger.info(
            "fnirs_%d  art_rate: %.4f->%.4f (reduction %.1f%%)  "
            "sim: %.4f->%.4f (+%.4f)  hf_ratio: %.4f->%.4f (reduction %.1f%%)",
            rec_id,
            metrics["art_rate_before"], metrics["art_rate_after"],
            metrics["art_rate_reduction"],
            metrics["sim_before"],      metrics["sim_after"],
            metrics["sim_improvement"],
            metrics["hf_ratio_before"], metrics["hf_ratio_after"],
            metrics["hf_reduction"],
        )

        records.append({"rec_id": rec_id, **metrics})

        # 缓存代表性记录的波形（用于可视化）
        if rec_id in (1, 5):
            waveform_cache[rec_id] = (sig_raw, sig_den, artifact_ivs)

    # 保存结果
    df = pd.DataFrame(records)
    csv_path = args.output_dir / "Table_III_physiobank.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Table III 已保存：%s", csv_path)

    _print_summary(df)

    # 生成可视化图
    pdf_path = args.output_dir / "figures" / "Figure_physiobank.pdf"
    plot_physiobank_figure(df, waveform_cache, pdf_path)

    logger.info("PhysioBank 评估完成。")


if __name__ == "__main__":
    main()
