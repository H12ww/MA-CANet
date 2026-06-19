"""在真实 BIDS fNIRS 数据上验证 MA-CANet 的泛化性能。

由于真实数据无 ground truth，采用三类无参考指标：
  1. **频谱相似度**：去噪后 SM/LM 与同被试 HT 段（无伪影参考）的
     Welch PSD Pearson 相关系数（0.01–0.5 Hz）
  2. **残余伪影检测率**：MAD-based 检测去噪前后每秒检测到的伪影事件数
  3. **视觉对比**：5 个典型样本（3 SM + 2 LM）的去噪前后通道波形图

输出：
  outputs/real_bids_results.csv    — 逐被试逐条件指标
  outputs/figures/Figure_6_waveforms.pdf  — 5 典型波形对比图
  outputs/figures/Figure_6_analysis.pdf  — PSD + 残余伪影分析图

用法::

    python scripts/07_eval_on_real_bids.py \\
        [--processed-dir data/processed] \\
        [--ckpt-dir outputs/checkpoints] \\
        [--folds 0 1 2 3 4] \\
        [--subjects all] \\
        [--output-dir outputs] \\
        [--device auto]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 防止 Windows 多 OpenMP 库冲突

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FS               = 10.0   # Hz
SEG_LEN          = 100    # 真实数据每段长度（采样点）
MODEL_LEN        = 512    # 模型训练时的输入长度
N_CH             = 16     # 通道数
ALL_SUBJECTS     = [f"sub-{i:02d}" for i in range(1, 21)]
_OUTLIER_SUBJECTS = {"sub-19", "sub-20"}   # 基线 SNR 过低，去噪后频谱相似度下降


# ===========================================================================
# 数据加载
# ===========================================================================

def load_subject_segments(
    processed_dir: Path,
    subject: str,
    condition: str,
) -> Optional[np.ndarray]:
    """加载单个被试的某一条件所有试次。

    Args:
        processed_dir: processed/ 根目录。
        subject:       被试 ID，如 'sub-01'。
        condition:     'HT', 'SM', 'LM' 之一。

    Returns:
        float32 数组 (n_trials, 16, L)，或 None（文件不存在时）。
    """
    agg_path = processed_dir / subject / f"{subject}_{condition}.npy"
    if agg_path.exists():
        return np.load(agg_path).astype(np.float32)   # (N, 16, L)

    # 回退：逐文件加载
    pattern = str(processed_dir / subject / f"{subject}_{condition}_*.npy")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        logger.warning("%s/%s: 未找到数据文件，跳过。", subject, condition)
        return None

    trials = [np.load(p).astype(np.float32) for p in paths]
    return np.stack(trials, axis=0)   # (N, 16, L)


# ===========================================================================
# 模型推理
# ===========================================================================

def _load_ensemble(
    ckpt_dir: Path,
    folds: list[int],
    cfg: dict,
    device: torch.device,
) -> list[torch.nn.Module]:
    """加载各折最优 checkpoint，返回模型列表。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.macanet import MACANet

    models = []
    for fold in folds:
        pattern = str(ckpt_dir / f"fold_{fold}" / "checkpoints" / "best_*.pth")
        paths   = sorted(glob.glob(pattern))
        if not paths:
            logger.warning("fold_%d: 未找到 checkpoint，跳过。", fold)
            continue
        m = MACANet.from_config(cfg)
        state = torch.load(Path(paths[-1]), map_location="cpu", weights_only=False)
        m.load_state_dict(state.get("model_state_dict", state))
        m.to(device).eval()
        models.append(m)
        logger.info("  fold_%d: %s", fold, Path(paths[-1]).name)

    if not models:
        logger.error("没有可用的 MA-CANet checkpoint，请先训练。")
        sys.exit(1)
    return models


@torch.no_grad()
def infer_segments(
    models: list[torch.nn.Module],
    segments: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """对短段（< MODEL_LEN 的任意长度）做 pad → infer → crop 集成推理。

    短段用反射 padding 补齐到 MODEL_LEN，推理后裁回原长，以保持边缘一致性。

    Args:
        models:   模型列表（集成）。
        segments: (N, 16, L) float32。
        device:   推理设备。

    Returns:
        (N, 16, L) 去噪结果。
    """
    N, C, L = segments.shape
    out = np.empty_like(segments)

    need_pad = L < MODEL_LEN
    pad_left = (MODEL_LEN - L) // 2 if need_pad else 0
    pad_right = MODEL_LEN - L - pad_left if need_pad else 0

    for i in range(N):
        seg = segments[i].astype(np.float32)   # (16, L)

        if need_pad:
            seg_p = np.pad(seg, [(0, 0), (pad_left, pad_right)], mode="reflect")
        else:
            seg_p = seg

        preds = np.zeros_like(seg_p)
        for ch in range(C):
            x = torch.from_numpy(seg_p[ch:ch+1, :][None]).to(device)   # (1,1,L_pad)
            y = sum(m(x) for m in models) / len(models)
            preds[ch] = y.squeeze().cpu().numpy()

        out[i] = preds[:, pad_left:pad_left + L] if need_pad else preds

    return out


# ===========================================================================
# 指标 1 — 频谱相似度
# ===========================================================================

def _compute_psd_welch(
    segments: np.ndarray,
    fs: float = FS,
    nperseg: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """拼接所有试次后用 Welch 法计算平均 PSD。

    Args:
        segments: (N, C, L)。
        fs:       采样率。
        nperseg:  Welch 窗长；分辨率 = fs/nperseg。

    Returns:
        freqs (F,), psd_mean (C, F) — 各通道平均 PSD。
    """
    from scipy.signal import welch

    N, C, L = segments.shape
    # 拼接所有试次形成长时间序列（试次间用 0 间隔，避免拼接边缘影响）
    concat = np.concatenate(
        [segments[:, ch, :].reshape(-1) for ch in range(C)],
        axis=0,
    ).reshape(C, -1)   # (C, N*L)

    # 对每个通道计算 Welch PSD
    freqs, _ = welch(concat[0], fs=fs, nperseg=min(nperseg, N * L))
    psds = []
    for ch in range(C):
        f, psd = welch(concat[ch], fs=fs, nperseg=min(nperseg, N * L))
        psds.append(psd)

    return freqs, np.stack(psds, axis=0)   # (C, F)


def spectral_similarity(
    noisy_segs:    np.ndarray,
    denoised_segs: np.ndarray,
    ht_segs:       np.ndarray,
    fs:            float = FS,
    freq_lo:       float = 0.01,
    freq_hi:       float = 0.5,
    nperseg:       int   = 256,
) -> dict[str, float]:
    """计算去噪前后与 HT 参考的频谱相似度。

    Args:
        noisy_segs:    (N, C, L) 含伪影段。
        denoised_segs: (N, C, L) 去噪后段。
        ht_segs:       (M, C, L) 无伪影 HT 参考段。

    Returns:
        dict 含 'sim_before', 'sim_after', 'sim_improvement'（均为通道均值 Pearson r）。
    """
    freqs, psd_noisy    = _compute_psd_welch(noisy_segs,    fs, nperseg)
    freqs, psd_denoised = _compute_psd_welch(denoised_segs, fs, nperseg)
    freqs, psd_ht       = _compute_psd_welch(ht_segs,       fs, nperseg)

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if mask.sum() == 0:
        return {"sim_before": float("nan"), "sim_after": float("nan"),
                "sim_improvement": float("nan")}

    from scipy.stats import pearsonr

    sim_before_list, sim_after_list = [], []
    for ch in range(psd_noisy.shape[0]):
        a_noisy    = np.log1p(psd_noisy[ch][mask])
        a_denoised = np.log1p(psd_denoised[ch][mask])
        a_ht       = np.log1p(psd_ht[ch][mask])

        if np.std(a_ht) < 1e-8 or np.std(a_noisy) < 1e-8 or np.std(a_denoised) < 1e-8:
            continue
        r_before, _ = pearsonr(a_noisy,    a_ht)
        r_after,  _ = pearsonr(a_denoised, a_ht)
        if np.isfinite(r_before):
            sim_before_list.append(r_before)
        if np.isfinite(r_after):
            sim_after_list.append(r_after)

    sb = float(np.mean(sim_before_list)) if sim_before_list else float("nan")
    sa = float(np.mean(sim_after_list))  if sim_after_list  else float("nan")
    return {
        "sim_before":      sb,
        "sim_after":       sa,
        "sim_improvement": sa - sb,
    }


# ===========================================================================
# 指标 2 — 残余伪影检测率
# ===========================================================================

def _mad_artifact_rate(
    segments:  np.ndarray,
    threshold: float = 3.0,
    fs:        float = FS,
) -> float:
    """用 MAD-based 检测器计算每秒伪影事件数（跨通道平均）。

    Args:
        segments:  (N, C, L)。
        threshold: MAD 阈值倍数。

    Returns:
        每秒平均伪影点数（标量）。
    """
    total_art = 0
    total_pts = 0

    for seg in segments:              # seg: (C, L)
        diff = np.diff(seg, axis=1, prepend=seg[:, :1])
        med  = np.median(diff, axis=1, keepdims=True)
        mad  = np.median(np.abs(diff - med), axis=1, keepdims=True)
        sigma_hat = 1.4826 * mad
        artifact_mask = np.abs(diff - med) > threshold * (sigma_hat + 1e-10)
        total_art += artifact_mask.sum()
        total_pts += artifact_mask.size

    duration_s = segments.shape[0] * segments.shape[2] / fs
    return total_art / (duration_s * N_CH + 1e-10)


def artifact_reduction(
    noisy_segs:    np.ndarray,
    denoised_segs: np.ndarray,
    fs:            float = FS,
    threshold:     float = 3.0,
) -> dict[str, float]:
    """计算去噪前后的残余伪影检测率及降低比。"""
    rate_before = _mad_artifact_rate(noisy_segs,    threshold, fs)
    rate_after  = _mad_artifact_rate(denoised_segs, threshold, fs)
    reduction_pct = (
        100.0 * (rate_before - rate_after) / (rate_before + 1e-10)
    )
    return {
        "art_rate_before":    rate_before,
        "art_rate_after":     rate_after,
        "art_reduction_pct":  reduction_pct,
    }


# ===========================================================================
# Figure 6a — 典型波形对比（5 个样本）
# ===========================================================================

def _setup_paper_rc() -> None:
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = (
        "Times New Roman" if "Times New Roman" in available else "serif"
    )
    plt.rcParams.update({
        "font.size":        9,
        "axes.titlesize":  10,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "lines.linewidth":  1.0,
        "axes.linewidth":   0.7,
        "grid.linewidth":   0.4,
        "grid.alpha":       0.4,
    })


def _select_typical_samples(
    noisy_segs: np.ndarray,
    n: int = 5,
) -> list[int]:
    """按伪影强度（std 降序）选出 n 个最典型的含伪影样本。"""
    scores = [noisy_segs[i].std() for i in range(len(noisy_segs))]
    return list(np.argsort(scores)[::-1][:n])


def plot_waveform_comparison(
    subject_data: list[dict],
    output_pdf: Path,
    n_samples: int = 5,
    channel: int = 0,
    target_subjects: list[str] | None = None,
) -> None:
    """Figure 7：典型样本的 SM + LM 去噪前后波形对比。

    每行 = 1 个样本，左列 = SM，右列 = LM。
    颜色：噪声信号（红/橙）, 去噪后（蓝/绿）。

    Args:
        subject_data:     列表，每项为 {subject, condition,
                          noisy (N,C,L), denoised (N,C,L), sample_idx, channel}。
        output_pdf:       输出路径。
        n_samples:        并列行数（target_subjects 指定时忽略）。
        channel:          展示的通道编号（HbO #0）。
        target_subjects:  若指定，则只显示该列表中的被试，顺序与列表一致。
    """
    _setup_paper_rc()

    sm_entries = [d for d in subject_data if d["condition"] == "SM"]
    lm_entries = [d for d in subject_data if d["condition"] == "LM"]

    if not sm_entries or not lm_entries:
        logger.warning("SM 或 LM 数据不足，跳过波形图。")
        return

    if target_subjects is not None:
        sm_map = {d["subject"]: d for d in sm_entries}
        lm_map = {d["subject"]: d for d in lm_entries}
        sm_picks = [sm_map[s] for s in target_subjects if s in sm_map]
        lm_picks = [lm_map[s] for s in target_subjects if s in lm_map]
        if not sm_picks or not lm_picks:
            logger.warning("target_subjects 中没有匹配的 SM/LM 数据，回退到默认选取。")
            sm_picks = sm_entries[:n_samples]
            lm_picks = lm_entries[:n_samples]
    else:
        sm_picks = sm_entries[:n_samples]
        lm_picks = lm_entries[:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, n_samples * 2.0), dpi=300)
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    t = np.arange(SEG_LEN) / FS

    for row in range(n_samples):
        for col, (picks, cond_label) in enumerate([(sm_picks, "SM"), (lm_picks, "LM")]):
            ax = axes[row, col]
            if row >= len(picks):
                ax.set_visible(False)
                continue

            entry    = picks[row]
            idx      = entry["sample_idx"]
            noisy_ch = entry["noisy"][idx, channel, :] if entry["noisy"].ndim == 3 else entry["noisy"][channel, :]
            den_ch   = entry["denoised"][idx, channel, :] if entry["denoised"].ndim == 3 else entry["denoised"][channel, :]

            ax.plot(t, noisy_ch, color="#d62728", linewidth=0.9, alpha=0.85,
                    label="Noisy" if row == 0 else "_nolegend_")
            ax.plot(t, den_ch, color="#1f77b4", linewidth=1.1,
                    label="Denoised" if row == 0 else "_nolegend_")

            ax.set_xlim(0, t[-1])
            ax.grid(True, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(
                    f"{cond_label} — {entry['subject']} ch{channel}",
                    fontweight="bold",
                )
                ax.legend(loc="upper right", framealpha=0.8)
            else:
                ax.set_title(f"{cond_label} — {entry['subject']} ch{channel}")

            if row == n_samples - 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("z-score")

    fig.suptitle(
        "Figure 7 — MA-CANet Denoising on Real BIDS Data (Typical Samples)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Figure 7 已保存：%s", output_pdf)


# ===========================================================================
# Figure 6b — PSD + 伪影率分析图
# ===========================================================================

def plot_analysis_figure(
    results_df: pd.DataFrame,
    psd_cache:  dict,
    output_pdf: Path,
) -> None:
    """Figure 6b：2×2 子图，汇总频谱和伪影率分析。

    子图布局：
      (0,0) SM PSD 对比  (0,1) LM PSD 对比
      (1,0) 频谱相似度箱线图  (1,1) 伪影率降低条形图
    """
    _setup_paper_rc()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

    # ── PSD 对比（SM/LM，均值 ± std 带） ────────────────────────────────────
    for col, cond in enumerate(["SM", "LM"]):
        ax = axes[0, col]
        if cond not in psd_cache:
            ax.set_visible(False)
            continue

        freqs  = psd_cache[cond]["freqs"]
        mask   = (freqs >= 0.01) & (freqs <= 0.5)
        f_plot = freqs[mask]

        for label, key, color, lw in [
            ("Noisy",    "psd_noisy",    "#d62728", 1.0),
            ("Denoised", "psd_denoised", "#1f77b4", 1.2),
            ("HT Ref",   "psd_ht",       "#2ca02c", 1.0),
        ]:
            if key not in psd_cache[cond]:
                continue
            # 各被试 PSD 取 log
            all_ch_psds = np.array(psd_cache[cond][key])   # (subjects*C, F)
            mean_log = np.log1p(all_ch_psds[:, mask]).mean(axis=0)
            std_log  = np.log1p(all_ch_psds[:, mask]).std(axis=0)
            ax.plot(f_plot, mean_log, color=color, linewidth=lw, label=label)
            ax.fill_between(
                f_plot, mean_log - std_log, mean_log + std_log,
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log(1 + PSD)")
        ax.set_title(f"{cond} — PSD Comparison")
        ax.legend(framealpha=0.8)
        ax.grid(True, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── 频谱相似度箱线图（保留所有被试，离群值用红色 × 标出）────────────────
    ax_sim = axes[1, 0]
    sim_data = {}
    for cond in ["SM", "LM"]:
        sub_df = results_df[results_df["condition"] == cond]
        sim_data[f"{cond}\nbefore"] = sub_df["sim_before"].dropna().values
        sim_data[f"{cond}\nafter"]  = sub_df["sim_after"].dropna().values

    box_labels = list(sim_data.keys())
    data       = [sim_data[k] for k in box_labels]
    colors     = ["#fca582", "#d62728", "#82b4fc", "#1f77b4"]

    bp = ax_sim.boxplot(
        data, tick_labels=box_labels, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
        # 离群点用红色 × 显示，与普通离群点区分
        flierprops=dict(marker="x", markersize=5, alpha=0.7,
                        markerfacecolor="#d62728", markeredgecolor="#d62728"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # 在各列上叠加散点，用特殊符号标记离群被试
    for col_idx, (col_label, cond, metric) in enumerate([
        ("SM\nbefore", "SM", "sim_before"),
        ("SM\nafter",  "SM", "sim_after"),
        ("LM\nbefore", "LM", "sim_before"),
        ("LM\nafter",  "LM", "sim_after"),
    ], start=1):
        sub_df = results_df[results_df["condition"] == cond]
        out_df = sub_df[sub_df["subject"].isin(_OUTLIER_SUBJECTS)]
        if not out_df.empty:
            ax_sim.scatter(
                [col_idx] * len(out_df),
                out_df[metric].values,
                marker="*", s=80, color="#9467bd", zorder=5,
                label="outlier" if col_idx == 1 else "_nolegend_",
            )

    ax_sim.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax_sim.set_ylabel("Pearson r (0.01–0.5 Hz)")
    ax_sim.set_title("Spectral Similarity with HT Reference")
    ax_sim.grid(axis="y", linestyle="--")
    ax_sim.spines["top"].set_visible(False)
    ax_sim.spines["right"].set_visible(False)

    # 离群被试标注文字
    outlier_note = (
        f"* {', '.join(sorted(_OUTLIER_SUBJECTS))} are outliers\n"
        "  (low baseline SNR; denoising less effective)"
    )
    ax_sim.text(
        0.02, 0.04, outlier_note,
        transform=ax_sim.transAxes,
        fontsize=7, color="#9467bd",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#9467bd", alpha=0.8, linewidth=0.7),
    )
    ax_sim.legend(
        handles=[plt.Line2D([0], [0], marker="*", color="w",
                            markerfacecolor="#9467bd", markersize=8,
                            label="Outlier (sub-19/20)")],
        loc="upper right", fontsize=7, framealpha=0.8,
    )

    # ── 伪影率降低条形图（排除离群被试，使用中位数更鲁棒）──────────────────
    ax_art = axes[1, 1]
    df_in  = results_df[~results_df["subject"].isin(_OUTLIER_SUBJECTS)]
    cond_labels, meds_before, meds_after, iqr_before, iqr_after = [], [], [], [], []

    for cond in ["SM", "LM"]:
        sub_df = df_in[df_in["condition"] == cond]
        cond_labels.append(cond)
        meds_before.append(sub_df["art_rate_before"].median())
        meds_after.append( sub_df["art_rate_after"].median())
        # IQR / 2 作为误差棒（比 std 更鲁棒）
        iqr_before.append((sub_df["art_rate_before"].quantile(0.75)
                          - sub_df["art_rate_before"].quantile(0.25)) / 2)
        iqr_after.append( (sub_df["art_rate_after"].quantile(0.75)
                          - sub_df["art_rate_after"].quantile(0.25)) / 2)

    x     = np.arange(len(cond_labels))
    width = 0.35
    ax_art.bar(x - width/2, meds_before, width, yerr=iqr_before,
               label="Before", color="#d62728", alpha=0.75,
               error_kw=dict(elinewidth=0.8, capsize=3))
    ax_art.bar(x + width/2, meds_after,  width, yerr=iqr_after,
               label="After",  color="#1f77b4", alpha=0.75,
               error_kw=dict(elinewidth=0.8, capsize=3))

    # 标注中位数降低百分比
    for i, cond in enumerate(cond_labels):
        sub_df = df_in[df_in["condition"] == cond]
        pct = sub_df["art_reduction_pct"].median()
        sign = "↓" if pct >= 0 else "↑"
        ax_art.text(
            i, max(meds_before[i], meds_after[i]) * 1.08,
            f"{sign}{abs(pct):.1f}%", ha="center", va="bottom", fontsize=8,
        )

    ax_art.set_xticks(x)
    ax_art.set_xticklabels(cond_labels)
    ax_art.set_ylabel("Artifact events / s / channel")
    ax_art.set_title(
        "Residual Artifact Rate — median ± IQR/2\n"
        f"(outliers excluded: {', '.join(sorted(_OUTLIER_SUBJECTS))})"
    )
    ax_art.legend(framealpha=0.8)
    ax_art.grid(axis="y", linestyle="--")
    ax_art.spines["top"].set_visible(False)
    ax_art.spines["right"].set_visible(False)

    fig.suptitle(
        "Figure 6b — Spectral Analysis & Artifact Reduction (Real BIDS Data)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Figure 6b 已保存：%s", output_pdf)


# ===========================================================================
# 主逻辑
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在真实 BIDS fNIRS 数据上评估 MA-CANet 泛化性能",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path,
                   default=Path("data/processed"))
    p.add_argument("--config",         type=Path,
                   default=Path("configs/default.yaml"))
    p.add_argument("--ckpt-dir",       type=Path,
                   default=Path("outputs/checkpoints"))
    p.add_argument("--folds",          type=int, nargs="+",
                   default=list(range(5)))
    p.add_argument("--subjects",       type=str, nargs="+",
                   default=["all"],
                   help="被试列表，或 'all' 表示全部 20 名被试。")
    p.add_argument("--conditions",     type=str, nargs="+",
                   default=["SM", "LM"],
                   help="评估的运动条件。")
    p.add_argument("--output-dir",     type=Path,
                   default=Path("outputs"))
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--n-waveform-samples", type=int, default=5,
                   help="波形对比图中展示的样本数。")
    return p.parse_args()


def _print_summary(df: pd.DataFrame) -> None:
    """打印三层统计汇总：全量 mean、中位数、排除离群被试后 mean。"""
    n_sub    = df["subject"].nunique()
    outliers = sorted(_OUTLIER_SUBJECTS & set(df["subject"].unique()))
    n_out    = len(outliers)

    print("\n" + "=" * 88)
    print(f"  REAL BIDS EVALUATION — 逐条件汇总（共 {n_sub} 被试）")
    if outliers:
        print(f"  离群被试：{', '.join(outliers)}（基线 SNR 过低，已单独标注）")
    print("=" * 88)

    hdr = (f"  {'Cond':>4}  {'统计':>10}  {'sim_before':>10}  {'sim_after':>10}  "
           f"{'Δsim':>8}  {'art_reduct%':>11}")
    print(hdr)
    print("  " + "-" * 84)

    for cond, g in df.groupby("condition"):
        g_in = g[~g["subject"].isin(_OUTLIER_SUBJECTS)]

        def _row(label: str, subset: pd.DataFrame, func: str) -> str:
            agg = subset[["sim_before", "sim_after", "sim_improvement",
                           "art_reduction_pct"]].agg(func)
            return (
                f"  {cond if label == 'mean(all)' else '':>4}  "
                f"{label:>10}  "
                f"{agg['sim_before']:>+10.4f}  "
                f"{agg['sim_after']:>+10.4f}  "
                f"{agg['sim_improvement']:>+8.4f}  "
                f"{agg['art_reduction_pct']:>+10.1f}%"
            )

        print(_row("mean(all)",  g,    "mean"))
        print(_row("median",     g,    "median"))
        if n_out > 0:
            label_ex = f"mean(-{n_out})"
            print(_row(label_ex, g_in, "mean"))
        print()

    print("=" * 88 + "\n")


def main() -> None:
    args = parse_args()
    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("推理设备：%s", device)

    # 被试列表
    subjects = ALL_SUBJECTS if "all" in args.subjects else args.subjects

    # 加载模型
    logger.info("加载 MA-CANet 集成（%d 折）…", len(args.folds))
    models = _load_ensemble(args.ckpt_dir, args.folds, cfg, device)

    # 逐被试 × 逐条件评估
    records:    list[dict] = []
    wave_data:  list[dict] = []
    psd_cache:  dict       = {}   # cond → {freqs, psd_noisy, psd_denoised, psd_ht}

    logger.info("开始逐被试评估（%d 被试 × %d 条件）…",
                len(subjects), len(args.conditions))

    for sub in subjects:
        ht_segs = load_subject_segments(args.processed_dir, sub, "HT")
        if ht_segs is None:
            logger.warning("%s: HT 数据不存在，跳过。", sub)
            continue

        for cond in args.conditions:
            segs = load_subject_segments(args.processed_dir, sub, cond)
            if segs is None:
                continue

            # 模型推理
            denoised = infer_segments(models, segs, device)

            # 指标 1：频谱相似度
            seg_l = segs.shape[2]   # 实际段长（真实数据为 100，非训练用的 SEG_LEN=512）
            spec = spectral_similarity(
                segs, denoised, ht_segs,
                fs=FS, freq_lo=0.01, freq_hi=0.5,
                nperseg=min(256, segs.shape[0] * seg_l),
            )

            # 指标 2：残余伪影率
            art = artifact_reduction(segs, denoised, fs=FS)

            records.append({
                "subject":    sub,
                "condition":  cond,
                "is_outlier": sub in _OUTLIER_SUBJECTS,
                **spec,
                **art,
            })

            # 为波形图缓存代表性样本
            idxs = _select_typical_samples(segs, n=1)
            wave_data.append({
                "subject":   sub,
                "condition": cond,
                "noisy":     segs,
                "denoised":  denoised,
                "sample_idx": idxs[0],
            })

            # 缓存 PSD（用于 Figure 6b）
            ht_l = ht_segs.shape[2]
            _, psd_n = _compute_psd_welch(segs,     FS, min(256, segs.shape[0] * seg_l))
            _, psd_d = _compute_psd_welch(denoised, FS, min(256, segs.shape[0] * seg_l))
            f_ht, psd_h = _compute_psd_welch(ht_segs, FS, min(256, ht_segs.shape[0] * ht_l))

            if cond not in psd_cache:
                psd_cache[cond] = {
                    "freqs":       f_ht,
                    "psd_noisy":   [],
                    "psd_denoised": [],
                    "psd_ht":      [],
                }
            psd_cache[cond]["psd_noisy"].extend(psd_n.tolist())
            psd_cache[cond]["psd_denoised"].extend(psd_d.tolist())
            psd_cache[cond]["psd_ht"].extend(psd_h.tolist())

            logger.info("  %s/%s  sim_before=%.3f → sim_after=%.3f  "
                        "art_reduction=%.1f%%",
                        sub, cond,
                        spec["sim_before"], spec["sim_after"],
                        art["art_reduction_pct"])

    if not records:
        logger.error("没有可用数据，请检查 --processed-dir 路径。")
        sys.exit(1)

    results_df = pd.DataFrame(records)

    # 保存 CSV
    csv_path = args.output_dir / "real_bids_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    logger.info("结果已保存：%s  (%d 行)", csv_path, len(results_df))

    # 控制台汇总
    _print_summary(results_df)

    # Figure 7 — 波形对比（固定显示 sub-01/19/20）
    plot_waveform_comparison(
        wave_data,
        output_pdf=args.output_dir / "figures" / "Figure_6_waveforms.pdf",
        n_samples=3,
        channel=0,
        target_subjects=["sub-01", "sub-19", "sub-20"],
    )

    # Figure 6b — PSD + 伪影率分析
    plot_analysis_figure(
        results_df,
        psd_cache,
        output_pdf=args.output_dir / "figures" / "Figure_6_analysis.pdf",
    )

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
