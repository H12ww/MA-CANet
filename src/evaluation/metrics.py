"""fNIRS 伪影去除评估指标：ΔSNR、RMSE、Pearson r、SSIM、η。

所有函数的信号格式：(n_channels, n_timepoints) 的 float32/float64 numpy 数组。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ===========================================================================
# 数值常量
# ===========================================================================

_VAR_EPS  = 1e-10   # 方差下界：防止 log10(0) 或除零
_DSNR_MIN = -60.0   # ΔSNR 下界（dB）
_DSNR_MAX = 120.0   # ΔSNR 上界（dB）


# ===========================================================================
# 基础工具
# ===========================================================================

def _snr_db_safe(var_signal: float, var_noise: float) -> float:
    """数值安全的单值 SNR（dB）计算。

    两个方差都用 _VAR_EPS 截断，消除：
    - 除零（var_noise ≈ 0）
    - log10(≈0) 溢出（var_signal ≈ 0）
    - inf 返回（不再出现，改为有限 dB 值）
    """
    var_signal = max(float(var_signal), _VAR_EPS)
    var_noise  = max(float(var_noise),  _VAR_EPS)
    return 10.0 * np.log10(var_signal / var_noise)


# ===========================================================================
# 五大评估指标
# ===========================================================================

def delta_snr(
    noisy:    np.ndarray,
    denoised: np.ndarray,
    clean:    np.ndarray,
) -> float:
    """逐行（逐通道/逐样本）ΔSNR 均值，越高越好。

    ΔSNR = SNR_after − SNR_before（dB）
    SNR  = 10·log10( var(clean) / var(residual) )

    数值安全处理：
    - 方差用 _VAR_EPS 截断，消除除零和 log10(0)
    - 最终 ΔSNR 按行计算后 clamp 到 [_DSNR_MIN, _DSNR_MAX]，再取均值
    - 若某行 var(noisy−clean) < 1e-8（意外出现的无伪影行），记录 warning 并仍纳入
      均值（此时 SNR_before 极高，ΔSNR 会为大负数 —— 提示数据生成有问题）

    Args:
        noisy:    含伪影输入，形状 (n_rows, n_timepoints) 或 (n_timepoints,)。
        denoised: 方法输出，形状同上。
        clean:    无伪影参考，形状同上。

    Returns:
        各行 ΔSNR 的均值（dB）。
    """
    # 统一为 2D：(n_rows, T)
    if noisy.ndim == 1:
        noisy    = noisy[np.newaxis, :]
        denoised = denoised[np.newaxis, :]
        clean    = clean[np.newaxis, :]

    n_rows    = noisy.shape[0]
    dsnrs:    list[float] = []
    n_no_art  = 0

    for i in range(n_rows):
        c  = clean[i].astype(np.float64)
        n  = noisy[i].astype(np.float64)
        d  = denoised[i].astype(np.float64)

        var_c          = np.var(c)
        var_art_before = np.var(n - c)
        var_art_after  = np.var(d - c)

        if var_art_before < 1e-8:
            n_no_art += 1

        snr_before = _snr_db_safe(var_c, var_art_before)
        snr_after  = _snr_db_safe(var_c, var_art_after)

        dsnr = float(np.clip(snr_after - snr_before, _DSNR_MIN, _DSNR_MAX))
        dsnrs.append(dsnr)

    if n_no_art > 0:
        # 少量无伪影行（< 10%）属正常退化情况，用 debug；占比过高时才 warning
        # 仅当所有行均无伪影（真正的数据问题）时 warning；部分无伪影属退化情况，降为 debug
        log_fn = logger.warning if n_no_art >= n_rows else logger.debug
        log_fn(
            "delta_snr: %d/%d 行 var(noisy-clean)<1e-8（无伪影），"
            "ΔSNR 对这些行无意义，请检查数据生成逻辑。",
            n_no_art, n_rows,
        )

    return float(np.mean(dsnrs))


def rmse(denoised: np.ndarray, clean: np.ndarray) -> float:
    """均方根误差，越低越好。

    Args:
        denoised: 处理后信号。
        clean:    参考干净信号。

    Returns:
        RMSE。
    """
    return float(np.sqrt(np.mean((denoised - clean) ** 2)))


def pearson_r(
    denoised:  np.ndarray,
    clean:     np.ndarray,
    verbose:   bool = False,
) -> float:
    """逐通道 Pearson 相关系数均值，越接近 1.0 越好。

    **修复**：对 std < 1e-8 的常数通道跳过并记录，避免除零 NaN 警告。

    Args:
        denoised: 处理后信号，形状 (n_channels, n_timepoints)。
        clean:    参考干净信号，形状相同。
        verbose:  若为 True，打印被跳过的通道数（调试用）。

    Returns:
        有效通道的平均 Pearson r；若所有通道无效则返回 nan。
    """
    if denoised.ndim == 1:
        denoised = denoised[np.newaxis, :]
        clean    = clean[np.newaxis, :]

    n_channels = denoised.shape[0]
    rs:      list[float] = []
    skipped: list[int]   = []

    for ch in range(n_channels):
        d = denoised[ch].astype(np.float64)
        c = clean[ch].astype(np.float64)

        # 替换 NaN/Inf（方法产生数值异常时回退到 clean）
        if not np.isfinite(d).all():
            d = np.where(np.isfinite(d), d, c)

        if np.std(d) < 1e-8 or np.std(c) < 1e-8:
            skipped.append(ch)
            continue

        # 用 scipy.stats.pearsonr 替代 np.corrcoef，保证数值稳定性
        r, _ = stats.pearsonr(d, c)
        if np.isfinite(r):
            rs.append(float(r))
        else:
            skipped.append(ch)

    if verbose and skipped:
        logger.debug(
            "pearson_r: 跳过 %d/%d 个通道（std≈0），idx=%s",
            len(skipped), n_channels, skipped,
        )

    return float(np.mean(rs)) if rs else float("nan")


def ssim_metric(
    denoised:    np.ndarray,
    clean:       np.ndarray,
    window_size: int            = 51,
    data_range:  Optional[float] = None,
) -> float:
    """逐通道平均 SSIM，越接近 1.0 越好。

    Args:
        denoised:    处理后信号，形状 (n_channels, n_timepoints)。
        clean:       参考干净信号，形状相同。
        window_size: 局部统计滑动窗口大小（采样点数）。
        data_range:  信号幅值范围；None 时从 clean 自动估算。

    Returns:
        各通道 SSIM 均值。
    """
    if denoised.ndim == 1:
        denoised = denoised[np.newaxis, :]
        clean    = clean[np.newaxis, :]

    if data_range is None:
        data_range = float(clean.max() - clean.min()) + 1e-10

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    sigma = window_size / 6.0        # 高斯核标准差（覆盖 ±3σ ≈ window）
    half  = window_size // 2

    def _gaussian_kernel(size: int, sig: float) -> np.ndarray:
        x = np.arange(size) - size // 2
        g = np.exp(-x ** 2 / (2 * sig ** 2))
        return g / g.sum()

    kernel = _gaussian_kernel(window_size, sigma)

    ssims = []
    for ch in range(denoised.shape[0]):
        d = denoised[ch].astype(np.float64)
        c = clean[ch].astype(np.float64)

        # 使用 np.convolve 计算局部统计量（same 模式）
        mu_d  = np.convolve(d,   kernel, mode="same")
        mu_c  = np.convolve(c,   kernel, mode="same")
        mu_d2 = np.convolve(d*d, kernel, mode="same")
        mu_c2 = np.convolve(c*c, kernel, mode="same")
        mu_dc = np.convolve(d*c, kernel, mode="same")

        sigma_d  = np.maximum(mu_d2 - mu_d ** 2, 0)
        sigma_c  = np.maximum(mu_c2 - mu_c ** 2, 0)
        sigma_dc = mu_dc - mu_d * mu_c

        numer = (2 * mu_d * mu_c + c1) * (2 * sigma_dc + c2)
        denom = (mu_d ** 2 + mu_c ** 2 + c1) * (sigma_d + sigma_c + c2)

        ssim_map = numer / np.maximum(denom, 1e-30)
        ssims.append(float(ssim_map[half:-half].mean()))   # 去掉边缘效应

    return float(np.mean(ssims))


def eta(
    denoised: np.ndarray,
    clean:    np.ndarray,
    noisy:    np.ndarray,
) -> float:
    """残余运动伪影比率 η（%），越低越好。

    η = 100 × ‖denoised − clean‖ / ‖noisy − clean‖

    0% = 完美去除；100% = 无改善；>100% = 反而变差。

    Args:
        denoised: 处理后信号。
        clean:    无伪影参考信号。
        noisy:    原始含伪影信号。

    Returns:
        η（%）。
    """
    num = np.linalg.norm(denoised - clean)
    den = np.linalg.norm(noisy    - clean)
    if den < 1e-20:
        return 0.0
    return float(100.0 * num / den)


# ===========================================================================
# 组合调用
# ===========================================================================

def compute_all_metrics(
    noisy:    np.ndarray,
    denoised: np.ndarray,
    clean:    np.ndarray,
    verbose:  bool = False,
) -> Dict[str, float]:
    """一次性计算五项评估指标。

    若 denoised 含 NaN/Inf（部分方法数值不稳定时），先替换为对应 clean 值再计算，
    保证各指标可计算，同时防止 scipy/numpy 抛出异常。

    Args:
        noisy:    含伪影信号，形状 (n_channels, n_timepoints)。
        denoised: 处理后输出，形状相同。
        clean:    无伪影参考，形状相同。
        verbose:  传递给 pearson_r 的调试开关。

    Returns:
        字典：{'delta_snr', 'rmse', 'pearson_r', 'ssim', 'eta'}。
    """
    # 全局 NaN/Inf 守护：替换为 clean，避免下游各函数分别处理
    if not np.isfinite(denoised).all():
        denoised = np.where(np.isfinite(denoised), denoised, clean)

    return {
        "delta_snr": delta_snr(noisy, denoised, clean),
        "rmse":      rmse(denoised, clean),
        "pearson_r": pearson_r(denoised, clean, verbose=verbose),
        "ssim":      ssim_metric(denoised, clean),
        "eta":       eta(denoised, clean, noisy),
    }


# ===========================================================================
# 统计检验
# ===========================================================================

def statistical_test(
    scores_a:    np.ndarray,
    scores_b:    np.ndarray,
    test:        str = "wilcoxon",
    alternative: str = "greater",
) -> Dict[str, float]:
    """配对统计检验（Wilcoxon 符号秩检验 或 配对 t 检验）。

    Args:
        scores_a:    方法 A 的指标分数，形状 (n_subjects,) 或 (n_samples,)。
        scores_b:    方法 B 的指标分数，形状相同。
        test:        'wilcoxon' 或 'ttest'。
        alternative: 假设方向，'greater' / 'less' / 'two-sided'。

    Returns:
        {'statistic': float, 'pvalue': float}。
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    if test == "wilcoxon":
        stat, pvalue = stats.wilcoxon(a, b, alternative=alternative)
    elif test == "ttest":
        stat, pvalue = stats.ttest_rel(a, b, alternative=alternative)
    else:
        raise ValueError(f"test 必须为 'wilcoxon' 或 'ttest'，收到 '{test}'")

    return {"statistic": float(stat), "pvalue": float(pvalue)}


# ===========================================================================
# 单元测试
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  metrics.py 单元测试")
    print("=" * 60)

    rng = np.random.default_rng(42)
    C, T = 4, 512
    clean_   = rng.standard_normal((C, T)).astype(np.float32)
    noise_   = 0.3 * rng.standard_normal((C, T)).astype(np.float32)
    noisy_   = clean_ + noise_
    # 理想去噪 = clean（完美方法）
    perfect_ = clean_.copy()
    # 差去噪（略微改善）
    bad_     = noisy_ * 0.9

    # ── delta_snr ──────────────────────────────────────────────
    dsnr_perfect = delta_snr(noisy_, perfect_, clean_)
    dsnr_bad     = delta_snr(noisy_, bad_,     clean_)
    assert dsnr_perfect > dsnr_bad, "完美去噪应有更高 ΔSNR"
    print(f"[PASS] delta_snr  perfect={dsnr_perfect:.2f} dB  bad={dsnr_bad:.2f} dB")

    # ── rmse ────────────────────────────────────────────────────
    rmse_perfect = rmse(perfect_, clean_)
    rmse_bad     = rmse(bad_,     clean_)
    assert rmse_perfect < rmse_bad
    assert rmse_perfect < 1e-6
    print(f"[PASS] rmse       perfect={rmse_perfect:.2e}  bad={rmse_bad:.4f}")

    # ── pearson_r — 正常情况 ────────────────────────────────────
    r_perfect = pearson_r(perfect_, clean_, verbose=True)
    r_bad     = pearson_r(bad_,     clean_, verbose=True)
    assert abs(r_perfect - 1.0) < 1e-5, f"完美去噪 Pearson r 应为 1.0，得 {r_perfect}"
    assert r_bad > 0.9
    print(f"[PASS] pearson_r  perfect={r_perfect:.6f}  bad={r_bad:.4f}")

    # ── pearson_r — 含常数通道（std=0）的健壮性 ────────────────
    constant_ch = np.zeros((2, T), dtype=np.float32)
    mixed       = np.vstack([clean_[:2], constant_ch])  # 后 2 通道全零
    r_mixed = pearson_r(mixed, clean_, verbose=True)
    assert np.isfinite(r_mixed), f"含常数通道时应返回有限值，得 {r_mixed}"
    print(f"[PASS] pearson_r（含 2 个常数通道）= {r_mixed:.4f}（仅前 2 通道有效）")

    # ── pearson_r — 全常数通道时返回 nan（非 warning）──────────
    all_const = np.zeros((C, T), dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error")   # 有 warning 则报错
        r_nan = pearson_r(all_const, clean_, verbose=True)
    assert np.isnan(r_nan), f"全常数信号应返回 nan，得 {r_nan}"
    print(f"[PASS] pearson_r（全常数信号）= {r_nan}  无 RuntimeWarning")

    # ── ssim_metric ─────────────────────────────────────────────
    ssim_p = ssim_metric(perfect_, clean_)
    ssim_b = ssim_metric(bad_,     clean_)
    assert abs(ssim_p - 1.0) < 0.01
    assert ssim_p > ssim_b
    print(f"[PASS] ssim_metric  perfect={ssim_p:.4f}  bad={ssim_b:.4f}")

    # ── eta ─────────────────────────────────────────────────────
    eta_perfect = eta(perfect_, clean_, noisy_)
    eta_bad     = eta(bad_,     clean_, noisy_)
    assert eta_perfect < 1.0          # 完美去噪 eta ≈ 0%
    assert eta_bad < 100.0
    print(f"[PASS] eta         perfect={eta_perfect:.4f}%  bad={eta_bad:.2f}%")

    # ── compute_all_metrics ─────────────────────────────────────
    metrics = compute_all_metrics(noisy_, bad_, clean_, verbose=True)
    assert set(metrics) == {"delta_snr", "rmse", "pearson_r", "ssim", "eta"}
    print(f"[PASS] compute_all_metrics = {metrics}")

    # ── statistical_test ────────────────────────────────────────
    a_scores = rng.standard_normal(20) + 1.0
    b_scores = rng.standard_normal(20)
    res = statistical_test(a_scores, b_scores, test="wilcoxon", alternative="greater")
    assert "pvalue" in res and res["pvalue"] < 0.05
    print(f"[PASS] statistical_test (wilcoxon) pvalue={res['pvalue']:.4f}")

    res_t = statistical_test(a_scores, b_scores, test="ttest", alternative="greater")
    assert res_t["pvalue"] < 0.05
    print(f"[PASS] statistical_test (ttest)    pvalue={res_t['pvalue']:.4f}")

    print("=" * 60)
    print("  全部测试通过")
    print("=" * 60)
    sys.exit(0)
