"""EnhancedDAE 快速测试集评估。

在 data/semi_synthetic/test_*.npy 上评估训练好的 EnhancedDAE，
输出 DELTA_SNR / RMSE / Pearson r，与 Table I 的 MA-CANet 和旧 DAE 对比。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from pathlib import Path

from src.models.baselines import EnhancedDAE
from src.evaluation.metrics import compute_all_metrics


def run_eval(
    ckpt_path: Path,
    data_dir:  Path,
    device:    torch.device,
    batch_size: int = 128,
) -> dict:
    """加载权重，在测试集上逐样本计算指标。

    Returns:
        各指标的 mean / std 字典。
    """
    # 加载模型
    model = EnhancedDAE()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval().to(device)

    # 加载测试集
    noisy_np = np.load(data_dir / "test_noisy.npy")   # (N, 16, 512)
    clean_np = np.load(data_dir / "test_clean.npy")   # (N, 16, 512)
    print(f"测试集形状：noisy={noisy_np.shape}  clean={clean_np.shape}")

    N, C, L = noisy_np.shape
    # 推理时逐样本展平通道：(N, C, L) → (N, C*1, L) 逐通道送入模型
    noisy_flat = noisy_np.reshape(N * C, L)[:, np.newaxis, :]   # (N*C, 1, 512)
    noisy_t    = torch.from_numpy(noisy_flat.astype(np.float32))

    # 批量推理，输出 (N*C, 512)
    denoised_list = []
    with torch.no_grad():
        for i in range(0, len(noisy_t), batch_size):
            batch = noisy_t[i:i + batch_size].to(device)
            out   = model(batch)                          # (B, 1, 512)
            denoised_list.append(out.squeeze(1).cpu().numpy())
    denoised_flat = np.concatenate(denoised_list, axis=0)  # (N*C, 512)

    # 还原为 (N, C, L) 后逐样本计算指标（compute_all_metrics 期望 (C, L) 输入）
    denoised_np = denoised_flat.reshape(N, C, L)

    all_metrics = []
    for idx in range(N):
        m = compute_all_metrics(
            noisy    = noisy_np[idx],       # (C, L) = (16, 512)
            denoised = denoised_np[idx],    # (C, L)
            clean    = clean_np[idx],       # (C, L)
        )
        all_metrics.append(m)

    # 汇总
    keys = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]
    results = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        results[f"{k}_mean"] = float(np.mean(vals))
        results[f"{k}_std"]  = float(np.std(vals))

    return results


if __name__ == "__main__":
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = Path("outputs/checkpoints/enhanced_dae/best_enhanced_dae.pth")
    data_dir = Path("data/semi_synthetic")

    if not ckpt.exists():
        print(f"[ERROR] 未找到 checkpoint：{ckpt}")
        sys.exit(1)

    print(f"设备：{device}")
    print(f"权重：{ckpt}")
    results = run_eval(ckpt, data_dir, device)

    print("\n" + "=" * 60)
    print("  EnhancedDAE 测试集评估结果")
    print("=" * 60)
    print(f"  DELTA_SNR  : {results['delta_snr_mean']:+.2f} +/- {results['delta_snr_std']:.2f} dB")
    print(f"  RMSE       : {results['rmse_mean']:.3f} +/- {results['rmse_std']:.3f}")
    print(f"  Pearson r  : {results['pearson_r_mean']:.3f} +/- {results['pearson_r_std']:.3f}")
    print(f"  SSIM       : {results['ssim_mean']:.3f} +/- {results['ssim_std']:.3f}")
    print(f"  eta (%)    : {results['eta_mean']:.1f} +/- {results['eta_std']:.1f}")
    print("=" * 60)
    print("\n对比参考（来自 Table I）：")
    print("  MA-CANet   DELTA_SNR = 13.60 dB  RMSE = 0.238  Pearson r = 0.975")
    print("  旧 DAE     DELTA_SNR =  5.19 dB  RMSE = 0.507  Pearson r = 0.752")
    dsnr = results['delta_snr_mean']
    if dsnr > 13.60:
        label = "优于 MA-CANet（异常，请检查数据泄露）"
    elif dsnr > 5.19:
        label = f"介于旧 DAE 与 MA-CANet 之间（+{dsnr-5.19:.1f} dB 于旧 DAE）"
    else:
        label = "低于旧 DAE（意外，请检查训练配置）"
    print(f"\n  EnhancedDAE 定位：{label}")
