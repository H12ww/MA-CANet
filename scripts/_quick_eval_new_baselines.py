"""CNNwP / LSTM-AE 快速测试集评估。

在 data/semi_synthetic/test_*.npy 上评估训练好的模型，
输出 DELTA_SNR / RMSE / Pearson r / SSIM / eta，与 Table I 对比。

用法::

    python scripts/_quick_eval_new_baselines.py --model cnnwp
    python scripts/_quick_eval_new_baselines.py --model lstm_ae
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from pathlib import Path

from src.evaluation.metrics import compute_all_metrics


# 已知结果（对比参考）
REFERENCE = {
    "MA-CANet":   dict(delta_snr=13.60, rmse=0.238, pearson_r=0.975, params=320_801),
    "EnhancedDAE": dict(delta_snr=6.06,  rmse=0.465, pearson_r=0.809, params=322_081),
    "旧 DAE":     dict(delta_snr=5.19,  rmse=0.507, pearson_r=0.752, params=22_209),
}


def run_eval(
    model_cls,
    ckpt_path: Path,
    data_dir:  Path,
    device:    torch.device,
    batch_size: int = 128,
) -> dict:
    """加载权重，在测试集上逐样本计算指标。

    Returns:
        各指标的 mean / std 字典，以及参数量。
    """
    model = model_cls()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    noisy_np = np.load(data_dir / "test_noisy.npy")   # (N, 16, 512)
    clean_np = np.load(data_dir / "test_clean.npy")   # (N, 16, 512)
    print(f"测试集形状：noisy={noisy_np.shape}  clean={clean_np.shape}")

    N, C, L = noisy_np.shape
    noisy_flat = noisy_np.reshape(N * C, L)[:, np.newaxis, :]   # (N*C, 1, 512)
    noisy_t    = torch.from_numpy(noisy_flat.astype(np.float32))

    # 批量推理
    denoised_list = []
    with torch.no_grad():
        for i in range(0, len(noisy_t), batch_size):
            batch = noisy_t[i:i + batch_size].to(device)
            out   = model(batch)                          # (B, 1, 512)
            denoised_list.append(out.squeeze(1).cpu().numpy())
    denoised_flat = np.concatenate(denoised_list, axis=0)  # (N*C, 512)

    # 还原为 (N, C, L) 后逐样本计算指标
    denoised_np = denoised_flat.reshape(N, C, L)

    all_metrics = []
    for idx in range(N):
        m = compute_all_metrics(
            noisy    = noisy_np[idx],       # (C, L) = (16, 512)
            denoised = denoised_np[idx],    # (C, L)
            clean    = clean_np[idx],       # (C, L)
        )
        all_metrics.append(m)

    keys = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]
    results = {"n_params": n_params}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        results[f"{k}_mean"] = float(np.mean(vals))
        results[f"{k}_std"]  = float(np.std(vals))

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="快速评估 CNNwP 或 LSTM-AE")
    p.add_argument(
        "--model", type=str, required=True, choices=["cnnwp", "lstm_ae"],
        help="选择评估的模型。",
    )
    p.add_argument("--data-dir",    type=Path, default=Path("data/semi_synthetic"))
    p.add_argument("--ckpt-root",   type=Path, default=Path("outputs/checkpoints"))
    p.add_argument("--batch-size",  type=int,  default=128)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    from src.models.baselines import CNNwP, LSTMAutoencoder

    if args.model == "cnnwp":
        model_cls  = CNNwP
        model_name = "CNNwP (Huang 2024)"
        ckpt_path  = args.ckpt_root / "cnnwp" / "best_cnnwp.pth"
    else:
        model_cls  = LSTMAutoencoder
        model_name = "LSTM-AE (Yang 2025)"
        ckpt_path  = args.ckpt_root / "lstm_ae" / "best_lstm_ae.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] 未找到 checkpoint：{ckpt_path}")
        sys.exit(1)

    print(f"模型：{model_name}")
    print(f"权重：{ckpt_path}")

    results = run_eval(model_cls, ckpt_path, args.data_dir, device, args.batch_size)

    print("\n" + "=" * 65)
    print(f"  {model_name} 测试集评估结果")
    print("=" * 65)
    print(f"  参数量     : {results['n_params']:,}")
    print(f"  DELTA_SNR  : {results['delta_snr_mean']:+.2f} +/- {results['delta_snr_std']:.2f} dB")
    print(f"  RMSE       : {results['rmse_mean']:.3f} +/- {results['rmse_std']:.3f}")
    print(f"  Pearson r  : {results['pearson_r_mean']:.3f} +/- {results['pearson_r_std']:.3f}")
    print(f"  SSIM       : {results['ssim_mean']:.3f} +/- {results['ssim_std']:.3f}")
    print(f"  eta (%)    : {results['eta_mean']:.1f} +/- {results['eta_std']:.1f}")
    print("=" * 65)
    print("\n对比参考（Table I）：")
    for name, ref in REFERENCE.items():
        print(f"  {name:<15} DELTA_SNR={ref['delta_snr']:+.2f} dB  "
              f"RMSE={ref['rmse']:.3f}  Pearson r={ref['pearson_r']:.3f}  "
              f"params={ref['params']:,}")

    dsnr = results["delta_snr_mean"]
    if dsnr > 13.60:
        label = "[WARNING] 高于 MA-CANet，请检查数据泄露！"
    elif dsnr > 6.06:
        label = f"优于 EnhancedDAE（+{dsnr-6.06:.1f} dB）"
    elif dsnr > 5.19:
        label = f"介于旧 DAE 与 EnhancedDAE 之间（+{dsnr-5.19:.1f} dB 于旧 DAE）"
    else:
        label = "[WARNING] 低于旧 DAE，请检查训练配置！"
    print(f"\n  {model_name} 定位：{label}")


if __name__ == "__main__":
    main()
