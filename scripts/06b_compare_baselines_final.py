"""最终基线对比评估脚本（Table I Final）。

在 test 集上评估 9 种方法：
  5 传统方法（直接复用 comparison_scores.csv 中的已有分数）+
  3 DL 基线（EnhancedDAE/"DAE-Large"、CNNwP、LSTM-AE，重新推理）+
  MA-CANet（5 折集成，重新推理）

输出：
  - outputs/Table_I_final.csv          新主表
  - outputs/comparison_scores_final.csv 逐样本分数

用法::

    python scripts/06b_compare_baselines_final.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 常量
# ===========================================================================

METHOD_ORDER = [
    "Bandpass", "Wavelet", "Spline", "TDDR", "PCA",
    "DAE-Large", "CNNwP", "LSTM-AE", "MA-CANet",
]

METRIC_COLS = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]

METRIC_LABELS = {
    "delta_snr":  "ΔSNR (dB)",
    "rmse":       "RMSE",
    "pearson_r":  "Pearson r",
    "ssim":       "SSIM",
    "eta":        "η (%)",
}

METRIC_HIGHER_BETTER = {
    "delta_snr": True,
    "rmse":      False,
    "pearson_r": True,
    "ssim":      True,
    "eta":       False,
}

# DL 基线配置：(显示名, 模型类, checkpoint路径)
DL_BASELINES = [
    ("DAE-Large", "EnhancedDAE", "outputs/checkpoints/enhanced_dae/best_enhanced_dae.pth"),
    ("CNNwP",     "CNNwP",       "outputs/checkpoints/cnnwp/best_cnnwp.pth"),
    ("LSTM-AE",   "LSTMAutoencoder", "outputs/checkpoints/lstm_ae/best_lstm_ae.pth"),
]


# ===========================================================================
# MA-CANet 集成推理（与原脚本一致）
# ===========================================================================

def load_macanet_ensemble(ckpt_dir: Path, folds: list[int], cfg: dict, device: torch.device):
    import glob
    from src.models.macanet import MACANet
    models = []
    for fold in folds:
        pattern = str(ckpt_dir / f"fold_{fold}" / "checkpoints" / "best_*.pth")
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.warning("fold_%d: 未找到 checkpoint，跳过。", fold)
            continue
        state = torch.load(Path(paths[-1]), map_location="cpu", weights_only=False)
        model = MACANet.from_config(cfg)
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(device).eval()
        models.append(model)
        logger.info("fold_%d: 已加载 %s", fold, Path(paths[-1]).name)
    if not models:
        logger.error("没有可用的 MA-CANet checkpoint，请先训练。")
        sys.exit(1)
    return models


@torch.no_grad()
def run_macanet(models, noisy: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    N, C, L = noisy.shape
    flat = noisy.reshape(N * C, 1, L).astype(np.float32)
    preds = np.zeros_like(flat)
    for start in range(0, len(flat), batch_size):
        x = torch.from_numpy(flat[start:start + batch_size]).to(device)
        batch_pred = sum(m(x) for m in models) / len(models)
        preds[start:start + batch_size] = batch_pred.cpu().numpy()
    return preds.reshape(N, C, L)


# ===========================================================================
# DL 基线推理
# ===========================================================================

@torch.no_grad()
def run_dl_baseline(model_cls_name: str, ckpt_path: Path, noisy: np.ndarray,
                    device: torch.device, batch_size: int = 128) -> np.ndarray:
    """加载 DL 基线权重并对 (N, C, L) 数据做推理。"""
    from src.models.baselines import EnhancedDAE, CNNwP, LSTMAutoencoder
    cls_map = {
        "EnhancedDAE":    EnhancedDAE,
        "CNNwP":          CNNwP,
        "LSTMAutoencoder": LSTMAutoencoder,
    }
    model = cls_map[model_cls_name]()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval().to(device)

    N, C, L = noisy.shape
    flat   = noisy.reshape(N * C, L)[:, np.newaxis, :].astype(np.float32)
    noisy_t = torch.from_numpy(flat)
    outs = []
    for i in range(0, len(noisy_t), batch_size):
        batch = noisy_t[i:i + batch_size].to(device)
        out   = model(batch)
        outs.append(out.squeeze(1).cpu().numpy())
    flat_out = np.concatenate(outs, axis=0)
    return flat_out.reshape(N, C, L)


# ===========================================================================
# 逐样本指标计算
# ===========================================================================

def compute_scores(method: str, noisy: np.ndarray, denoised: np.ndarray,
                   clean: np.ndarray) -> list[dict]:
    N = noisy.shape[0]
    records = []
    for i in range(N):
        m = compute_all_metrics(noisy[i], denoised[i], clean[i])
        records.append({
            "method":     method,
            "sample_idx": i,
            "delta_snr":  m["delta_snr"],
            "rmse":       m["rmse"],
            "pearson_r":  m["pearson_r"],
            "ssim":       m["ssim"],
            "eta":        m["eta"],
        })
    return records


# ===========================================================================
# 汇总表生成
# ===========================================================================

def make_table(scores_df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = scores_df[scores_df["method"] == method]
        if len(sub) == 0:
            logger.warning("方法 %s 无数据，跳过。", method)
            continue
        row: dict = {"Method": method}
        for metric in METRIC_COLS:
            vals = sub[metric].dropna().values
            mean_ = float(np.mean(vals)) if len(vals) else float("nan")
            std_  = float(np.std(vals))  if len(vals) else float("nan")
            row[metric]            = mean_
            row[f"{metric}_std"]   = std_
            row[f"{metric}_fmt"]   = f"{mean_:.4f}±{std_:.4f}"
        rows.append(row)

    table = pd.DataFrame(rows)

    # 标注最佳值
    for metric in METRIC_COLS:
        vals = table[metric].values
        if METRIC_HIGHER_BETTER[metric]:
            best_idx = int(np.nanargmax(vals))
        else:
            best_idx = int(np.nanargmin(vals))
        table.at[best_idx, f"{metric}_fmt"] += " *"

    fmt_cols = ["Method"] + [f"{m}_fmt" for m in METRIC_COLS]
    table[fmt_cols].to_csv(output_csv, index=False)
    logger.info("Table I Final 已保存：%s", output_csv)

    # 控制台打印
    print("\n" + "=" * 110)
    print("  TABLE I FINAL — 9 方法对比（test set，mean±std，* = 最佳值）")
    print("=" * 110)
    header = f"  {'Method':<16}" + "".join(f"  {METRIC_LABELS[m]:>20}" for m in METRIC_COLS)
    print(header)
    print("  " + "-" * 106)
    for _, r in table.iterrows():
        line = f"  {r['Method']:<16}"
        for metric in METRIC_COLS:
            val = r.get(f"{metric}_fmt", "")
            line += f"  {str(val):>20}"
        print(line)
    print("=" * 110 + "\n")

    return table


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    root    = Path(".")
    cfg_path = root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("推理设备：%s", device)

    test_noisy = np.load("data/semi_synthetic/test_noisy.npy")   # (N, C, L)
    test_clean  = np.load("data/semi_synthetic/test_clean.npy")
    N, C, L = test_noisy.shape
    logger.info("测试集：%s", test_noisy.shape)

    all_records: list[dict] = []

    # ── Step 1：复用传统方法已有分数 ───────────────────────────────────────
    trad_methods = ["Bandpass", "Wavelet", "Spline", "TDDR", "PCA"]
    existing_csv = root / "outputs" / "comparison_scores.csv"
    if existing_csv.exists():
        existing_df = pd.read_csv(existing_csv)
        trad_df = existing_df[existing_df["method"].isin(trad_methods)]
        all_records.extend(trad_df.to_dict("records"))
        logger.info("传统方法分数已从 %s 加载（%d 行）", existing_csv, len(trad_df))
    else:
        logger.warning("未找到 %s，将重新计算传统方法分数。", existing_csv)
        from src.models.baselines import (
            bandpass_filter, wavelet_denoise, spline_correction,
            tddr_correction, pca_denoise,
        )
        fs = float(cfg.get("data", {}).get("sampling_rate", 10.0))
        trad_fns = [
            ("Bandpass", lambda x: bandpass_filter(x, fs=fs)),
            ("Wavelet",  lambda x: wavelet_denoise(x, wavelet="db4", level=4)),
            ("Spline",   lambda x: spline_correction(x, fs=fs)),
            ("TDDR",     lambda x: tddr_correction(x, fs=fs)),
            ("PCA",      lambda x: pca_denoise(x, n_artifact_components=1)),
        ]
        for name, fn in trad_fns:
            t0 = time.time()
            out = np.stack([fn(test_noisy[i]) for i in range(N)])
            all_records.extend(compute_scores(name, test_noisy, out, test_clean))
            logger.info("%s  完成  %.1f s", name, time.time() - t0)

    # ── Step 2：DL 基线推理 ────────────────────────────────────────────────
    for display_name, cls_name, ckpt_rel in DL_BASELINES:
        ckpt_path = root / ckpt_rel
        if not ckpt_path.exists():
            logger.error("[%s] checkpoint 不存在：%s", display_name, ckpt_path)
            sys.exit(1)
        t0 = time.time()
        denoised = run_dl_baseline(cls_name, ckpt_path, test_noisy, device)
        all_records.extend(compute_scores(display_name, test_noisy, denoised, test_clean))
        logger.info("%s  完成  %.1f s", display_name, time.time() - t0)

    # ── Step 3：MA-CANet 集成推理 ──────────────────────────────────────────
    t0 = time.time()
    ckpt_dir = root / "outputs" / "checkpoints"
    models   = load_macanet_ensemble(ckpt_dir, list(range(5)), cfg, device)
    denoised = run_macanet(models, test_noisy, device)
    all_records.extend(compute_scores("MA-CANet", test_noisy, denoised, test_clean))
    logger.info("MA-CANet  完成  %.1f s", time.time() - t0)

    # ── 保存逐样本分数 ──────────────────────────────────────────────────────
    scores_df = pd.DataFrame(all_records)
    out_dir   = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    scores_csv = out_dir / "comparison_scores_final.csv"
    scores_df.to_csv(scores_csv, index=False)
    logger.info("逐样本分数已保存：%s  (%d 行)", scores_csv, len(scores_df))

    # ── 生成汇总表 ──────────────────────────────────────────────────────────
    make_table(scores_df, out_dir / "Table_I_final.csv")


if __name__ == "__main__":
    main()
