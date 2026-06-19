"""对 comparison_scores_final.csv 做 Wilcoxon 配对检验（各方法 vs MA-CANet）。

输出：
  - outputs/wilcoxon_tests_final.csv

用法::

    python scripts/06c_wilcoxon_final.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHOD_ORDER = [
    "Bandpass", "Wavelet", "Spline", "TDDR", "PCA",
    "DAE-Large", "CNNwP", "LSTM-AE",
]

METRIC_COLS = ["delta_snr", "rmse", "pearson_r", "ssim", "eta"]


def main() -> None:
    scores_path = Path("outputs/comparison_scores_final.csv")
    if not scores_path.exists():
        print(f"[ERROR] 未找到 {scores_path}，请先运行 06b_compare_baselines_final.py")
        return

    scores_df = pd.read_csv(scores_path)
    ma_df     = scores_df[scores_df["method"] == "MA-CANet"]

    records = []
    for method in METHOD_ORDER:
        m_df = scores_df[scores_df["method"] == method]
        if len(m_df) == 0:
            print(f"[WARN] 方法 {method} 无数据，跳过")
            continue
        for metric in METRIC_COLS:
            a = ma_df[metric].values
            b = m_df[metric].values
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            try:
                # 单边检验：alternative='greater' 验证 MA-CANet 在 delta_snr/pearson_r/ssim 更大
                # 对 rmse/eta 则方向相反（MA-CANet 更小）
                if METRIC_COLS.index(metric) in [0, 2, 3]:  # higher=better
                    stat, pval = wilcoxon(a, b, alternative="greater")
                else:  # lower=better
                    stat, pval = wilcoxon(a, b, alternative="less")
            except ValueError:
                stat, pval = float("nan"), float("nan")
            records.append({
                "method":    method,
                "metric":    metric,
                "statistic": stat,
                "p_value":   pval,
                "n_samples": n,
                "significant": pval < 0.001 if not np.isnan(pval) else False,
            })

    result_df = pd.DataFrame(records)
    out_path  = Path("outputs/wilcoxon_tests_final.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nWilcoxon 检验结果已保存：{out_path}")

    # 汇总打印（仅 delta_snr）
    print("\n" + "=" * 70)
    print("  Wilcoxon 配对检验（各方法 vs MA-CANet，ΔSNR 指标，单边）")
    print("=" * 70)
    print(f"  {'方法':<16}  {'统计量':>12}  {'p-value':>14}  {'n':>6}  {'显著*':>6}")
    print("  " + "-" * 66)
    snr_df = result_df[result_df["metric"] == "delta_snr"]
    for _, row in snr_df.iterrows():
        sig = "p<0.001" if row["significant"] else (
            "p<0.05" if row["p_value"] < 0.05 else "n.s."
        )
        print(
            f"  {row['method']:<16}  {row['statistic']:>12.1f}  "
            f"{row['p_value']:>14.3e}  {int(row['n_samples']):>6}  {sig:>6}"
        )
    print("=" * 70)
    n_sig = snr_df["significant"].sum()
    print(f"\n  {n_sig}/{len(snr_df)} 个方法在 ΔSNR 上显著低于 MA-CANet（p < 0.001）")

    # 全指标汇总
    n_total_sig = result_df["significant"].sum()
    n_total     = len(result_df)
    print(f"  全指标：{n_total_sig}/{n_total} 对检验 p < 0.001\n")


if __name__ == "__main__":
    main()
