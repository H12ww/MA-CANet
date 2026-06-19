"""Table IV — 各方法推理时间、参数量与 ΔSNR 综合对比。

测量方法：
  - 神经网络方法（MA-CANet、DAE）：warmup 10 次 + 计时 100 次取均值
  - 传统信号处理方法：timeit 重复 100 次取均值
  - 所有方法统一输入：单样本 (1, 512) 信号，1 通道

ΔSNR 来源：
  - 基线方法：从 outputs/Table_I.csv 读取均值
  - MA-CANet：从 outputs/Table_II_ablation.csv 读取 A5 行

输出：
  outputs/Table_IV_inference_speed.csv    — 完整对比表
  （控制台）TABLE IV 格式化打印

用法::

    python scripts/11_inference_speed.py \\
        [--ckpt-a5   outputs/checkpoints/ablation/A5_best.pth] \\
        [--config    configs/default.yaml] \\
        [--output-dir outputs] \\
        [--n-warmup  10] \\
        [--n-measure 100]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIGNAL_LEN = 512    # 模型统一输入长度（样本数）
FS         = 10.0   # 采样频率（Hz）


# ============================================================
# 辅助：参数量统计
# ============================================================

def count_params(model: torch.nn.Module) -> int:
    """统计可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 神经网络方法计时（CPU / GPU）
# ============================================================

@torch.no_grad()
def time_nn_method(
    model:     torch.nn.Module,
    device:    torch.device,
    n_warmup:  int = 10,
    n_measure: int = 100,
) -> float:
    """测量单样本 (1, 1, 512) 推理时间（毫秒）。

    在 GPU 上使用 CUDA Event 实现精确计时；
    CPU 上使用 time.perf_counter。

    Args:
        model:     已加载权重、已 .eval() 的模型。
        device:    推理设备。
        n_warmup:  预热次数（不计入统计）。
        n_measure: 正式计时次数。

    Returns:
        平均推理时间（ms）。
    """
    model = model.to(device).eval()
    x = torch.randn(1, 1, SIGNAL_LEN, device=device)

    # 预热
    for _ in range(n_warmup):
        _ = model(x)

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(n_measure):
            _ = model(x)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed_ms = start_ev.elapsed_time(end_ev) / n_measure
    else:
        t0 = time.perf_counter()
        for _ in range(n_measure):
            _ = model(x)
        elapsed_ms = (time.perf_counter() - t0) / n_measure * 1000.0

    return elapsed_ms


# ============================================================
# 传统方法计时
# ============================================================

def time_traditional_method(
    method_fn,
    n_warmup:  int = 10,
    n_measure: int = 100,
) -> float:
    """测量传统方法单样本 (1, 512) 推理时间（毫秒）。

    Args:
        method_fn: 接受 (n_channels, n_timepoints) 的可调用对象。
        n_warmup:  预热次数。
        n_measure: 正式计时次数。

    Returns:
        平均推理时间（ms）。
    """
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((1, SIGNAL_LEN)).astype(np.float32)

    # 预热
    for _ in range(n_warmup):
        method_fn(sig)

    t0 = time.perf_counter()
    for _ in range(n_measure):
        method_fn(sig)
    return (time.perf_counter() - t0) / n_measure * 1000.0


# ============================================================
# ΔSNR 读取
# ============================================================

def _parse_delta_snr(fmt_str: str) -> float:
    """从 "X.XX±Y.YY" 或 "X.XX±Y.YY *" 中提取均值。"""
    return float(fmt_str.split("±")[0].strip().rstrip(" *"))


def load_delta_snr(output_dir: Path) -> dict[str, float | None]:
    """从 Table_I.csv 和 Table_II_ablation.csv 提取各方法 ΔSNR 均值。

    Returns:
        方法名 → ΔSNR (dB) 的映射；未找到时为 None。
    """
    result: dict[str, float | None] = {}

    table1_path = output_dir / "Table_I.csv"
    if table1_path.exists():
        import pandas as pd
        df = pd.read_csv(table1_path)
        for _, row in df.iterrows():
            name = str(row["Method"])
            try:
                result[name] = _parse_delta_snr(str(row["delta_snr_fmt"]))
            except Exception:
                result[name] = None
        logger.info("从 Table_I.csv 读取 %d 个基线 ΔSNR", len(result))
    else:
        logger.warning("未找到 %s，ΔSNR 列将为空", table1_path)

    table2_path = output_dir / "Table_II_ablation.csv"
    if table2_path.exists():
        import pandas as pd
        df2 = pd.read_csv(table2_path)
        row_a5 = df2[df2["ablation_id"] == "A5"]
        if not row_a5.empty:
            result["MA-CANet"] = float(row_a5.iloc[0]["delta_snr"])
            logger.info("MA-CANet ΔSNR = %.2f dB（来自 Table_II_ablation）",
                        result["MA-CANet"])

    return result


# ============================================================
# 构建各方法
# ============================================================

def build_methods(ckpt_a5: Path, cfg: dict):
    """返回待测方法列表，每项为 dict：
        name, category, fn_or_model, device, has_params
    """
    from src.models.baselines import (
        BandpassFilter, WaveletThreshold, SplineInterpolation,
        TDDR, PCAMethod, DAENet,
    )
    from src.models.macanet import MACANetAblation

    methods = []

    # ── 传统方法 ─────────────────────────────────────────────

    bp = BandpassFilter(fs=FS)
    methods.append({
        "name":       "Bandpass",
        "category":   "traditional",
        "fn":         lambda s, _bp=bp: _bp.process(s),
        "n_params":   None,
    })

    wt = WaveletThreshold(wavelet="db4", level=4)
    methods.append({
        "name":       "Wavelet",
        "category":   "traditional",
        "fn":         lambda s, _wt=wt: _wt.process(s),
        "n_params":   None,
    })

    sp = SplineInterpolation(fs=FS)
    methods.append({
        "name":       "Spline",
        "category":   "traditional",
        "fn":         lambda s, _sp=sp: _sp.process(s),
        "n_params":   None,
    })

    td = TDDR(fs=FS)
    methods.append({
        "name":       "TDDR",
        "category":   "traditional",
        "fn":         lambda s, _td=td: _td.process(s),
        "n_params":   None,
    })

    pca = PCAMethod(n_artifact_components=1)
    methods.append({
        "name":       "PCA",
        "category":   "traditional",
        "fn":         lambda s, _pca=pca: _pca.process(s),
        "n_params":   None,
    })

    # ── 神经网络方法 ─────────────────────────────────────────

    dae = DAENet(input_length=SIGNAL_LEN)
    dae.eval()
    methods.append({
        "name":     "DAE",
        "category": "neural",
        "model":    dae,
        "n_params": count_params(dae),
    })

    # MA-CANet（A5 配置）
    macanet = MACANetAblation(
        ablation_id="A5",
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        ms_out_channels=cfg.get("model", {}).get("ms_out_channels", 32),
        encoder_channels=cfg.get("model", {}).get("encoder_channels", [32, 64, 128, 128]),
        se_reduction=cfg.get("model", {}).get("se_reduction", 8),
        dropout=cfg.get("model", {}).get("dropout", 0.3),
    )
    if ckpt_a5.exists():
        state = torch.load(ckpt_a5, map_location="cpu", weights_only=False)
        macanet.load_state_dict(state.get("model_state_dict", state))
        logger.info("MA-CANet 权重已加载：%s", ckpt_a5.name)
    else:
        logger.warning("未找到 MA-CANet checkpoint：%s（使用随机权重）", ckpt_a5)
    macanet.eval()
    methods.append({
        "name":     "MA-CANet",
        "category": "neural",
        "model":    macanet,
        "n_params": count_params(macanet),
    })

    return methods


# ============================================================
# 主测量循环
# ============================================================

def run_measurements(
    methods:   list[dict],
    devices:   list[torch.device],
    n_warmup:  int,
    n_measure: int,
    delta_snr: dict[str, float | None],
) -> list[dict]:
    """对所有方法在指定设备上计时，返回结果记录列表。"""
    records = []

    for m in methods:
        name     = m["name"]
        category = m["category"]
        n_params = m.get("n_params")

        logger.info("测量：%s  [%s]", name, category)

        row: dict = {
            "Method":   name,
            "Category": category,
            "Params":   n_params,
            "ΔSNR(dB)": delta_snr.get(name),
        }

        if category == "traditional":
            t_ms = time_traditional_method(m["fn"], n_warmup, n_measure)
            row["Time_CPU_ms"] = round(t_ms, 3)
            row["Time_GPU_ms"] = None
            logger.info("  CPU: %.3f ms", t_ms)

        else:  # neural
            model = m["model"]

            # CPU 计时
            t_cpu = time_nn_method(model, torch.device("cpu"), n_warmup, n_measure)
            row["Time_CPU_ms"] = round(t_cpu, 3)
            logger.info("  CPU: %.3f ms", t_cpu)

            # GPU 计时（若可用）
            if torch.cuda.is_available():
                t_gpu = time_nn_method(model, torch.device("cuda"), n_warmup, n_measure)
                row["Time_GPU_ms"] = round(t_gpu, 3)
                logger.info("  GPU: %.3f ms", t_gpu)
            else:
                row["Time_GPU_ms"] = None
                logger.info("  GPU: 不可用")

        records.append(row)

    return records


# ============================================================
# 打印 & 保存
# ============================================================

def _fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _fmt_ms(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "N/A"


def _fmt_snr(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


def print_table(records: list[dict]) -> None:
    """在控制台输出 Table IV。"""
    cols = ["Method", "Params", "Time_CPU_ms", "Time_GPU_ms", "ΔSNR(dB)"]
    widths = [12, 10, 14, 14, 10]

    print("\n" + "=" * 66)
    print("  TABLE IV — Inference Time & Complexity Comparison")
    print("=" * 66)
    hdr = "  " + "  ".join(f"{c:>{widths[i]}}" for i, c in enumerate(cols))
    print(hdr)
    print("  " + "-" * 62)

    for r in records:
        line = (
            f"  {r['Method']:>{widths[0]}}  "
            f"{_fmt_params(r['Params']):>{widths[1]}}  "
            f"{_fmt_ms(r['Time_CPU_ms']):>{widths[2]}}  "
            f"{_fmt_ms(r['Time_GPU_ms']):>{widths[3]}}  "
            f"{_fmt_snr(r['ΔSNR(dB)']):>{widths[4]}}"
        )
        print(line)

    print("=" * 66)
    print("  注：推理时间 = 单样本 512 点 (1 通道) 平均值；")
    print("      Params 仅对神经网络有意义；ΔSNR 来自测试集均值。")
    print("=" * 66 + "\n")


def save_csv(records: list[dict], output_dir: Path) -> None:
    """保存 Table IV 至 CSV。"""
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = output_dir / "Table_IV_inference_speed.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Table IV 已保存：%s", csv_path)


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="测量各方法推理速度并输出 Table IV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-a5",    type=Path,
                   default=Path("outputs/checkpoints/ablation/A5_best.pth"))
    p.add_argument("--config",     type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--n-warmup",   type=int,  default=10,
                   help="预热次数（不计入统计）")
    p.add_argument("--n-measure",  type=int,  default=100,
                   help="正式计时次数")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("预热 %d 次 + 计时 %d 次，输入长度 %d 点",
                args.n_warmup, args.n_measure, SIGNAL_LEN)

    # 加载 ΔSNR
    delta_snr = load_delta_snr(args.output_dir)

    # 构建方法
    methods = build_methods(args.ckpt_a5, cfg)

    # 测量
    devices = [torch.device("cpu")]
    records = run_measurements(methods, devices, args.n_warmup, args.n_measure, delta_snr)

    # 输出
    print_table(records)
    save_csv(records, args.output_dir)

    logger.info("Table IV 生成完毕。")


if __name__ == "__main__":
    main()
