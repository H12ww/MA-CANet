#!/usr/bin/env python3
"""Generate semi-synthetic (noisy, clean) training pairs for MA-CANet.

Pipeline
--------
1. Load BS/R segments from data/processed/ as physiological background
2. Synthesise HRF via gamma-based kernel (Gao 2022 make_HRFs.m)
3. Simulate spike and baseline-shift motion artifacts
4. Scale artifact to target SNR (uniform draw from snr_range)
5. noisy = clean + scaled artifact
6. Split by subject: 14 train / 3 val / 3 test  (≈ 70/15/15 %)
7. Save to data/semi_synthetic/ as float32 .npy files

Output layout
-------------
data/semi_synthetic/
    train_noisy.npy   float32  (3500, 16, 512)
    train_clean.npy   float32  (3500, 16, 512)
    val_noisy.npy     float32  ( 750, 16, 512)
    val_clean.npy     float32  ( 750, 16, 512)
    test_noisy.npy    float32  ( 750, 16, 512)
    test_clean.npy    float32  ( 750, 16, 512)

Usage
-----
    python scripts/03_generate_semi_synthetic.py
    python scripts/03_generate_semi_synthetic.py --n-samples 2000 --snr-min -5 --snr-max 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.semi_synthetic import (
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    generate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate semi-synthetic fNIRS training pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing preprocessed .npy files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "semi_synthetic",
        help="Output directory for generated .npy files.",
    )
    p.add_argument("--n-samples",  type=int,   default=5000,  help="Total number of pairs.")
    p.add_argument("--snr-min",    type=float, default=-10.0, help="Min SNR (dB).")
    p.add_argument("--snr-max",    type=float, default=10.0,  help="Max SNR (dB).")
    p.add_argument("--hrf-amp-min",  type=float, default=0.5, help="Min HRF amplitude (z-score).")
    p.add_argument("--hrf-amp-max",  type=float, default=2.0, help="Max HRF amplitude (z-score).")
    p.add_argument("--hrf-ttp",      type=float, default=6.0, help="HRF time-to-peak (s).")
    p.add_argument("--hrf-duration", type=float, default=16.5,help="HRF boxcar duration (s).")
    p.add_argument("--no-hrf-frac",  type=float, default=0.1, help="Fraction of rest-only samples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ── Statistics display ────────────────────────────────────────────────────────

def print_stats(
    results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    W = 68
    print("\n" + "=" * W)
    print("  SEMI-SYNTHETIC DATASET STATISTICS")
    print("=" * W)
    print(f"  {'Split':<8}  {'Pairs':>6}  {'Shape':>20}  {'MB noisy':>9}  {'MB clean':>9}")
    print("  " + "-" * (W - 2))

    for split, arrs in results.items():
        noisy = arrs["noisy"]
        clean = arrs["clean"]
        print(
            f"  {split:<8}  {len(noisy):>6}  {str(noisy.shape):>20}"
            f"  {noisy.nbytes / 1e6:>9.1f}  {clean.nbytes / 1e6:>9.1f}"
        )

    print("  " + "-" * (W - 2))
    all_noisy = np.concatenate([r["noisy"] for r in results.values()], axis=0)
    all_clean = np.concatenate([r["clean"] for r in results.values()], axis=0)
    print(
        f"  {'TOTAL':<8}  {len(all_noisy):>6}  {str(all_noisy.shape):>20}"
        f"  {all_noisy.nbytes / 1e6:>9.1f}  {all_clean.nbytes / 1e6:>9.1f}"
    )
    print("=" * W)

    print()
    print("  Signal statistics (all splits combined):")
    print(f"  {'':>8}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    print("  " + "-" * 52)
    for label, arr in [("noisy", all_noisy), ("clean", all_clean)]:
        print(
            f"  {label:<8}  {arr.mean():>10.4f}  {arr.std():>10.4f}"
            f"  {arr.min():>10.4f}  {arr.max():>10.4f}"
        )

    print()
    print("  Artifact power  (var(noisy-clean) / var(clean), per split):")
    print(f"  {'Split':<8}  {'mean SNR (dB)':>14}  {'std SNR (dB)':>13}")
    print("  " + "-" * 40)
    for split, arrs in results.items():
        diff = arrs["noisy"] - arrs["clean"]
        snr_per_sample = 10 * np.log10(
            (np.var(arrs["clean"], axis=(1, 2)) + 1e-12)
            / (np.var(diff, axis=(1, 2)) + 1e-12)
        )
        print(f"  {split:<8}  {snr_per_sample.mean():>14.2f}  {snr_per_sample.std():>13.2f}")

    print()
    print("  Subject split:")
    print(f"  train : {TRAIN_SUBJECTS[0]} … {TRAIN_SUBJECTS[-1]}  ({len(TRAIN_SUBJECTS)} subjects)")
    print(f"  val   : {VAL_SUBJECTS[0]} … {VAL_SUBJECTS[-1]}  ({len(VAL_SUBJECTS)} subjects)")
    print(f"  test  : {TEST_SUBJECTS[0]} … {TEST_SUBJECTS[-1]}  ({len(TEST_SUBJECTS)} subjects)")

    print()
    npy_files = sorted(output_dir.rglob("*.npy"))
    total_mb = sum(f.stat().st_size for f in npy_files) / 1e6
    print(f"  Saved {len(npy_files)} .npy files  |  {total_mb:.1f} MB on disk")
    print(f"  Location: {output_dir.resolve()}")
    print("=" * W + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{'=' * 55}")
    print("  Semi-Synthetic Data Generation — MA-CANet")
    print(f"{'=' * 55}")
    print(f"  Processed dir : {args.processed_dir}")
    print(f"  Output dir    : {args.output}")
    print(f"  n_samples     : {args.n_samples}")
    print(f"  SNR range     : [{args.snr_min}, {args.snr_max}] dB")
    print(f"  artifact type : spike_only(30%) / shift_only(30%) / both(30%) / mild(10%)")
    print(f"  HRF amp range : [{args.hrf_amp_min}, {args.hrf_amp_max}]")
    print(f"  HRF ttp / dur : {args.hrf_ttp} s / {args.hrf_duration} s")
    print(f"  no_hrf_frac   : {args.no_hrf_frac}")
    print(f"  seed          : {args.seed}")
    print(f"{'=' * 55}\n")

    if not args.processed_dir.exists():
        logger.error(
            "Processed directory not found: %s\n"
            "Run scripts/02_preprocess.py first.",
            args.processed_dir,
        )
        sys.exit(1)

    t0 = time.time()
    results = generate_dataset(
        processed_dir=args.processed_dir,
        output_dir=args.output,
        n_samples=args.n_samples,
        snr_range=(args.snr_min, args.snr_max),
        hrf_amp_range=(args.hrf_amp_min, args.hrf_amp_max),
        hrf_time_to_peak=args.hrf_ttp,
        hrf_duration=args.hrf_duration,
        no_hrf_fraction=args.no_hrf_frac,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    logger.info("Generation complete in %.1f s", elapsed)
    print_stats(results, args.output)


if __name__ == "__main__":
    main()
