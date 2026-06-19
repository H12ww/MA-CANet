#!/usr/bin/env python3
"""Preprocess all 20 subjects: .snirf → HbO/HbR segments → .npy files.

Pipeline (per subject)
----------------------
1. Load .snirf via MNE  →  raw CW amplitude  (16 ch, 10 Hz)
2. optical_density()    →  ΔOD
3. beer_lambert_law()   →  HbO + HbR  (8 pairs × 2 = 16 ch)
4. Segment by event     →  HT / SM / LM / R / BS
5. Z-score normalise    →  per segment, per channel
6. Save to disk         →  data/processed/<subject_id>/

Output layout
-------------
data/processed/
  sub-01/
    sub-01_HT.npy    float32  (25, 16, 100)   ← stacked, uniform length
    sub-01_SM.npy    float32  (25, 16, 100)
    sub-01_LM.npy    float32  (25, 16, 100)
    sub-01_R.npy     float32  (75, 16, 200)
    sub-01_BS_000.npy float32 (16, N)          ← individual, variable length
    sub-01_BS_001.npy float32 (16, M)
  sub-02/ ...
  ...
  sub-20/ ...

Axis-0 channel order in all arrays:
  [0–7]   HbO  for S1_D1 … S8_D2
  [8–15]  HbR  for S1_D1 … S8_D2

Usage
-----
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --subjects sub-01 sub-02 sub-03
    python scripts/02_preprocess.py --ppf 5.1 --output data/processed_ppf5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    SUBJECT_IDS,
    EVENT_TYPES,
    preprocess_all,
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
        description="Preprocess all fNIRS subjects: .snirf → .npy segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bids-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "BIDSdata_fNIRS_motion_artifact",
        help="Root of the BIDS dataset directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for .npy files.",
    )
    p.add_argument(
        "--ppf",
        type=float,
        default=6.0,
        help="Differential pathlength factor for Beer-Lambert law.",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        metavar="ID",
        default=None,
        help="Process only these subject IDs (e.g. sub-01 sub-02). Default: all 20.",
    )
    return p.parse_args()


# ── Detailed statistics display ───────────────────────────────────────────────

def print_detailed_stats(
    all_stats: dict[str, dict[str, list[int]]],
    output_dir: Path,
) -> None:
    """Print per-subject and aggregate statistics after preprocessing."""

    W = 72

    # ── Per-subject table ─────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  PER-SUBJECT SEGMENT COUNTS")
    print("=" * W)
    header = f"  {'Subject':<10}" + "".join(f"{et:>7}" for et in EVENT_TYPES) + f"  {'Total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for subject_id in sorted(all_stats.keys()):
        subj = all_stats[subject_id]
        counts = [len(subj.get(et, [])) for et in EVENT_TYPES]
        total  = sum(counts)
        row    = f"  {subject_id:<10}" + "".join(f"{c:>7}" for c in counts) + f"  {total:>7}"
        print(row)

    # Totals row
    grand = [sum(len(all_stats[s].get(et, [])) for s in all_stats) for et in EVENT_TYPES]
    print("  " + "-" * (len(header) - 2))
    grand_total = sum(grand)
    trow = f"  {'TOTAL':<10}" + "".join(f"{c:>7}" for c in grand) + f"  {grand_total:>7}"
    print(trow)
    print("=" * W)

    # ── Segment length table ──────────────────────────────────────────────────
    print()
    print(f"  {'Event':<6}  {'n_segs':>7}  {'n_samples':>10}  {'Duration':>10}  {'MB total':>9}")
    print("  " + "-" * 50)

    for et in EVENT_TYPES:
        lengths: list[int] = [
            l
            for s in all_stats.values()
            for l in s.get(et, [])
        ]
        if not lengths:
            continue
        unique = sorted(set(lengths))
        samp_str = str(unique[0]) if len(unique) == 1 else f"{min(lengths)}-{max(lengths)}"
        dur_s    = f"{unique[0]/10:.0f}s" if len(unique) == 1 else "variable"
        mb       = sum(16 * l * 4 for l in lengths) / 1e6
        print(f"  {et:<6}  {len(lengths):>7}  {samp_str:>10}  {dur_s:>10}  {mb:>9.2f}")

    print("  " + "-" * 50)
    all_lengths = [l for s in all_stats.values() for ls in s.values() for l in ls]
    total_mb    = sum(16 * l * 4 for l in all_lengths) / 1e6
    print(f"  {'ALL':<6}  {grand_total:>7}                          {total_mb:>9.2f}")

    # ── File inventory ────────────────────────────────────────────────────────
    print()
    print("  Saved files:")
    npy_files = sorted(output_dir.rglob("*.npy"))
    total_disk_mb = sum(f.stat().st_size for f in npy_files) / 1e6
    print(f"    {len(npy_files)} .npy files  |  {total_disk_mb:.1f} MB on disk")
    print(f"    Location: {output_dir.resolve()}")
    print("=" * W + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    subjects = args.subjects if args.subjects else SUBJECT_IDS
    n = len(subjects)

    print(f"\n{'='*55}")
    print(f"  fNIRS Preprocessing Pipeline — MA-CANet")
    print(f"{'='*55}")
    print(f"  Subjects   : {n}  ({subjects[0]} … {subjects[-1]})")
    print(f"  BIDS dir   : {args.bids_dir}")
    print(f"  Output dir : {args.output}")
    print(f"  PPF        : {args.ppf}")
    print(f"{'='*55}\n")

    t0 = time.time()
    all_stats = preprocess_all(
        bids_dir=args.bids_dir,
        output_dir=args.output,
        ppf=args.ppf,
        subjects=subjects,
    )
    elapsed = time.time() - t0

    if not all_stats:
        logger.error("No subjects processed successfully.")
        sys.exit(1)

    n_ok     = len(all_stats)
    n_failed = n - n_ok
    logger.info(
        "Done in %.1f s  |  %d/%d subjects OK  |  %d failed",
        elapsed, n_ok, n, n_failed,
    )

    print_detailed_stats(all_stats, args.output)


if __name__ == "__main__":
    main()
