"""Exact per-layer parameter count for MA-CANet (for Table architecture summary)."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.macanet import MACANet, MACANetAblation


def section_params(model, prefix_filter):
    return sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and name.startswith(prefix_filter)
    )


print("=" * 65)
print("  MA-CANet Per-Module Parameter Count")
print("=" * 65)

model = MACANet()
model.eval()

modules = {
    "stem (MS-Conv)": "stem",
    "encoders[0]":    "encoders.0",
    "encoders[1]":    "encoders.1",
    "encoders[2]":    "encoders.2",
    "encoders[3]":    "encoders.3",
    "bottleneck":     "bottleneck",
    "decoders[0]":    "decoders.0",
    "decoders[1]":    "decoders.1",
    "decoders[2]":    "decoders.2",
    "decoders[3]":    "decoders.3",
    "output_conv":    "output_conv",
}

total_counted = 0
for label, prefix in modules.items():
    n = section_params(model, prefix)
    total_counted += n
    print(f"  {label:<22}  {n:>8,}")

total_model = model.count_parameters()
print(f"\n  {'Sum of sections':<22}  {total_counted:>8,}")
print(f"  {'Model total':<22}  {total_model:>8,}")
assert total_counted == total_model, f"Mismatch: {total_counted} != {total_model}"
print("  [PASS] Sum == total")

# Detailed breakdown
print("\n  Detailed sub-module params:")
for name, module in model.named_modules():
    p = sum(x.numel() for x in module.parameters() if x.requires_grad)
    # Only leaf-ish: skip top-level and go one level deeper than top
    depth = name.count(".")
    if p > 0 and depth == 1:
        print(f"    {name:<40}  {p:>8,}")

# Ablation counts
print("\n  Ablation variant parameter counts:")
print(f"  {'ID':<6}  {'Params':>8}")
print("  " + "-" * 18)
for aid in ["A1", "A2", "A3", "A4", "A5"]:
    m = MACANetAblation(ablation_id=aid)
    n = m.count_parameters()
    print(f"  {aid:<6}  {n:>8,}")

print("=" * 65)
