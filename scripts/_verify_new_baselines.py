"""验证新增基线模型的参数量是否符合预期。"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import torch
from src.models.baselines import DAENet, EnhancedDAE, SmallDAE, CNNwP, LSTMAutoencoder
from src.models.macanet import MACANet


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_forward(model, name):
    """验证 forward 通路。"""
    model.eval()
    x = torch.randn(2, 1, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 512), f"{name} 输出 shape 错误: {y.shape}"
    return y.shape


if __name__ == "__main__":
    print("=" * 70)
    print("新基线模型参数量与 forward 通路验证")
    print("=" * 70)

    models = {
        "MA-CANet (本文方法)":  MACANet(),
        "DAE (旧, ~22K)":       DAENet(),
        "EnhancedDAE (300K)":   EnhancedDAE(),
        "SmallDAE-60K":         SmallDAE(base_channels=16),
        "SmallDAE-120K":        SmallDAE(base_channels=24),
        "CNNwP (Huang 2024)":   CNNwP(window_size=512),
        "LSTM-AE (Yang 2025)":  LSTMAutoencoder(),
    }

    print(f"\n{'模型':<30} {'参数量':>12}  {'输出 shape':<15}")
    print("-" * 70)
    all_pass = True
    for name, model in models.items():
        try:
            params = count_params(model)
            shape = test_forward(model, name)
            print(f"{name:<30} {params:>12,}  {str(shape):<15}")
        except Exception as e:
            print(f"{name:<30} 错误: {e}")
            all_pass = False

    print("\n" + "=" * 70)
    print("目标参数量参考：")
    print("  - MA-CANet: 320,801")
    print("  - EnhancedDAE: 280K~340K 为合格")
    print("  - SmallDAE-60K: 50K~70K")
    print("  - SmallDAE-120K: 100K~140K")
    print("  - CNNwP: 250K~400K 为合格")
    print("  - LSTM-AE: 250K~350K 为合格")
    print("=" * 70)
    if all_pass:
        print("\n[OK] 所有模型 forward 通路验证通过")
    else:
        print("\n[FAIL] 部分模型验证失败，请检查上方错误信息")
