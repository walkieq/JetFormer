import torch
from collections import defaultdict
from typing import Dict, Tuple
import os
import sys
from bitnet import BitLinear

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.bit_model import JetFormer


def bytes_of_tensor(t: torch.Tensor) -> int:
    # Use fp32
    return t.numel() * 4


def is_bitlinear_param(module: torch.nn.Module, name: str) -> bool:
    # Only calculate weights in BitLinear, BitLinear does not use bias
    return isinstance(module, BitLinear) and name.endswith("weight")


def get_model_size_quantized(
    model: torch.nn.Module, include_buffers: bool = True, return_breakdown: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Estimate quantized model size in KB, assuming BitLinear weights are 1-bit.
    Args:
        model: the model to estimate size
        include_buffers: whether to include buffers (e.g. BatchNorm running mean/var)
        return_breakdown: whether to return breakdown of different components
    Returns:
        size_kb: estimated model size in KB
        breakdown_kb: breakdown of different components in KB"""
    total_bytes = 0
    breakdown = defaultdict(int)

    # Get all modules and their submodules
    for module_prefix, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if p.numel() == 0:
                continue

            if is_bitlinear_param(module, name):
                # Convert to storage size in bytes
                w_bytes = (p.numel() + 7) // 8
                total_bytes += w_bytes
                breakdown["bitlinear_weight(1bit)"] += w_bytes
            elif isinstance(module, BitLinear) and name.endswith("bias"):
                # Ignore bias in BitLinear
                continue
            else:
                b = bytes_of_tensor(p)
                total_bytes += b
                breakdown["other_params_fp32"] += b

        #  E.g. BatchNorm running mean/var
        if include_buffers:
            for bname, buf in module.named_buffers(recurse=False):
                if buf.numel() == 0:
                    continue
                b = bytes_of_tensor(buf)
                total_bytes += b
                breakdown["buffers_fp32"] += b

    size_kb = total_bytes / 1024

    if return_breakdown:
        # Convert to KB
        breakdown_kb = {k: v / 1024 for k, v in breakdown.items()}
        return size_kb, breakdown_kb
    else:
        return size_kb, {}


def get_model_size_fp32(
    model: torch.nn.Module, include_buffers: bool = True, return_breakdown: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Estimate original model size in KB, assuming all parameters are fp32.
    Args:
        model: the model to estimate size
        include_buffers: whether to include buffers (e.g. BatchNorm running mean/var)
        return_breakdown: whether to return breakdown of different components
    Returns:
        size_kb: estimated model size in KB
        breakdown_kb: breakdown of different components in KB"""
    total_bytes = 0
    breakdown = defaultdict(int)

    for module_prefix, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if p.numel() == 0:
                continue
            b = bytes_of_tensor(p)
            total_bytes += b
            breakdown["params_fp32"] += b

        # buffers
        if include_buffers:
            for bname, buf in module.named_buffers(recurse=False):
                if buf.numel() == 0:
                    continue
                b = bytes_of_tensor(buf)
                total_bytes += b
                breakdown["buffers_fp32"] += b

    size_kb = total_bytes / 1024
    if return_breakdown:
        breakdown_kb = {k: v / 1024 for k, v in breakdown.items()}
        return size_kb, breakdown_kb
    else:
        return size_kb, {}


if __name__ == "__main__":
    num_particles = 32
    num_feats = 3

    num_transformers = 3
    embbed_dim = 64
    num_heads = 2
    dropout = 0

    model_config = {
        "in_dim": num_feats,
        "embbed_dim": embbed_dim,
        "num_heads": num_heads,
        "num_classes": 5,
        "num_transformers": num_transformers,
        "dropout": dropout,
        "num_particles": num_particles,
        "activation": "ReLU",
        "normalization": "Batch",
    }
    model = JetFormer(**model_config)

    size_kb_fp32, details_fp32 = get_model_size_fp32(model)
    print(f"[JetFormer estimated original size] {size_kb_fp32:.3f} KB")
    for k, v in sorted(details_fp32.items(), key=lambda x: -x[1]):
        print(f"  - {k:26s}: {v:.3f} KB")

    size_kb_quantized, details_quantized = get_model_size_quantized(model)
    print(f"[JetFormer estimated quantized size] {size_kb_quantized:.3f} KB")
    for k, v in sorted(details_quantized.items(), key=lambda x: -x[1]):
        print(f"  - {k:26s}: {v:.3f} KB")

    reduction_pct = 100 * (1 - size_kb_quantized / size_kb_fp32)
    print(f"Model size reduction: {reduction_pct:.2f}%")
