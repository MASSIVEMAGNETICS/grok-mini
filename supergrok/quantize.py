"""
Quantize helper for checkpoints.

This module offers:
- quantize_checkpoint_bnb: use bitsandbytes if installed to produce 8-bit/4-bit weights (preferred)
- quantize_checkpoint_simple: fallback per-tensor quantize for storage only (not optimized inference)

Usage:
  from supergrok.quantize import quantize_checkpoint_bnb, quantize_checkpoint_simple
"""
import os
import torch

# Try to use bitsandbytes if available (recommended)
try:
    import bitsandbytes as bnb  # type: ignore
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

def quantize_checkpoint_bnb(input_path: str, output_path: str, dtype="8bit"):
    """
    Attempt to load a torch checkpoint and quantize linear weights using bitsandbytes.
    This function is a best-effort helper — using bitsandbytes' quantization patterns is recommended.
    """
    if not _HAS_BNB:
        raise RuntimeError("bitsandbytes not installed. Install via `pip install bitsandbytes` or use quantize_checkpoint_simple.")
    ckpt = torch.load(input_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    quant_state = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.dtype.is_floating_point:
            # Use 8bit quantization for large matrices
            try:
                q = bnb.nn.Int8Params(v.to(torch.float32), requires_grad=False, **{"desc_sorted": False})
                quant_state[k] = q.state_dict() if hasattr(q, "state_dict") else v.to(torch.float16)
            except Exception:
                quant_state[k] = v.to(torch.float16)
        else:
            quant_state[k] = v
    torch.save({"model_state": quant_state, "meta": ckpt.get("meta", {})}, output_path)
    return output_path

def quantize_checkpoint_simple(input_path: str, output_path: str, bits: int = 8):
    """
    Simple per-tensor quantization: quantize floating tensors to int8 storage via PyTorch quantize_per_tensor.
    This is intended for disk storage reduction, not optimized inference.
    """
    ckpt = torch.load(input_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    qstate = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            # compute scale/zero_point from tensor statistics for better quantization
            try:
                v_min, v_max = v.min().item(), v.max().item()
                scale = (v_max - v_min) / 255.0 if v_max != v_min else 1.0
                zero_point = 0
                q = torch.quantize_per_tensor(v.contiguous(), scale=scale, zero_point=zero_point, dtype=torch.qint8)
                qstate[k] = {"q_tensor": q.int_repr(), "scale": q.q_scale(), "zero_point": q.q_zero_point(), "shape": list(v.shape)}
            except Exception:
                qstate[k] = {"float": v.to(torch.float16)}
        else:
            qstate[k] = {"raw": v}
    torch.save({"model_state": qstate, "meta": ckpt.get("meta", {})}, output_path)
    return output_path
