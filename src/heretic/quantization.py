# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference Q4_0 fake quantization for QAT compatibility experiments.

This is an experimental, weight-only *simulation*, not a GGUF writer and not a
replacement for ggml/llama.cpp validation. In particular, GGUF tensor layout,
per-tensor quantization choices, and model-wide quality must be checked against
an actual GGUF produced by the deployment toolchain.
"""

from __future__ import annotations

import torch
from torch import Tensor

Q4_0_BLOCK_SIZE = 32


def dequantize_after_q4_0(weight: Tensor) -> Tensor:
    """Simulate ggml's reference Q4_0 quantize+dequantize on the last dimension.

    The signed value with the largest magnitude determines a *signed* scale.
    Quantized nibbles have values 0..15 and decode as ``(q - 8) * scale``.
    Scales are rounded to fp16 for storage, while the nibble calculation uses
    the unrounded float32 scale, as in ``quantize_row_q4_0_ref``.

    This function is intentionally non-differentiable; use
    :func:`fake_quantize_q4_0` to approximate the backward pass.
    """
    if not isinstance(weight, Tensor) or not weight.is_floating_point():
        raise TypeError("Q4_0 simulation requires a floating-point tensor")
    if weight.ndim < 1 or weight.shape[-1] == 0 or weight.shape[-1] % Q4_0_BLOCK_SIZE:
        raise ValueError("last tensor dimension must be a nonzero multiple of 32")
    if not torch.isfinite(weight).all():
        raise ValueError("Q4_0 simulation requires finite input weights")

    with torch.no_grad():
        blocks = weight.detach().to(torch.float32).reshape(
            *weight.shape[:-1], -1, Q4_0_BLOCK_SIZE
        )
        max_index = blocks.abs().argmax(dim=-1, keepdim=True)
        signed_max = blocks.gather(-1, max_index)
        scale = signed_max / -8.0
        inverse_scale = torch.where(scale == 0, 0.0, 1.0 / scale)
        # C reference uses int8_t(x / d + 8.5f) (truncation toward zero).
        # This quantity is nonnegative for finite inputs within each block.
        quantized = torch.clamp(
            torch.trunc(blocks * inverse_scale + 8.5), 0, 15
        )
        stored_scale = scale.to(torch.float16).to(torch.float32)
        dequantized = (quantized - 8.0) * stored_scale
        return dequantized.reshape(weight.shape).to(weight.dtype)


def fake_quantize_q4_0(weight: Tensor) -> Tensor:
    """Q4_0 forward with identity straight-through estimated gradients.

    STE provides a surrogate gradient, *not* the real gradient of Q4_0.
    It should be evaluated with the final GGUF runtime before adoption.
    """
    quantized = dequantize_after_q4_0(weight)
    return weight + (quantized - weight).detach()
