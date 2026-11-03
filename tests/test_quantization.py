# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the provisional Q4_0 reference implementation."""

import math

import pytest
import torch

from heretic.quantization import (
    Q4_0_BLOCK_SIZE,
    dequantize_after_q4_0,
    fake_quantize_q4_0,
)


def _scalar_reference(block):
    """Independent scalar translation of ggml quantize_row_q4_0_ref."""
    extreme = max(block, key=lambda v: abs(v))
    scale = extreme / -8.0
    inverse = 1.0 / scale if scale else 0.0
    stored_scale = torch.tensor(scale, dtype=torch.float16).item()
    result = []
    for value in block:
        q = min(15, int(value * inverse + 8.5))
        result.append((q - 8) * stored_scale)
    return result


def test_block_size_is_32():
    assert Q4_0_BLOCK_SIZE == 32


def test_matches_scalar_reference_for_positive_and_negative_extremes():
    blocks = torch.tensor(
        [[-6.0] + [i / 13.0 for i in range(1, 32)],
         [7.0] + [-i / 9.0 for i in range(1, 32)]],
        dtype=torch.float32,
    )
    expected = torch.tensor([_scalar_reference(row.tolist()) for row in blocks])
    actual = dequantize_after_q4_0(blocks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_block_and_multiple_blocks():
    x = torch.zeros(2, 64, dtype=torch.float32)
    assert torch.count_nonzero(dequantize_after_q4_0(x)) == 0
    assert dequantize_after_q4_0(x).shape == (2, 64)


def test_scale_is_stored_as_float16_not_full_precision():
    x = torch.zeros(32)
    x[0] = -3.123456
    x[1] = 0.9
    result = dequantize_after_q4_0(x)
    expected = _scalar_reference(x.tolist())
    assert result[0].item() == pytest.approx(expected[0], rel=0, abs=1e-7)


def test_fake_quant_has_identity_gradient():
    weight = torch.linspace(-2.0, 3.0, 32, requires_grad=True)
    out = fake_quantize_q4_0(weight)
    torch.testing.assert_close(out.detach(), dequantize_after_q4_0(weight))
    (out * torch.arange(1, 33, dtype=torch.float32)).sum().backward()
    torch.testing.assert_close(weight.grad, torch.arange(1, 33, dtype=torch.float32))


@pytest.mark.parametrize("bad", [torch.ones(31), torch.empty(0), torch.tensor(1.0)])
def test_invalid_shapes_rejected(bad):
    with pytest.raises(ValueError):
        dequantize_after_q4_0(bad)


def test_nonfinite_input_rejected():
    x = torch.zeros(32)
    x[0] = math.nan
    with pytest.raises(ValueError):
        dequantize_after_q4_0(x)


def test_integer_input_rejected():
    with pytest.raises(TypeError):
        dequantize_after_q4_0(torch.arange(32))
