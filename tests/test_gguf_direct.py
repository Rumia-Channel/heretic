# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct GGUF byte-level patch tests using a tiny synthetic Q4_0 payload."""

import hashlib
import struct
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from heretic.gguf_direct import (
    decode_q4_0_blocks,
    encode_q4_0_blocks,
    patch_q4_0_tensor,
)


def _install_fake_gguf(monkeypatch, tensor, extra=()):
    class Reader:
        def __init__(self, _path):
            self.tensors = [tensor, *extra]

    module = ModuleType("gguf")
    module.GGUFReader = Reader
    monkeypatch.setitem(sys.modules, "gguf", module)


def test_q4_0_nibble_order_decodes_second_half():
    payload = struct.pack("<e", 1.0) + bytes([0xF0] * 16)
    result = decode_q4_0_blocks(payload)
    np.testing.assert_array_equal(result[0, :16], [-8.0] * 16)
    np.testing.assert_array_equal(result[0, 16:], [7.0] * 16)


@pytest.mark.parametrize("shape", [(1, 31), (32,), (0, 32), (2, 33)])
def test_invalid_quantization_shapes(shape):
    with pytest.raises(ValueError):
        encode_q4_0_blocks(np.zeros(shape, dtype=np.float32))


def test_reference_quantization_stores_signed_scale_and_nibbles():
    values = np.zeros((1, 32), dtype=np.float32)
    values[0, 0] = -8.0
    result = encode_q4_0_blocks(values)
    assert result[:2] == struct.pack("<e", 1.0)
    assert result[2] & 0x0F == 0
    assert result[2] >> 4 == 8
    actual = decode_q4_0_blocks(result)
    np.testing.assert_array_equal(actual[0, :1], [-8.0])


def test_patch_only_target_bytes_and_preserves_original(monkeypatch, tmp_path):
    src = tmp_path / "base.gguf"
    dst = tmp_path / "edited.gguf"
    prefix, suffix = b"GGUF-test-header", b"untouched-other-tensor"
    original_values = np.zeros((2, 32), dtype=np.float32)
    original_values[0, 0], original_values[1, 0] = -8.0, -4.0
    old_bytes = encode_q4_0_blocks(original_values)
    src.write_bytes(prefix + old_bytes + suffix)
    tensor = SimpleNamespace(
        name="blk.0.attn_output.weight",
        tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32, 2], dtype=np.uint64),
        n_elements=64,
        n_bytes=len(old_bytes),
        data_offset=len(prefix),
    )
    _install_fake_gguf(monkeypatch, tensor)
    delta_path = tmp_path / "delta.npy"
    delta = np.zeros((2, 32), dtype=np.float32)
    delta[0, 1] = -2.0
    np.save(delta_path, delta)
    original_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    report = patch_q4_0_tensor(
        src, dst, tensor_name=tensor.name, delta_npy=delta_path,
        expected_source_sha256=original_hash, chunk_blocks=1,
    )
    assert src.read_bytes() == prefix + old_bytes + suffix
    modified = dst.read_bytes()
    assert modified[:len(prefix)] == prefix
    assert modified[-len(suffix):] == suffix
    assert report["blocks_changed"] == 1
    assert modified[len(prefix) + 18:len(prefix) + 36] == old_bytes[18:]
    assert modified[len(prefix):len(prefix) + 18] != old_bytes[:18]
    with pytest.raises(FileExistsError):
        patch_q4_0_tensor(
            src, dst, tensor_name=tensor.name, delta_npy=delta_path
        )


def test_bad_delta_rejected_without_writing_output(monkeypatch, tmp_path):
    src, dst = tmp_path / "a.gguf", tmp_path / "b.gguf"
    prefix = b"GGUF-test-header"
    raw = encode_q4_0_blocks(np.zeros((1, 32), dtype=np.float32))
    src.write_bytes(prefix + raw)
    tensor = SimpleNamespace(
        name="test.weight",
        tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32], dtype=np.uint64),
        n_elements=32, n_bytes=18, data_offset=len(prefix),
    )
    _install_fake_gguf(monkeypatch, tensor)
    delta_path = tmp_path / "bad.npy"
    delta = np.zeros(32, dtype=np.float32)
    delta[0] = float("nan")
    np.save(delta_path, delta)
    with pytest.raises(ValueError, match="Non-finite"):
        patch_q4_0_tensor(
            src, dst, tensor_name=tensor.name, delta_npy=delta_path
        )
    assert not dst.exists()


def test_wrong_tensor_type_rejected(monkeypatch, tmp_path):
    src, dst = tmp_path / "a.gguf", tmp_path / "b.gguf"
    src.write_bytes(bytes(512))
    tensor = SimpleNamespace(
        name="test.weight",
        tensor_type=SimpleNamespace(name="Q5_0"),
        shape=np.array([32], dtype=np.uint64),
        n_elements=32, n_bytes=18, data_offset=64,
    )
    _install_fake_gguf(monkeypatch, tensor)
    delta_path = tmp_path / "delta.npy"
    np.save(delta_path, np.zeros((32,), dtype=np.float32))
    with pytest.raises(ValueError, match="not Q4_0"):
        patch_q4_0_tensor(
            src, dst, tensor_name=tensor.name, delta_npy=delta_path
        )
    assert not dst.exists()
