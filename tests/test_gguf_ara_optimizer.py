# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local ARA optimizer smoke tests, using a synthetic GGUF tensor and CPU."""

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from heretic.gguf_ara_optimizer import Q4ARAOptions, optimise_ara_tensor_on_q4_0
from heretic.gguf_direct import encode_q4_0_blocks


def _install_reader(monkeypatch, offset, nbytes):
    tensor = SimpleNamespace(
        name="blk.0.attn_output.weight",
        tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32, 2], dtype=np.uint64),
        n_elements=64,
        n_bytes=nbytes,
        data_offset=offset,
    )

    class FakeReader:
        def __init__(self, _path):
            self.tensors = [tensor]

    module = ModuleType("gguf")
    module.GGUFReader = FakeReader
    monkeypatch.setitem(sys.modules, "gguf", module)
    return tensor


def test_q4_ara_local_optimiser_writes_gguf_and_keeps_source(monkeypatch, tmp_path):
    values = np.zeros((2, 32), dtype=np.float32)
    values[0, 0] = -8.0
    values[1, 3] = 5.0
    encoded = encode_q4_0_blocks(values)
    prefix = b"GGUF-fake-format"
    original = prefix + encoded + b"other-tensor"
    source, destination = tmp_path / "original.gguf", tmp_path / "candidate.gguf"
    source.write_bytes(original)
    _install_reader(monkeypatch, len(prefix), len(encoded))
    rng = np.random.default_rng(123)
    gx = rng.normal(size=(6, 32)).astype(np.float32)
    bx = rng.normal(size=(4, 32)).astype(np.float32)
    gy, by = gx @ values.T, bx @ values.T
    io_path = tmp_path / "io.npz"
    np.savez(io_path, good_x=gx, good_y=gy, bad_x=bx, bad_y=by)
    options = Q4ARAOptions(steps=2, learning_rate=0.005, device="cpu")
    report = optimise_ara_tensor_on_q4_0(
        source, destination, tensor_name="blk.0.attn_output.weight",
        module_io_npz=io_path, options=options,
    )
    assert destination.exists()
    assert source.read_bytes() == original
    assert math.isfinite(report["ara_final_loss"])
    assert report["blocks_total"] == 2


def test_missing_io_fails_before_creating_gguf(monkeypatch, tmp_path):
    values = np.zeros((2, 32), dtype=np.float32)
    encoded = encode_q4_0_blocks(values)
    source, destination = tmp_path / "original.gguf", tmp_path / "candidate.gguf"
    source.write_bytes(b"header" + encoded)
    _install_reader(monkeypatch, 6, len(encoded))
    io_path = tmp_path / "missing.npz"
    np.savez(io_path, good_x=np.ones((1, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="needs good_x"):
        optimise_ara_tensor_on_q4_0(
            source, destination, tensor_name="blk.0.attn_output.weight",
            module_io_npz=io_path, options=Q4ARAOptions(steps=1, device="cpu"),
        )
    assert not destination.exists()
