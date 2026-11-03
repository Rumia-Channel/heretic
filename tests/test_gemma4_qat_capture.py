# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma-only activation-to-ARA bridge tests, using a tiny synthetic GGUF."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from heretic.gguf_direct import encode_q4_0_blocks
from gemma4_qat_capture.prepare_ara_io import prepare_io


def _inputs(tmp_path, monkeypatch, *, architecture="gemma4", width=32):
    original = tmp_path / "original.gguf"
    base = np.zeros((2, 32), dtype=np.float32)
    base[0, 0] = -8
    base[1, 0] = 7
    header = b"test-GGUF-header"
    raw = encode_q4_0_blocks(base)
    original.write_bytes(header + raw)
    tensor = SimpleNamespace(
        name="blk.0.attn_output.weight",
        tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32, 2], dtype=np.uint64),
        n_bytes=len(raw),
        data_offset=len(header),
    )

    class Reader:
        def __init__(self, _):
            self.fields = {
                "general.architecture": SimpleNamespace(
                    contents=lambda: architecture
                )
            }
            self.tensors = [tensor]

    gguf = ModuleType("gguf")
    gguf.GGUFReader = Reader
    monkeypatch.setitem(sys.modules, "gguf", gguf)
    captured = tmp_path / "captured"
    captured.mkdir()
    (captured / "capture.txt").write_text(
        "format=gemma4.kqv_out.last_prompt_token.f32.v1\n"
        "good_count=2\nbad_count=2\nlayer=0,features=32\n",
        encoding="utf-8",
    )
    gx = np.zeros((2, width), dtype="<f4")
    bx = np.ones((2, width), dtype="<f4")
    gx[:, 0] = 1
    gx.tofile(captured / "layer-0.good.f32")
    bx.tofile(captured / "layer-0.bad.f32")
    return original, captured, tmp_path / "io", base


def test_prepares_real_quantized_projection_reference(monkeypatch, tmp_path):
    model, captured, dest, base = _inputs(tmp_path, monkeypatch)
    report = prepare_io(model, captured, dest, device="cpu")
    assert report["architecture"] == "gemma4"
    assert report["good_count"] == report["bad_count"] == 2
    assert report["source_sha256"]
    with np.load(dest / "layer-0.npz") as bundle:
        np.testing.assert_allclose(bundle["good_y"], bundle["good_x"] @ base.T)
        np.testing.assert_allclose(bundle["bad_y"], bundle["bad_x"] @ base.T)
        assert bundle["tensor_name"].item() == "blk.0.attn_output.weight"
    targets = json.loads((dest / "ara_targets.json").read_text())
    assert targets == [{
        "tensor": "blk.0.attn_output.weight",
        "module_io": "layer-0.npz",
    }]


def test_rejects_wrong_layer_width(monkeypatch, tmp_path):
    model, captured, dest, _ = _inputs(tmp_path, monkeypatch, width=31)
    with pytest.raises(ValueError, match="Incorrect good activation size"):
        prepare_io(model, captured, dest, device="cpu")
    assert not dest.exists()


def test_rejects_wrong_architecture(monkeypatch, tmp_path):
    model, captured, dest, _ = _inputs(tmp_path, monkeypatch, architecture="llama")
    with pytest.raises(ValueError, match="ONLY for Gemma 4"):
        prepare_io(model, captured, dest, device="cpu")
    assert not dest.exists()
