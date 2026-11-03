# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic multi-tensor GGUF transaction tests."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from heretic.gguf_batch import patch_q4_0_manifest
from heretic.gguf_direct import encode_q4_0_blocks


def _reader(monkeypatch, specs):
    class Reader:
        def __init__(self, _path):
            self.tensors = specs

    gguf = ModuleType("gguf")
    gguf.GGUFReader = Reader
    monkeypatch.setitem(sys.modules, "gguf", gguf)


def test_two_edits_one_gguf_and_preserved_other_bytes(tmp_path, monkeypatch):
    start, tail = b"GGUF-header", b"untouched-tail"
    raw1 = encode_q4_0_blocks(np.array([[-8.0] + [0] * 31], dtype=np.float32))
    raw2 = encode_q4_0_blocks(np.array([[0, -8.0] + [0] * 30], dtype=np.float32))
    original = start + raw1 + b"aligned-space" + raw2 + tail
    source = tmp_path / "base.gguf"
    output = tmp_path / "edited.gguf"
    source.write_bytes(original)
    offset2 = len(start) + len(raw1) + len(b"aligned-space")
    specs = [
        SimpleNamespace(
            name="blk.0.attn_output.weight", tensor_type=SimpleNamespace(name="Q4_0"),
            shape=np.array([32]), n_bytes=len(raw1), data_offset=len(start),
        ),
        SimpleNamespace(
            name="blk.1.attn_output.weight", tensor_type=SimpleNamespace(name="Q4_0"),
            shape=np.array([32]), n_bytes=len(raw2), data_offset=offset2,
        ),
    ]
    _reader(monkeypatch, specs)
    d1 = np.zeros(32, dtype=np.float32)
    d1[0] = -1
    d2 = np.zeros(32, dtype=np.float32)
    d2[1] = -2
    np.save(tmp_path / "delta1.npy", d1)
    np.save(tmp_path / "delta2.npy", d2)
    manifest = tmp_path / "patches.json"
    manifest.write_text(json.dumps([
        {"tensor": specs[0].name, "delta": "delta1.npy"},
        {"tensor": specs[1].name, "delta": "delta2.npy"},
    ]), encoding="utf-8")
    report = patch_q4_0_manifest(source, output, manifest=manifest)
    assert len(report["tensor_patches"]) == 2
    assert source.read_bytes() == original
    after = output.read_bytes()
    assert len(after) == len(original)
    assert after[:len(start)] == start
    assert after[-len(tail):] == tail
    assert after[len(start) + 18:offset2] == b"aligned-space"


def test_manifest_duplicate_tensor_rejected(tmp_path, monkeypatch):
    tensor = SimpleNamespace(
        name="t", tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32]), n_bytes=18, data_offset=0,
    )
    _reader(monkeypatch, [tensor])
    source, output = tmp_path / "original.gguf", tmp_path / "new.gguf"
    source.write_bytes(encode_q4_0_blocks(np.zeros((1, 32), dtype=np.float32)))
    np.save(tmp_path / "d.npy", np.zeros(32, dtype=np.float32))
    manifest = tmp_path / "patches.json"
    manifest.write_text(json.dumps([
        {"tensor": "t", "delta": "d.npy"},
        {"tensor": "t", "delta": "d.npy"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        patch_q4_0_manifest(source, output, manifest=manifest)
    assert not output.exists()
