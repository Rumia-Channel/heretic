# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q4_0 multi-layer ARA provenance and atomic output tests (fake GGUF)."""

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from heretic.gguf_ara_batch_optimizer import optimise_ara_q4_manifest
from heretic.gguf_ara_optimizer import Q4ARAOptions
from heretic.gguf_direct import encode_q4_0_blocks


def _fake_reader(monkeypatch, tensors):
    class Reader:
        def __init__(self, _path):
            self.tensors = tensors

    mod = ModuleType("gguf")
    mod.GGUFReader = Reader
    monkeypatch.setitem(sys.modules, "gguf", mod)


def _fixture(tmp_path, monkeypatch):
    before = b"test-gguf-header"
    weights = np.zeros((2, 32), dtype=np.float32)
    weights[0, 0], weights[1, 1] = -8.0, 5.0
    blob = encode_q4_0_blocks(weights)
    source = tmp_path / "source.gguf"
    source.write_bytes(before + blob)
    name = "blk.0.attn_output.weight"
    tensor = SimpleNamespace(
        name=name,
        tensor_type=SimpleNamespace(name="Q4_0"),
        shape=np.array([32, 2], dtype=np.uint64),
        n_elements=64,
        n_bytes=len(blob),
        data_offset=len(before),
    )
    _fake_reader(monkeypatch, [tensor])
    capture = tmp_path / "capture.npz"
    good_x = np.ones((3, 32), dtype=np.float32)
    bad_x = np.full((3, 32), 0.5, dtype=np.float32)
    np.savez(
        capture,
        good_x=good_x, good_y=good_x @ weights.T,
        bad_x=bad_x, bad_y=bad_x @ weights.T,
        source_sha256=np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
        tensor_name=np.array(name),
    )
    manifest = tmp_path / "targets.json"
    manifest.write_text(json.dumps([
        {"tensor": name, "module_io": "capture.npz"}
    ]), encoding="utf-8")
    return source, tmp_path / "out.gguf", manifest, capture


def test_multi_layer_optimizer_publishes_gguf(monkeypatch, tmp_path):
    source, output, manifest, _ = _fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    report = optimise_ara_q4_manifest(
        source, output, manifest,
        options=Q4ARAOptions(device="cpu", steps=2, learning_rate=0.001),
    )
    assert source.read_bytes() == original
    assert output.is_file()
    assert len(output.read_bytes()) == len(original)
    assert len(report["q4_0_patches"]) == 1
    assert len(report["ara_layers"]) == 1


def test_wrong_checkpoint_identity_rejected(monkeypatch, tmp_path):
    source, output, manifest, capture = _fixture(tmp_path, monkeypatch)
    with np.load(capture, allow_pickle=False) as bundle:
        arr = {key: bundle[key] for key in bundle.files}
    arr["source_sha256"] = np.array("bad-hash")
    np.savez(capture, **arr)
    with pytest.raises(ValueError, match="provenance"):
        optimise_ara_q4_manifest(
            source, output, manifest,
            options=Q4ARAOptions(device="cpu", steps=1),
        )
    assert not output.exists()


def test_wrong_tensor_name_rejected(monkeypatch, tmp_path):
    source, output, manifest, capture = _fixture(tmp_path, monkeypatch)
    with np.load(capture, allow_pickle=False) as bundle:
        arr = {key: bundle[key] for key in bundle.files}
    arr["tensor_name"] = np.array("another.tensor")
    np.savez(capture, **arr)
    with pytest.raises(ValueError, match="provenance"):
        optimise_ara_q4_manifest(
            source, output, manifest,
            options=Q4ARAOptions(device="cpu", steps=1),
        )
    assert not output.exists()
