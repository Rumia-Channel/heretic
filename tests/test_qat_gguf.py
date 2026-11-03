# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only GGUF Q4_0 inspection tests without a multi-GB checkpoint."""

import hashlib
from types import SimpleNamespace

import pytest

from heretic.qat_gguf import sha256_file, summarize_gguf_reader


def _reader(types):
    tensors = [
        SimpleNamespace(tensor_type=SimpleNamespace(name=dtype), n_bytes=size)
        for dtype, size in types
    ]
    return SimpleNamespace(
        fields={
            "general.architecture": SimpleNamespace(contents=lambda: "gemma4")
        },
        tensors=tensors,
    )


def test_gguf_summary_tracks_mixed_storage_types():
    report = summarize_gguf_reader(
        _reader([("Q4_0", 1000), ("Q4_0", 900), ("F32", 200)])
    )
    assert report["architecture"] == "gemma4"
    assert report["tensor_count"] == 3
    assert report["q4_0_tensors"] == 2
    assert report["other_tensors"] == 1
    assert report["quantization_types"] == {"F32": 1, "Q4_0": 2}
    assert report["total_tensor_storage_bytes"] == 2100


def test_gguf_summary_refuses_non_q4_0():
    with pytest.raises(ValueError, match="No Q4_0"):
        summarize_gguf_reader(_reader([("Q5_0", 111)]))


def test_gguf_summary_refuses_empty():
    with pytest.raises(ValueError, match="no tensors"):
        summarize_gguf_reader(_reader([]))


def test_sha256_streaming(tmp_path):
    path = tmp_path / "tiny.gguf"
    payload = b"synthetic QAT test file"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=3) == hashlib.sha256(payload).hexdigest()


def test_sha256_rejects_invalid_chunk(tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"anything")
    with pytest.raises(ValueError):
        sha256_file(path, chunk_size=0)
