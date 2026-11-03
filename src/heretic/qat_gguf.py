# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only inspection of original QAT GGUF tensor types.

The source artifact is never modified. This reports what is actually stored,
rather than assuming a GGUF marked Q4_0 contains *only* Q4_0 tensors.
GGUF tensor inference or gradient-based editing is not implemented here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_gguf_reader(reader: Any) -> dict[str, object]:
    """Summarise a gguf.GGUFReader; also accepts a fake reader in unit tests."""
    fields = reader.fields
    arch_field = fields.get("general.architecture")
    architecture = arch_field.contents() if arch_field is not None else None
    types: Counter[str] = Counter()
    total_bytes = 0
    for tensor in reader.tensors:
        dtype_name = tensor.tensor_type.name
        types[dtype_name] += 1
        total_bytes += int(tensor.n_bytes)
    if not types:
        raise ValueError("GGUF contains no tensors")
    if types["Q4_0"] == 0:
        raise ValueError(
            "No Q4_0 tensors found; refusing to treat this artifact as QAT Q4_0"
        )
    return {
        "architecture": architecture,
        "tensor_count": sum(types.values()),
        "quantization_types": dict(sorted(types.items())),
        "q4_0_tensors": types["Q4_0"],
        "other_tensors": sum(types.values()) - types["Q4_0"],
        "total_tensor_storage_bytes": total_bytes,
    }


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_qat_gguf(path: str | Path, *, hash_source: bool = False) -> dict[str, object]:
    """Inspect one GGUF file through gguf's memory-mapped reference reader."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "Install gguf to inspect a GGUF: uv sync --extra qat_gguf"
        ) from exc

    reader = GGUFReader(str(path))
    summary = summarize_gguf_reader(reader)
    summary["source_path"] = str(path.resolve())
    summary["source_size_bytes"] = path.stat().st_size
    if hash_source:
        summary["source_sha256"] = sha256_file(path)
    return summary
