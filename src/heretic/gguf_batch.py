# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch GGUF Q4_0 edits: patch many tensors with ONE atomic model copy."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .gguf_direct import (
    Q4_0_BLOCK_BYTES,
    decode_q4_0_blocks,
    encode_q4_0_blocks,
)


def patch_q4_0_manifest(
    source: str | Path,
    output: str | Path,
    *,
    manifest: str | Path,
    chunk_blocks: int = 8192,
) -> dict[str, object]:
    """Apply a list of \`{tensor, delta, scale}\` entries in a single transaction.

    All patches must target Q4_0 tensors. Delta shape MUST equal GGUF reader
    tensor shape (reversed GGUF dimensions) in contiguous memory order.
    """
    source, output, manifest = Path(source).resolve(), Path(output).resolve(), Path(manifest).resolve()
    if source == output:
        raise ValueError("Source and output cannot be the same")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("Install gguf: uv sync --extra qat_gguf") from exc

    edits = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(edits, list) or not edits:
        raise ValueError("Manifest must contain a nonempty list of edits")
    reader = GGUFReader(str(source))
    names = {t.name: t for t in reader.tensors}
    seen: set[str] = set()
    prepared: list[tuple[object, np.ndarray, float]] = []
    spans: list[tuple[int, int]] = []
    source_stat = source.stat()
    for item in edits:
        if not isinstance(item, dict):
            raise ValueError("Each edit must be an object")
        name = item.get("tensor")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Manifest tensor names must be nonempty and distinct")
        seen.add(name)
        tensor = names.get(name)
        if tensor is None or tensor.tensor_type.name != "Q4_0":
            raise ValueError(f"Not a Q4_0 GGUF tensor: {name}")
        dims = tuple(int(v) for v in reversed(tensor.shape))
        size = int(tensor.n_bytes)
        offset = int(tensor.data_offset)
        if (
            size < 18
            or size % Q4_0_BLOCK_BYTES
            or math.prod(dims) // 32 * 18 != size
            or math.prod(dims) % 32
            or offset < 0
            or offset + size > source_stat.st_size
        ):
            raise ValueError(f"Invalid Q4_0 span for {name}")
        delta_path = Path(item["delta"])
        if not delta_path.is_absolute():
            delta_path = manifest.parent / delta_path
        delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
        if (
            delta.shape != dims
            or delta.dtype not in (np.dtype("float16"), np.dtype("float32"))
            or not delta.flags.c_contiguous
        ):
            raise ValueError(f"Delta for {name} must have exact GGUF shape {dims}")
        scale = float(item.get("scale", 1.0))
        if not math.isfinite(scale):
            raise ValueError(f"Nonfinite scale for {name}")
        prepared.append((tensor, delta.reshape(-1), scale))
        spans.append((offset, offset + size))
    for left, right in zip(sorted(spans), sorted(spans)[1:]):
        if left[1] > right[0]:
            raise ValueError("Manifest patches overlap; refusing to corrupt GGUF")

    tmp_path: Path | None = None
    reports: list[dict[str, object]] = []
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".gguf-ara-batch-", suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, tmp, length=4 * 1024 * 1024)
            tmp.flush()
            os.fsync(tmp.fileno())
        with source.open("rb") as original, tmp_path.open("r+b") as destination:
            for tensor, delta, gain in prepared:
                size = int(tensor.n_bytes)
                offset = int(tensor.data_offset)
                blocks_total = size // 18
                blocks_changed = 0
                for start in range(0, blocks_total, chunk_blocks):
                    n = min(chunk_blocks, blocks_total - start)
                    original.seek(offset + start * 18)
                    raw = original.read(n * 18)
                    if len(raw) != n * 18:
                        raise IOError("Truncated source GGUF")
                    old = decode_q4_0_blocks(raw)
                    delta_flat = np.asarray(
                        delta[start * 32:(start + n) * 32], dtype=np.float32
                    ) * np.float32(gain)
                    if not np.isfinite(delta_flat).all():
                        raise ValueError(f"Invalid patch values for {tensor.name}")
                    update = delta_flat.reshape(n, 32)
                    target = old + update
                    if not np.isfinite(target).all():
                        raise ValueError(f"Overflow in patch for {tensor.name}")
                    encoded = np.frombuffer(
                        encode_q4_0_blocks(target), dtype=np.uint8
                    ).reshape(n, 18).copy()
                    original_bytes = np.frombuffer(
                        raw, dtype=np.uint8
                    ).reshape(n, 18)
                    unchanged = np.all(update == 0, axis=1)
                    encoded[unchanged] = original_bytes[unchanged]
                    blocks_changed += int(np.count_nonzero(
                        np.any(encoded != original_bytes, axis=1)
                    ))
                    destination.seek(offset + start * 18)
                    destination.write(encoded.tobytes())
                reports.append({
                    "tensor": tensor.name, "blocks_total": blocks_total,
                    "blocks_changed": blocks_changed, "scale": gain,
                })
            destination.flush()
            os.fsync(destination.fileno())
        changed = source.stat()
        if (
            source_stat.st_size != changed.st_size
            or source_stat.st_mtime_ns != changed.st_mtime_ns
        ):
            raise RuntimeError("Source changed during edit")
        if tmp_path.stat().st_size != source_stat.st_size:
            raise IOError("Edited GGUF size differs from original")
        os.link(tmp_path, output)
        return {
            "source": str(source), "output": str(output),
            "original_bytes": source_stat.st_size,
            "tensor_patches": reports,
        }
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
