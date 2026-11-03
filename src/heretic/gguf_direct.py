# SPDX-License-Identifier: AGPL-3.0-or-later
"""Direct, transaction-safe Q4_0 tensor patching inside an existing GGUF.

This modifies *encoded Q4_0 blocks*, not a reconstructed BF16 checkpoint.
A delta must be supplied in the EXACT GGUF reader tensor order (a .npy file),
rather than assuming a Transformers Linear tensor has the same orientation.

This component applies precomputed updates. It does not yet compute ARA
gradients from GGUF model activations or run a GGUF inference backend.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

Q4_0_BLOCK_VALUES = 32
Q4_0_BLOCK_BYTES = 18


def decode_q4_0_blocks(data: bytes) -> np.ndarray:
    """Decode GGML Q4_0 blocks to a float32 array of shape (blocks, 32).

    Byte layout: fp16 signed scale, then 16 bytes of low/high nibbles.
    Each high nibble decodes element j+16, NOT element 2*j+1.
    """
    if len(data) == 0 or len(data) % Q4_0_BLOCK_BYTES:
        raise ValueError("Q4_0 byte stream must contain complete 18-byte blocks")
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, Q4_0_BLOCK_BYTES)
    scale = np.frombuffer(
        blocks[:, :2].copy().tobytes(), dtype="<f2"
    ).astype(np.float32)
    packed = blocks[:, 2:]
    lo = (packed & np.uint8(0x0F)).astype(np.float32) - 8.0
    hi = (packed >> np.uint8(4)).astype(np.float32) - 8.0
    return np.concatenate((lo, hi), axis=1) * scale[:, None]


def encode_q4_0_blocks(values: np.ndarray) -> bytes:
    """Encode GGML Q4_0 reference blocks, preserving the 18-byte layout.

    Matches the reference signed-max selection, truncated int8 quantizer and
    half-precision scale storage. Exact agreement with compiled ggml on a
    target machine still requires a conformance test (rounding/FTZ).
    """
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != 32:
        raise ValueError("Q4_0 input must have shape (block_count, 32)")
    if not np.isfinite(matrix).all():
        raise ValueError("Cannot quantize non-finite values")
    extreme_index = np.abs(matrix).argmax(axis=1)
    signed_max = matrix[np.arange(matrix.shape[0]), extreme_index]
    scale = signed_max / np.float32(-8.0)
    stored_scale = scale.astype("<f2")
    if not np.isfinite(stored_scale).all():
        raise ValueError("Q4_0 fp16 scale overflow")
    if np.any((scale != 0) & (stored_scale == 0)):
        raise ValueError("Q4_0 fp16 scale underflow; refusing lossy patch")
    inv = np.divide(
        np.float32(1.0),
        scale,
        out=np.zeros_like(scale),
        where=scale != 0,
    )
    # ggml uses (int8_t)(x * inv_scale + 8.5f), truncated toward zero.
    q = np.minimum(
        np.trunc(matrix * inv[:, None] + np.float32(8.5)), 15
    ).astype(np.uint8)
    out = np.empty((matrix.shape[0], Q4_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = np.frombuffer(
        stored_scale.tobytes(), dtype=np.uint8
    ).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << np.uint8(4))
    return out.tobytes()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def patch_q4_0_tensor(
    source: str | Path,
    output: str | Path,
    *,
    tensor_name: str,
    delta_npy: str | Path,
    scale: float = 1.0,
    max_abs_delta: float | None = None,
    chunk_blocks: int = 8192,
    expected_source_sha256: str | None = None,
) -> dict[str, object]:
    """Write a new GGUF with one quantized tensor updated by a .npy delta.

    GGUF headers and all untouched tensors remain byte-identical. Original
    source is never opened for writing. Output is installed by an atomic
    no-overwrite hard link (same filesystem). No file remains on failure.
    """
    if not tensor_name:
        raise ValueError("Explicit tensor_name is required")
    if not math.isfinite(scale):
        raise ValueError("Delta scale must be finite")
    if chunk_blocks < 1:
        raise ValueError("chunk_blocks must be positive")
    if max_abs_delta is not None and (
        not math.isfinite(max_abs_delta) or max_abs_delta <= 0
    ):
        raise ValueError("max_abs_delta must be finite and positive")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Output must differ from source GGUF")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if expected_source_sha256 is not None:
        actual = _sha256_file(source)
        if actual.lower() != expected_source_sha256.lower():
            raise ValueError("Original GGUF SHA256 did not match expectation")

    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("Install gguf: uv sync --extra qat_gguf") from exc

    reader = GGUFReader(str(source))
    matches = [tensor for tensor in reader.tensors if tensor.name == tensor_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one GGUF tensor named {tensor_name!r}; got {len(matches)}"
        )
    tensor = matches[0]
    if tensor.tensor_type.name != "Q4_0":
        raise ValueError(
            f"Refusing to patch {tensor_name}: type is {tensor.tensor_type.name}, not Q4_0"
        )
    size = int(tensor.n_bytes)
    offset = int(tensor.data_offset)
    tensor_shape = tuple(int(x) for x in reversed(tensor.shape))
    total_values = math.prod(tensor_shape)
    if (
        size <= 0
        or size % Q4_0_BLOCK_BYTES
        or size // Q4_0_BLOCK_BYTES * Q4_0_BLOCK_VALUES != total_values
        or offset < 0
        or offset + size > source.stat().st_size
    ):
        raise ValueError("Inconsistent or out-of-bounds GGUF tensor metadata")

    delta = np.load(delta_npy, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(delta, np.ndarray)
        or delta.shape != tensor_shape
        or delta.dtype not in (np.dtype("float16"), np.dtype("float32"))
        or not delta.flags.c_contiguous
    ):
        raise ValueError(
            f"Delta must be C-contiguous float16/float32 .npy with GGUF shape "
            f"{tensor_shape}, got {getattr(delta, 'shape', None)}"
        )
    flat_delta = delta.reshape(-1)
    original_stat = source.stat()
    temp_path: Path | None = None
    changed_blocks = 0
    max_change = 0.0

    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".gguf-ara-", suffix=".tmp", delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, tmp, length=4 * 1024 * 1024)
            tmp.flush()
            os.fsync(tmp.fileno())

        # Change only the selected payload range, never tensor headers or
        # adjacent tensors (even when the source has mixed quantization types).
        with source.open("rb") as original, temp_path.open("r+b") as destination:
            for block_start in range(0, size // Q4_0_BLOCK_BYTES, chunk_blocks):
                count = min(chunk_blocks, size // Q4_0_BLOCK_BYTES - block_start)
                byte_offset = offset + block_start * Q4_0_BLOCK_BYTES
                original.seek(byte_offset)
                old_raw = original.read(count * Q4_0_BLOCK_BYTES)
                if len(old_raw) != count * Q4_0_BLOCK_BYTES:
                    raise IOError("Unexpected end of GGUF while reading target")
                original_values = decode_q4_0_blocks(old_raw)
                first = block_start * Q4_0_BLOCK_VALUES
                last = first + count * Q4_0_BLOCK_VALUES
                delta_block = np.asarray(flat_delta[first:last], dtype=np.float32)
                if not np.isfinite(delta_block).all():
                    raise ValueError("Non-finite values in delta")
                scaled_delta = delta_block * np.float32(scale)
                if not np.isfinite(scaled_delta).all():
                    raise ValueError("Scaled delta overflow")
                if max_abs_delta is not None and np.any(
                    np.abs(scaled_delta) > max_abs_delta
                ):
                    raise ValueError("Delta exceeds max_abs_delta")
                target = original_values + scaled_delta.reshape(count, 32)
                if not np.isfinite(target).all():
                    raise ValueError("Delta caused non-finite Q4_0 targets")
                encoded = encode_q4_0_blocks(target)
                new_values = decode_q4_0_blocks(encoded)
                max_change = max(
                    max_change,
                    float(np.max(np.abs(new_values - original_values))),
                )
                old_bytes = np.frombuffer(
                    old_raw, dtype=np.uint8
                ).reshape(count, Q4_0_BLOCK_BYTES)
                new_bytes = np.frombuffer(
                    encoded, dtype=np.uint8
                ).reshape(count, Q4_0_BLOCK_BYTES)
                changed_blocks += int(np.count_nonzero(np.any(
                    old_bytes != new_bytes, axis=1
                )))
                destination.seek(byte_offset)
                destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        after = source.stat()
        if (
            after.st_size != original_stat.st_size
            or after.st_mtime_ns != original_stat.st_mtime_ns
        ):
            raise RuntimeError("Source GGUF changed during patch; output discarded")
        if temp_path.stat().st_size != source.stat().st_size:
            raise IOError("Output length differs from input GGUF")
        # os.link is atomic and refuses to overwrite an existing output.
        # Works on NTFS and ordinary POSIX filesystems; no partial .gguf
        # can become visible as the finished artifact.
        os.link(temp_path, output)
        return {
            "source_path": str(source),
            "output_path": str(output),
            "tensor": tensor_name,
            "tensor_shape_gguf_reader": list(tensor_shape),
            "quantization": "Q4_0",
            "blocks_total": size // Q4_0_BLOCK_BYTES,
            "blocks_changed": changed_blocks,
            "max_abs_dequantized_weight_change": max_change,
            "bytes_total": original_stat.st_size,
            "delta_scale": scale,
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
