# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optimize many Q4_0 GGUF tensors with ARA, then publish one candidate GGUF.

Generic code: no Gemma-specific tensor names or routing assumptions. The
caller prepares per-module I/O from the EXACT quantized source model and
supplies a manifest [{tensor, module_io}]. A SHA256 provenance gate prevents
using activation arrays from a different checkpoint.

This does not turn Q4_0 into a differentiable tensor in llama.cpp; STE is
used in PyTorch for local weight optimisation. The final candidate is
re-encoded Q4_0, validated at block level, and MUST undergo real inference.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .gguf_ara_optimizer import Q4ARAOptions, compute_ara_delta_q4_0
from .gguf_batch import patch_q4_0_manifest


def hash_gguf_source(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_provenance(path: Path, tensor_name: str, source_hash: str) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        if "source_sha256" not in bundle or "tensor_name" not in bundle:
            raise ValueError(
                f"Missing source SHA256 / tensor identity in {path}; "
                "cross-model I/O cannot be accepted"
            )
        actual_hash = str(bundle["source_sha256"].item())
        actual_name = str(bundle["tensor_name"].item())
        if actual_hash.lower() != source_hash.lower() or actual_name != tensor_name:
            raise ValueError(
                f"I/O provenance mismatch for {tensor_name}: {path}"
            )


def optimise_ara_q4_manifest(
    source_gguf: str | Path,
    output_gguf: str | Path,
    targets_manifest: str | Path,
    *,
    options: Q4ARAOptions | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Train each listed Q4_0 tensor, then apply all deltas in one transaction."""
    source = Path(source_gguf).resolve()
    output = Path(output_gguf).resolve()
    manifest = Path(targets_manifest).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Never overwrite source QAT GGUF")
    if output.exists():
        raise FileExistsError(output)
    options = options or Q4ARAOptions()
    options.validate()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("ARA targets manifest must be a nonempty JSON array")
    seen: set[str] = set()
    prepared = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("ARA targets must be JSON objects")
        name, io_path = item.get("tensor"), item.get("module_io")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Each ARA tensor name must be unique and nonempty")
        if not isinstance(io_path, str) or not io_path:
            raise ValueError(f"Missing module_io path for {name}")
        seen.add(name)
        io_file = Path(io_path)
        if not io_file.is_absolute():
            io_file = manifest.parent / io_file
        if not io_file.is_file():
            raise FileNotFoundError(io_file)
        prepared.append((name, io_file))
    source_hash = hash_gguf_source(source)
    if expected_sha256 is not None and source_hash.lower() != expected_sha256.lower():
        raise ValueError("Original GGUF SHA256 differs from expected baseline")
    # Preflight ALL activations before spending GPU time on any layer.
    for name, path in prepared:
        _read_provenance(path, name, source_hash)

    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="heretic-q4-batch-ara-") as tmp_dir:
        tmp = Path(tmp_dir)
        edits = []
        for index, (name, io_file) in enumerate(prepared):
            delta, report = compute_ara_delta_q4_0(
                source,
                tensor_name=name,
                module_io_npz=io_file,
                options=options,
            )
            if not np.isfinite(delta).all():
                raise FloatingPointError(f"Non-finite delta: {name}")
            delta_path = tmp / f"delta-{index}.npy"
            np.save(delta_path, delta)
            edits.append({"tensor": name, "delta": str(delta_path), "scale": 1.0})
            reports.append(report)
            del delta
        manifest_path = tmp / "patches.json"
        manifest_path.write_text(json.dumps(edits), encoding="utf-8")
        patch = patch_q4_0_manifest(
            source, output, manifest=manifest_path,
        )
    return {
        "source_sha256": source_hash,
        "source_gguf": str(source),
        "output_gguf": str(output),
        "ara_layers": reports,
        "q4_0_patches": patch["tensor_patches"],
        "note": "Run real GGUF runtime quality gates; module-local ARA alone is not sufficient.",
    }
