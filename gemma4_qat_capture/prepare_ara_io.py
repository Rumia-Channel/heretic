#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare Gemma 4 ARA projection I/O from a REAL QAT GGUF cb_eval capture.

Capture only the kqv_out (pre-attention-output projection) input with
llama.cpp. The corresponding *raw linear projection* output y is computed
using the exact dequantized Q4_0 weights of the SAME GGUF. The runtime
may apply additional wo_s scales, normalization and residual additions:
this y is the projection-local reference, not the post-norm final output.

Outputs:
  io/layer-N.npz (good_x, good_y, bad_x, bad_y, source_sha256, tensor_name)
  io/ara_targets.json (list of {tensor, module_io} objects)
  io/source.json (SHA256, architecture, layer sizes, capture-format, counts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from heretic.gguf_direct import decode_q4_0_blocks


def _parse_capture_manifest(path: Path) -> tuple[int, int, dict[int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    props: dict[str, str] = {}
    dimensions: dict[int, int] = {}
    for line in lines:
        if line.startswith("layer="):
            fields = dict(pair.split("=", 1) for pair in line.split(","))
            layer, width = int(fields["layer"]), int(fields["features"])
            if layer in dimensions or layer < 0 or width < 1:
                raise ValueError("Invalid capture layer feature declaration")
            dimensions[layer] = width
        elif "=" in line:
            key, value = line.split("=", 1)
            if key in props:
                raise ValueError(f"Duplicate capture property {key}")
            props[key] = value
    if props.get("format") != "gemma4.kqv_out.last_prompt_token.f32.v1":
        raise ValueError("Unknown capture format; expected actual llama.cpp kqv_out v1")
    good, bad = int(props["good_count"]), int(props["bad_count"])
    if min(good, bad) <= 0 or not dimensions:
        raise ValueError("Capture must contain good/bad inputs and layers")
    return good, bad, dimensions


def _load_raw_vectors(path: Path, count: int, width: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * width * 4:
        raise ValueError(f"Capture size mismatch or missing file: {path}")
    matrix = np.fromfile(path, dtype="<f4").reshape(count, width)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite activation in {path}")
    return matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_io(
    source_gguf: Path,
    captured_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory must not exist: {output_dir}")
    from gguf import GGUFReader

    good_count, bad_count, dimensions = _parse_capture_manifest(
        captured_dir / "capture.txt"
    )
    reader = GGUFReader(str(source_gguf))
    architecture_field = reader.fields.get("general.architecture")
    if architecture_field is None:
        raise ValueError("GGUF architecture field missing")
    architecture = str(architecture_field.contents())
    if architecture not in ("gemma4", "gemma4_moe"):
        raise ValueError(f"This adapter is ONLY for Gemma 4, got {architecture!r}")
    tensors = {t.name: t for t in reader.tensors}
    source_hash = _sha256(source_gguf)
    planned = []
    # Verify every tensor and capture size before creating outputs.
    for layer, input_width in sorted(dimensions.items()):
        name = f"blk.{layer}.attn_output.weight"
        tensor = tensors.get(name)
        if tensor is None or tensor.tensor_type.name != "Q4_0":
            raise ValueError(f"Missing Q4_0 attention output tensor {name}")
        shape = tuple(int(d) for d in reversed(tensor.shape))
        if (
            len(shape) != 2 or shape[1] != input_width
            or int(tensor.n_bytes) != math.prod(shape) // 32 * 18
        ):
            raise ValueError(
                f"GGUF tensor orientation mismatch for {name}: "
                f"shape={shape}, captured_input_width={input_width}"
            )
        good_x_path = captured_dir / f"layer-{layer}.good.f32"
        bad_x_path = captured_dir / f"layer-{layer}.bad.f32"
        if good_x_path.stat().st_size != good_count * input_width * 4:
            raise ValueError(f"Incorrect good activation size for layer {layer}")
        if bad_x_path.stat().st_size != bad_count * input_width * 4:
            raise ValueError(f"Incorrect bad activation size for layer {layer}")
        planned.append((layer, name, tensor, shape, good_x_path, bad_x_path))
    output_dir.mkdir(parents=True)

    targets = []
    layout = {}
    for layer, name, tensor, shape, good_path, bad_path in planned:
        good_x = _load_raw_vectors(good_path, good_count, shape[1])
        bad_x = _load_raw_vectors(bad_path, bad_count, shape[1])
        with source_gguf.open("rb") as stream:
            stream.seek(int(tensor.data_offset))
            encoded = stream.read(int(tensor.n_bytes))
        if len(encoded) != int(tensor.n_bytes):
            raise IOError(f"Truncated Q4_0 tensor in {name}")
        base = decode_q4_0_blocks(encoded).reshape(shape)
        # Matrix multiplication is local to the original Q4_0 projection.
        # Do not silently combine the post-norm value as y!
        weights = torch.from_numpy(base.copy()).to(device=device)
        with torch.no_grad():
            good_y = (
                torch.from_numpy(good_x.copy()).to(device=device) @ weights.T
            ).cpu().numpy()
            bad_y = (
                torch.from_numpy(bad_x.copy()).to(device=device) @ weights.T
            ).cpu().numpy()
        if not np.isfinite(good_y).all() or not np.isfinite(bad_y).all():
            raise FloatingPointError(f"Invalid reconstructed projection output: {name}")
        path = output_dir / f"layer-{layer}.npz"
        np.savez(
            path,
            good_x=good_x, good_y=good_y, bad_x=bad_x, bad_y=bad_y,
            source_sha256=np.array(source_hash),
            tensor_name=np.array(name),
        )
        targets.append({"tensor": name, "module_io": path.name})
        layout[name] = list(shape)
        del weights

    (output_dir / "ara_targets.json").write_text(
        json.dumps(targets, indent=2), encoding="utf-8"
    )
    report = {
        "format": "gemma4-q4-ara-capture-v1",
        "architecture": architecture,
        "source_gguf": str(source_gguf.resolve()),
        "source_sha256": source_hash,
        "capture_directory": str(captured_dir.resolve()),
        "good_count": good_count,
        "bad_count": bad_count,
        "tensor_shapes": layout,
        "activation_source": "llama.cpp cb_eval / kqv_out / last token",
        "output_definition": "raw dequantized Q4_0 matmul x @ W.T, NOT post-norm",
    }
    (output_dir / "source.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_io(args.model, args.capture, args.out, device=args.device),
            ensure_ascii=False, indent=2,
        )
    )


if __name__ == "__main__":
    main()
