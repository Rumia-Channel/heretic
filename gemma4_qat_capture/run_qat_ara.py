#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run Gemma 4 QAT Q4_0 capture -> ARA -> patched GGUF end to end.

This is an experimental production of a CANDIDATE model, not an unconditional
quality certificate. Deploy only if held-out llama.cpp evaluation with d1-2B,
KL and task scores is acceptable. Retains the original official GGUF.
"""

import argparse
import json
import subprocess
from pathlib import Path

from heretic.gguf_ara_batch_optimizer import optimise_ara_q4_manifest
from heretic.gguf_ara_optimizer import Q4ARAOptions

from prepare_ara_io import prepare_io


def _parse_layers(text: str) -> list[int]:
    values = text.split(",")
    if not values or any(not v.isdecimal() for v in values):
        raise ValueError("--layers must be a comma-separated list like 0,1,2")
    indices = sorted({int(x) for x in values})
    return indices


def _validate_target_tensors(model_path: Path, indices: list[int]) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(str(model_path))
    arch = reader.fields.get("general.architecture")
    if arch is None or arch.contents() != "gemma4":
        raise ValueError("Requires Gemma 4 GGUF (architecture gemma4)")
    names = {tensor.name: tensor for tensor in reader.tensors}
    for layer in indices:
        name = f"blk.{layer}.attn_output.weight"
        tensor = names.get(name)
        if tensor is None or tensor.tensor_type.name != "Q4_0":
            raise ValueError(f"No editable Q4_0 attention projection: {name}")
        if len(tensor.shape) != 2:
            raise ValueError(f"Expected matrix at {name}; got {tensor.shape}")


def run_pipeline(args: argparse.Namespace) -> dict:
    original = args.model.resolve()
    collector = args.collector.resolve()
    output_dir = args.out.resolve()
    if not original.is_file() or not collector.is_file():
        raise FileNotFoundError("Original GGUF or llama.cpp collector binary not found")
    if not args.good.is_file() or not args.bad.is_file():
        raise FileNotFoundError("Good/bad prompt files missing")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    indices = _parse_layers(args.layers)
    _validate_target_tensors(original, indices)
    output_dir.mkdir(parents=True)
    captured = output_dir / "captured"
    io = output_dir / "io"
    candidate = output_dir / "candidate.gguf"
    command = [
        str(collector), str(original), str(args.good.resolve()),
        str(args.bad.resolve()), str(captured), args.layers,
        str(args.gpu_layers), str(args.batch), str(args.ctx), args.system,
    ]
    # No shell=True. Fail fast if llama.cpp callback encounters bad shapes.
    subprocess.run(command, check=True)
    prepared = prepare_io(original, captured, io, device=args.device)
    options = Q4ARAOptions(
        device=args.device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        neighbor_count=args.neighbors,
        preserve_good_weight=args.preserve_good_weight,
        steer_bad_weight=args.steer_bad_weight,
        overcorrect_weight=args.overcorrect_weight,
        delta_l2_weight=args.delta_l2_weight,
        max_abs_delta=args.max_abs_delta,
    )
    optimized = optimise_ara_q4_manifest(
        original, candidate, io / "ara_targets.json",
        options=options, expected_sha256=prepared["source_sha256"],
    )
    result = {
        "status": "candidate_generated_requires_holdout_quality_gate",
        "original_gguf": str(original),
        "candidate_gguf": str(candidate),
        "original_sha256": prepared["source_sha256"],
        "captured_layers": indices,
        "capture_summary": prepared,
        "ara_summary": optimized,
        "warning": (
            "This is not a validated final model. Independently compare original "
            "and candidate GGUF with llama.cpp using held-out test prompts, "
            "d1-2B loop monitoring, generation stability and task scoring."
        ),
    }
    (output_dir / "run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--collector", required=True, type=Path)
    parser.add_argument("--good", required=True, type=Path)
    parser.add_argument("--bad", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,2")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--system", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--preserve-good-weight", type=float, default=1.0)
    parser.add_argument("--steer-bad-weight", type=float, default=0.1)
    parser.add_argument("--overcorrect-weight", type=float, default=0.5)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--max-abs-delta", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
