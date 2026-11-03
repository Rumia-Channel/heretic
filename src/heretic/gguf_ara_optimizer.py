# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local Q4_0-aware ARA optimisation using recorded GGUF module I/O.

Optimizes an ACTUAL tensor decoded from a Q4_0 GGUF, not a BF16 base model.
Uses the same ARA preserve/steer/kNN objective as Heretic's dense module
optimizer, with a Q4_0 STE in the forward pass and direct GGUF patch output.

LIMITATION: Requires good_x/good_y/bad_x/bad_y .npz arrays captured from
the *same original GGUF tensor* at its input/output boundaries. Heretic does
not yet capture those internals from llama.cpp; recording them is a separate
integration task. Never substitute activations from a differently quantized
model without reporting the mismatch. Supports only 2D Q4_0 tensors.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .gguf_direct import decode_q4_0_blocks, patch_q4_0_tensor
from .quantization import fake_quantize_q4_0


@dataclass(frozen=True)
class Q4ARAOptions:
    steps: int = 15
    learning_rate: float = 0.0005
    neighbor_count: int = 4
    preserve_good_weight: float = 1.0
    steer_bad_weight: float = 0.1
    overcorrect_weight: float = 0.5
    delta_l2_weight: float = 0.01
    max_abs_delta: float = 0.25
    preserve_row_norm: bool = True
    device: str = "cuda"

    def validate(self) -> None:
        if self.steps < 1 or self.neighbor_count < 1:
            raise ValueError("steps and neighbor_count must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in (
            "preserve_good_weight",
            "steer_bad_weight",
            "overcorrect_weight",
            "delta_l2_weight",
            "max_abs_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_abs_delta == 0:
            raise ValueError("max_abs_delta cannot be zero")


def _load_io(npz_path: str | Path, shape: tuple[int, int], device: str) -> dict[str, torch.Tensor]:
    output_features, input_features = shape
    with np.load(npz_path, allow_pickle=False) as bundle:
        required = ("good_x", "good_y", "bad_x", "bad_y")
        if not all(name in bundle for name in required):
            raise ValueError("I/O archive needs good_x, good_y, bad_x and bad_y")
        arrays = {name: np.array(bundle[name], dtype=np.float32) for name in required}
    for name, arr in arrays.items():
        expected = input_features if name.endswith("_x") else output_features
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != expected:
            raise ValueError(f"{name} must have nonempty shape (examples, {expected})")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
    if len(arrays["good_x"]) != len(arrays["good_y"]):
        raise ValueError("good_x/good_y example counts differ")
    if len(arrays["bad_x"]) != len(arrays["bad_y"]):
        raise ValueError("bad_x/bad_y example counts differ")
    return {
        name: torch.from_numpy(arr).to(device=device)
        for name, arr in arrays.items()
    }


def _knn_distance(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Nearest-neighbor mean, capping k for sparse expert observations."""
    k = min(k, b.shape[0])
    return torch.cdist(a, b).topk(k, largest=False, dim=1).values.mean()


def compute_ara_delta_q4_0(
    source: str | Path,
    *,
    tensor_name: str,
    module_io_npz: str | Path,
    options: Q4ARAOptions | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one GGUF Q4_0 projection without writing a new multi-GB GGUF.

    The returned float32 delta is in GGUF reader row-major matrix order.
    Caller should apply multiple deltas in a SINGLE GGUF batch transaction.
    """
    options = options or Q4ARAOptions()
    options.validate()
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("Install gguf: uv sync --extra qat_gguf") from exc

    source = Path(source).resolve()
    reader = GGUFReader(str(source))
    matches = [t for t in reader.tensors if t.name == tensor_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one GGUF tensor named {tensor_name!r}")
    tensor = matches[0]
    if tensor.tensor_type.name != "Q4_0":
        raise ValueError("Only Q4_0 tensors can be directly edited")
    shape = tuple(int(x) for x in reversed(tensor.shape))
    if len(shape) != 2 or shape[1] % 32 != 0:
        raise ValueError("Only 2D row-major GGUF Q4_0 projections are supported")
    expected_nbytes = math.prod(shape) // 32 * 18
    if int(tensor.n_bytes) != expected_nbytes:
        raise ValueError("GGUF tensor size inconsistent with Q4_0 shape")

    io = _load_io(module_io_npz, shape, options.device)
    with source.open("rb") as stream:
        stream.seek(int(tensor.data_offset))
        raw = stream.read(expected_nbytes)
    if len(raw) != expected_nbytes:
        raise IOError("GGUF Q4_0 tensor data truncated")
    base = torch.from_numpy(
        decode_q4_0_blocks(raw).reshape(shape).copy()
    ).to(device=options.device)
    original_norm = torch.linalg.vector_norm(base, dim=1, keepdim=True)
    master = torch.nn.Parameter(base.clone())
    optimizer = torch.optim.Adam([master], lr=options.learning_rate)

    initial_loss = None
    final_loss = None
    for step in range(options.steps):
        optimizer.zero_grad(set_to_none=True)
        matrix = master
        if options.preserve_row_norm:
            # Keep zero-norm rows zero rather than creating NaNs.
            matrix = original_norm * F.normalize(master, dim=1, eps=1e-8)
        simulated_q4 = fake_quantize_q4_0(matrix)
        good_new = io["good_x"] @ simulated_q4.T
        bad_new = io["bad_x"] @ simulated_q4.T
        good_loss = (good_new - io["good_y"]).square().mean()
        bad_loss = _knn_distance(
            bad_new, io["good_y"], options.neighbor_count
        ) - options.overcorrect_weight * _knn_distance(
            bad_new, io["bad_y"], options.neighbor_count
        )
        delta = matrix - base
        loss = (
            options.preserve_good_weight * good_loss
            + options.steer_bad_weight * bad_loss
            + options.delta_l2_weight * delta.square().mean()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Q4_0 ARA loss at step {step}")
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        if master.grad is None or not torch.isfinite(master.grad).all():
            raise FloatingPointError(f"Invalid Q4_0 ARA gradient at step {step}")
        torch.nn.utils.clip_grad_norm_([master], 1.0)
        optimizer.step()
        if not torch.isfinite(master).all():
            raise FloatingPointError(f"Invalid Q4_0 ARA weights at step {step}")
        with torch.no_grad():
            master.copy_(
                torch.minimum(
                    torch.maximum(master, base - options.max_abs_delta),
                    base + options.max_abs_delta,
                )
            )
        final_loss = float(loss.detach().item())

    with torch.no_grad():
        final = master
        if options.preserve_row_norm:
            final = original_norm * F.normalize(master, dim=1, eps=1e-8)
        delta = final - base
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite final Q4_0 update")
        delta_np = delta.cpu().numpy().astype(np.float32)
    if not np.isfinite(delta_np).all():
        raise FloatingPointError("Non-finite GGUF delta")

    report: dict[str, Any] = {
        "tensor": tensor_name,
        "ara_initial_loss": initial_loss,
        "ara_final_loss": final_loss,
        "ara_options": asdict(options),
        "training_io_path": str(Path(module_io_npz).resolve()),
        "note": (
            "Module-local Q4_0 surrogate objective; evaluate actual GGUF "
            "with llama.cpp and d1 before accepting the candidate."
        ),
    }
    return delta_np, report


def optimise_ara_tensor_on_q4_0(
    source: str | Path,
    output: str | Path,
    *,
    tensor_name: str,
    module_io_npz: str | Path,
    options: Q4ARAOptions | None = None,
) -> dict[str, Any]:
    """Optimize a single GGUF Q4_0 projection and atomically create its GGUF."""
    delta_np, metrics = compute_ara_delta_q4_0(
        source,
        tensor_name=tensor_name,
        module_io_npz=module_io_npz,
        options=options,
    )
    with tempfile.TemporaryDirectory(prefix="heretic-q4-ara-") as work:
        delta_path = Path(work) / "delta.npy"
        np.save(delta_path, delta_np)
        report = patch_q4_0_tensor(
            source, output, tensor_name=tensor_name, delta_npy=delta_path,
            max_abs_delta=(options or Q4ARAOptions()).max_abs_delta * 2,
        )
    report.update(metrics)
    return report
