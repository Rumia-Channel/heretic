#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run layer-local ARA on the EXISTING QAT Q4_0 GGUF (requires captured I/O).

The .npz must contain good_x, good_y, bad_x, bad_y from the *same Q4_0
model*, matching the tensor's GGUF reader matrix orientation. No BF16 model
conversion is performed. This does NOT collect the layer I/O itself.
"""

import argparse
import json

from heretic.gguf_ara_optimizer import Q4ARAOptions, optimise_ara_tensor_on_q4_0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Original QAT GGUF")
    parser.add_argument("--output", required=True, help="New, nonexistent GGUF path")
    parser.add_argument("--tensor", required=True, help="Exact GGUF tensor name")
    parser.add_argument("--module-io", required=True, help=".npz with good_x/y and bad_x/y")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--preserve-good-weight", type=float, default=1.0)
    parser.add_argument("--steer-bad-weight", type=float, default=0.1)
    parser.add_argument("--overcorrect-weight", type=float, default=0.5)
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--max-abs-delta", type=float, default=0.25)
    parser.add_argument("--no-preserve-row-norm", action="store_true")
    args = parser.parse_args()

    options = Q4ARAOptions(
        steps=args.steps,
        learning_rate=args.learning_rate,
        neighbor_count=args.neighbors,
        preserve_good_weight=args.preserve_good_weight,
        steer_bad_weight=args.steer_bad_weight,
        overcorrect_weight=args.overcorrect_weight,
        delta_l2_weight=args.delta_l2_weight,
        max_abs_delta=args.max_abs_delta,
        preserve_row_norm=not args.no_preserve_row_norm,
        device=args.device,
    )
    report = optimise_ara_tensor_on_q4_0(
        args.model, args.output, tensor_name=args.tensor,
        module_io_npz=args.module_io, options=options,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
