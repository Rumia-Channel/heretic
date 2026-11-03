#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q4_0 GGUF ARA over many captured projection layers in one transaction."""

import argparse
import json

from heretic.gguf_ara_batch_optimizer import optimise_ara_q4_manifest
from heretic.gguf_ara_optimizer import Q4ARAOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", required=True, help="JSON list of {tensor, module_io}")
    parser.add_argument("--expect-sha256", default=None)
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
        device=args.device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        neighbor_count=args.neighbors,
        preserve_good_weight=args.preserve_good_weight,
        steer_bad_weight=args.steer_bad_weight,
        overcorrect_weight=args.overcorrect_weight,
        delta_l2_weight=args.delta_l2_weight,
        max_abs_delta=args.max_abs_delta,
        preserve_row_norm=not args.no_preserve_row_norm,
    )
    result = optimise_ara_q4_manifest(
        args.model,
        args.output,
        args.targets,
        options=options,
        expected_sha256=args.expect_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
