#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply an explicit precomputed tensor delta to an existing GGUF Q4_0 file.

This is a **direct encoded-tensor editor**, not yet a GGUF ARA optimizer.
The source file is preserved. Specify an output that does not yet exist.
"""

import argparse
import json

from heretic.gguf_direct import patch_q4_0_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Original GGUF path")
    parser.add_argument("--output", required=True, help="New GGUF path; must not exist")
    parser.add_argument("--tensor", required=True, help="Exact tensor name in GGUF")
    parser.add_argument(
        "--delta", required=True, help=".npy float16/float32 in GGUF reader tensor order"
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--max-abs-delta", type=float, default=None,
        help="Fail if any scaled delta exceeds this value",
    )
    parser.add_argument("--chunk-blocks", type=int, default=8192)
    parser.add_argument(
        "--expect-sha256", type=str, default=None,
        help="Optional expected original GGUF SHA-256 checksum",
    )
    args = parser.parse_args()
    report = patch_q4_0_tensor(
        args.model,
        args.output,
        tensor_name=args.tensor,
        delta_npy=args.delta,
        scale=args.scale,
        max_abs_delta=args.max_abs_delta,
        chunk_blocks=args.chunk_blocks,
        expected_source_sha256=args.expect_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
