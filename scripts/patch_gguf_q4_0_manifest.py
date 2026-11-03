#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply multiple precomputed ARA deltas to a QAT Q4_0 GGUF in one copy."""

import argparse
import json

from heretic.gguf_batch import patch_q4_0_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--manifest", required=True,
        help='JSON list of {"tensor": "...", "delta": "relative.npy", "scale": 1.0}',
    )
    parser.add_argument("--chunk-blocks", type=int, default=8192)
    args = parser.parse_args()
    report = patch_q4_0_manifest(
        args.model, args.output, manifest=args.manifest,
        chunk_blocks=args.chunk_blocks,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
