#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect official QAT Q4_0 GGUF before any attempted ARA editing."""

import argparse
import json

from heretic.qat_gguf import inspect_qat_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the existing GGUF file")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash the entire GGUF (slower for large models)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_qat_gguf(args.model, hash_source=args.sha256),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
