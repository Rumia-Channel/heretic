#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare train-only Japanese activation prompts for Gemma 4 QAT ARA.

Uses existing RumiaChannel datasets. Never includes the test split in
training activations; test prompts remain held out for model evaluation.
"""

import argparse
import json
from pathlib import Path


DATASETS = {
    "good": "RumiaChannel/harmless_behaviors_ja_synth",
    "bad": "RumiaChannel/harmful_behaviors_ja_synth",
}


def _choose_column(features, requested=None):
    if requested:
        if requested not in features:
            raise ValueError(f"Dataset missing requested column {requested!r}")
        return requested
    for name in ("prompt", "text", "instruction", "question"):
        if name in features:
            return name
    raise ValueError(f"No prompt column detected; features: {list(features)}")


def export_prompts(destination: Path, *, limit: int, good_col=None, bad_col=None):
    from datasets import load_dataset

    if limit < 1:
        raise ValueError("limit must be positive")
    if destination.exists():
        raise FileExistsError(destination)
    outputs = {}
    for kind, repo in DATASETS.items():
        split = load_dataset(repo, split="train")
        column = _choose_column(
            split.features, good_col if kind == "good" else bad_col,
        )
        prompts = []
        seen = set()
        for entry in split:
            value = entry[column]
            if not isinstance(value, str):
                continue
            text = " ".join(value.split()).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= limit:
                break
        if not prompts:
            raise ValueError(f"No usable training prompts from {repo}")
        outputs[kind] = {
            "repo": repo, "split": "train", "column": column, "count": len(prompts),
            "prompts": prompts,
        }
    destination.mkdir(parents=True)
    for kind, info in outputs.items():
        (destination / f"{kind}.txt").write_text(
            "\n".join(info["prompts"]) + "\n", encoding="utf-8",
        )
    report = {
        kind: {k: v for k, v in info.items() if k != "prompts"}
        for kind, info in outputs.items()
    }
    (destination / "sources.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--good-column", default=None)
    parser.add_argument("--bad-column", default=None)
    args = parser.parse_args()
    print(json.dumps(export_prompts(
        args.out, limit=args.limit,
        good_col=args.good_column, bad_col=args.bad_column,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
