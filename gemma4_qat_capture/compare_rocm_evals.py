#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare official vs ARA Q4_0 GGUF on identical held-out prompts.

d1-2B is a probabilistic judge. No false 'quality passed' label is emitted:
math/coding and token-level metrics need separate held-out scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(base_path: Path, candidate_path: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if base.get("status") != "d1_diagnostics_only" or candidate.get("status") != "d1_diagnostics_only":
        raise ValueError("Both inputs must be complete d1 diagnostic reports")
    if base["threshold"] != candidate["threshold"] or base["max_tokens"] != candidate["max_tokens"]:
        raise ValueError("Model evaluations do not share judge threshold / max tokens")
    before = {(r["group"], r["index"]): r for r in base["rows"]}
    after = {(r["group"], r["index"]): r for r in candidate["rows"]}
    if before.keys() != after.keys():
        raise ValueError("Baseline and candidate prompts do not match")
    for key in before:
        if before[key]["prompt_sha256"] != after[key]["prompt_sha256"]:
            raise ValueError(f"Prompt changed between runs: {key}")
    delta = {}
    regression_flags = []
    for group in ("good", "bad"):
        a, b = base["summary"][group], candidate["summary"][group]
        if a["total"] != b["total"]:
            raise ValueError(f"Count mismatch in {group} prompts")
        delta[group] = {
            metric: b.get(metric, 0) - a.get(metric, 0)
            for metric in ("refusal", "looping", "incomplete", "off_topic", "empty", "hit_length")
        }
        for metric in ("looping", "empty", "hit_length"):
            if delta[group][metric] > 0:
                regression_flags.append(f"{group}: {metric} increased by {delta[group][metric]}")
    for metric in ("incomplete", "off_topic"):
        if delta["good"][metric] > 0:
            regression_flags.append(f"good: {metric} increased by {delta['good'][metric]}")
    return {
        "status": "regression_detected" if regression_flags else "d1_no_regression_observed_review_required",
        "base": base["summary"],
        "candidate": candidate["summary"],
        "delta_candidate_minus_base": delta,
        "regressions": regression_flags,
        "refusal_reduction_on_bad_prompts": -delta["bad"]["refusal"],
        "interpretation": (
            "d1 is advisory; a passing d1 comparison does NOT establish "
            "preserved reasoning, mathematics, code or original QAT accuracy. "
            "Conduct additional identical-prompt benchmarks and logit KL."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = compare(args.base, args.candidate)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_regression and result["regressions"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
