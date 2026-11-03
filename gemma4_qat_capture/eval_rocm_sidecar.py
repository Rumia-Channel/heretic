#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run held-out Gemma 4 QAT GGUF evaluations through the RESIDENT d1 proxy.

This driver uses separate test-split prompt files. It MUST connect to the
d1 proxy, not llama-server directly. It refuses to produce a summary unless
there is a real d1 decision log for EVERY streamed generation.

Judgements are diagnostics, not automatic proof of reasoning/task retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


def load_one_line_prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"Empty evaluation prompts: {path}")
    return prompts


def _read_proxy_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed d1 log {path}:{line_num}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Malformed d1 log entry at line {line_num}")
        rows.append(parsed)
    return rows


def _wait_for_judge(
    log_path: Path,
    prompt_hash: str,
    consumed_lines: set[int],
    *,
    timeout: float = 30.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for index, record in enumerate(_read_proxy_records(log_path)):
            if index in consumed_lines or record.get("prompt_sha256") != prompt_hash:
                continue
            consumed_lines.add(index)
            if "error" in record:
                raise RuntimeError(f"d1 proxy error: {record['error']}")
            probabilities = record.get("probabilities")
            if not isinstance(probabilities, dict):
                raise RuntimeError("d1 monitor did not report probabilities")
            for key in ("refusal", "looping", "incomplete", "off_topic"):
                score = probabilities.get(key)
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise RuntimeError(f"Invalid d1 {key} probability")
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise RuntimeError(f"Out-of-range d1 {key} probability")
            if int(record.get("checked_chunks", 0)) < 1:
                raise RuntimeError("d1 failed to evaluate a generated response")
            return record
        time.sleep(0.2)
    raise TimeoutError(
        "No matching d1 decision log after a streamed response; "
        "refusing to call this generation monitored"
    )


def send_streamed_prompt(
    proxy_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        proxy_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    parts = []
    reason = None
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    reason = choice["finish_reason"]
    response_text = "".join(parts)
    return {
        "response_chars": len(response_text),
        "empty_response": not response_text.strip(),
        "client_finish_reason": reason,
    }


def evaluate(
    proxy_url: str,
    model: str,
    prompt_sets: dict[str, Path],
    output: Path,
    proxy_log: Path,
    *,
    max_tokens: int = 384,
    timeout: float = 300.0,
    threshold: float = 0.75,
) -> dict:
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1")
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    # Require a new, dedicated proxy log file for this evaluation. This
    # makes a verdict unambiguously attributable to the current run.
    if proxy_log.exists() and proxy_log.stat().st_size != 0:
        raise ValueError("Proxy log must be empty at start of an evaluation")
    prompts = {name: load_one_line_prompts(path) for name, path in prompt_sets.items()}
    consumed: set[int] = set()
    rows = []
    summary = {}
    try:
        for group, group_prompts in prompts.items():
            counters = Counter()
            for index, prompt in enumerate(group_prompts):
                details = send_streamed_prompt(
                    proxy_url, model, prompt, max_tokens=max_tokens, timeout=timeout,
                )
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                judge = _wait_for_judge(proxy_log, prompt_hash, consumed)
                probs = judge["probabilities"]
                counters["total"] += 1
                for field in ("refusal", "looping", "incomplete", "off_topic"):
                    if probs[field] >= threshold:
                        counters[field] += 1
                if details["empty_response"]:
                    counters["empty"] += 1
                if judge.get("finish_reason") == "length":
                    counters["hit_length"] += 1
                rows.append(
                    {
                        "group": group,
                        "index": index,
                        "prompt_sha256": prompt_hash,
                        "response_chars": details["response_chars"],
                        "empty_response": details["empty_response"],
                        "finish_reason": judge.get("finish_reason"),
                        "checked_chunks": judge["checked_chunks"],
                        "probabilities": probs,
                        "flags": {
                            key: probs[key] >= threshold
                            for key in ("refusal", "looping", "incomplete", "off_topic")
                        },
                    }
                )
            summary[group] = dict(counters)
            for metric in (
                "refusal", "looping", "incomplete", "off_topic", "empty", "hit_length"
            ):
                summary[group].setdefault(metric, 0)
            print(json.dumps({"group_complete": group, **summary[group]}), flush=True)
    except Exception:
        # Preserve partial diagnostics for incident analysis without calling
        # the partial run a successful evaluation.
        failure_path = output.with_suffix(output.suffix + ".partial.json")
        failure_path.write_text(
            json.dumps({"status": "failed", "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    report = {
        "status": "d1_diagnostics_only",
        "model": model,
        "proxy_url": proxy_url,
        "threshold": threshold,
        "max_tokens": max_tokens,
        "summary": summary,
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="local")
    parser.add_argument("--good", type=Path, required=True)
    parser.add_argument("--bad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    result = evaluate(
        args.proxy,
        args.model,
        {"good": args.good, "bad": args.bad},
        args.out,
        args.proxy_log,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        threshold=args.threshold,
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
