#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stream a QAT GGUF response from llama-server under resident d1 monitoring."""

import argparse
import json
import os
import sys
from pathlib import Path

from heretic.gguf_stream_monitor import stream_gguf_chat
from heretic.sidecar_d1 import D1Sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--judge-model", default="LiquidAI/d1-2B")
    parser.add_argument("--judge-device", default="cuda:1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--interval-chars", type=int, default=256)
    parser.add_argument("--loop-threshold", type=float, default=0.9)
    parser.add_argument("--stop-on-loop", action="store_true")
    parser.add_argument("--log", default="gguf-d1-monitor.jsonl")
    args = parser.parse_args()

    sidecar = D1Sidecar(model_name=args.judge_model, device=args.judge_device)
    sidecar._load()
    result = stream_gguf_chat(
        endpoint=args.server,
        model=args.model,
        prompt=args.prompt,
        system_prompt=args.system,
        sidecar=sidecar,
        max_tokens=args.max_tokens,
        check_every_chars=args.interval_chars,
        loop_threshold=args.loop_threshold,
        stop_on_loop=args.stop_on_loop,
        on_text=lambda content: (sys.stdout.write(content), sys.stdout.flush()),
    )
    sys.stdout.write("\n")
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            {"model": args.model, "prompt": args.prompt, **result.asdict()},
            ensure_ascii=False,
        ) + "\n")
    print(
        f"d1 checks={result.checked_chunks}; stopped_for_loop={result.stopped_for_loop}; "
        f"judgments={result.final_verdict.probabilities()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
