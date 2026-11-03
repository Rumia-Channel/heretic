#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Start an always-on d1-2B monitoring proxy for a local QAT GGUF llama-server."""

import argparse

from heretic.gguf_monitor_proxy import ResidentGGUFProxy
from heretic.sidecar_d1 import D1Sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--judge-model", default="LiquidAI/d1-2B")
    parser.add_argument("--judge-device", default="cuda:1")
    parser.add_argument("--log", default="gguf-d1-proxy.jsonl")
    parser.add_argument("--interval-chars", type=int, default=256)
    parser.add_argument("--loop-threshold", type=float, default=0.9)
    parser.add_argument("--stop-on-loop", action="store_true")
    parser.add_argument("--save-text", action="store_true")
    args = parser.parse_args()

    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(
            "Bind the proxy to localhost; put an authenticated TLS reverse proxy "
            "in front for external access."
        )
    judge = D1Sidecar(model_name=args.judge_model, device=args.judge_device)
    with ResidentGGUFProxy(
        (args.bind, args.port),
        upstream=args.upstream,
        judge=judge,
        log_path=args.log,
        interval_chars=args.interval_chars,
        loop_threshold=args.loop_threshold,
        stop_on_loop=args.stop_on_loop,
        save_text=args.save_text,
    ) as proxy:
        print(
            f"Resident d1-2B ready at http://{args.bind}:{args.port}; "
            f"upstream={args.upstream}; every streamed response monitored"
        )
        try:
            proxy.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
