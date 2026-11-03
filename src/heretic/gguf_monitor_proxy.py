# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resident d1-2B monitoring proxy for a local llama.cpp GGUF server.

Route ALL text chat completions through this proxy and bind the underlying
llama-server to localhost. It rejects non-streaming requests rather than
allowing completions that bypass the decision sidecar.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .gguf_stream_monitor import monitor_sse_stream
from .sidecar_d1 import D1Sidecar


class LockedJudge:
    """Serialize system_one calls; remote model code may not be thread-safe."""

    def __init__(self, judge: D1Sidecar) -> None:
        self.judge = judge
        self.lock = threading.Lock()
        self.judge._load()  # Load once at server startup.

    def _load(self) -> Any:
        return self.judge._load()

    def assess(self, **kwargs: Any):
        with self.lock:
            return self.judge.assess(**kwargs)


def _extract_last_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = [
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ]
            if text:
                return "\n".join(text)
    raise ValueError("A text user message is required for monitoring")


class ResidentGGUFProxy(ThreadingHTTPServer):
    """HTTP server with a single resident d1 model and serialized judge calls."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstream: str,
        judge: D1Sidecar,
        log_path: str | Path,
        interval_chars: int = 256,
        loop_threshold: float = 0.9,
        stop_on_loop: bool = False,
        upstream_timeout: float = 300.0,
        save_text: bool = False,
    ) -> None:
        if not upstream.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Proxy upstream must be a local llama.cpp server")
        if interval_chars < 1:
            raise ValueError("interval_chars must be positive")
        self.upstream = upstream.rstrip("/")
        self.judge = LockedJudge(judge)
        self.log_path = Path(log_path)
        self.log_lock = threading.Lock()
        self.interval_chars = interval_chars
        self.loop_threshold = loop_threshold
        self.stop_on_loop = stop_on_loop
        self.upstream_timeout = upstream_timeout
        self.save_text = save_text
        super().__init__(address, _ProxyHandler)

    def log_record(self, record: dict[str, Any]) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
        serialized = json.dumps(record, ensure_ascii=False)
        with self.log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")


class _ProxyHandler(BaseHTTPRequestHandler):
    server: ResidentGGUFProxy  # type: ignore[assignment]

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = b'{"status":"ok","judge":"resident"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Only chat/completions is available")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Invalid request length (max 2 MB)")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or payload.get("stream") is not True:
                raise ValueError("stream=true is required for uninterrupted monitoring")
            prompt = _extract_last_user_text(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))
            return

        request = Request(
            self.server.upstream + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            upstream = urlopen(request, timeout=self.server.upstream_timeout)
        except (URLError, OSError) as exc:
            self.server.log_record({"error": f"upstream: {exc}", "prompt_sha256": prompt_hash})
            self.send_error(502, "Local llama.cpp upstream unavailable")
            return

        try:
            with upstream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def tee():
                    for raw in upstream:
                        self.wfile.write(raw)
                        self.wfile.flush()
                        yield raw

                result = monitor_sse_stream(
                    tee(),
                    prompt=prompt,
                    sidecar=self.server.judge,  # type: ignore[arg-type]
                    check_every_chars=self.server.interval_chars,
                    loop_threshold=self.server.loop_threshold,
                    stop_on_loop=self.server.stop_on_loop,
                )
                record: dict[str, Any] = {
                    "prompt_sha256": prompt_hash,
                    "model": payload.get("model"),
                    "finish_reason": result.finish_reason,
                    "checked_chunks": result.checked_chunks,
                    "stopped_for_loop": result.stopped_for_loop,
                    "probabilities": result.final_verdict.probabilities(),
                    "loop_hint": result.token_repetition_hint,
                }
                if self.server.save_text:
                    record["prompt"] = prompt
                    record["response"] = result.response
                self.server.log_record(record)
        except (Exception, KeyboardInterrupt) as exc:
            self.server.log_record({
                "prompt_sha256": prompt_hash,
                "error": f"{type(exc).__name__}: {exc}",
            })
            # Abort rather than complete a response without a working judge.
            self.close_connection = True
            if isinstance(exc, KeyboardInterrupt):
                raise
