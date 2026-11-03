# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local HTTP smoke test for the resident GGUF monitoring gateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from heretic.gguf_monitor_proxy import ResidentGGUFProxy
from heretic.sidecar_d1 import D1Sidecar


class FakeD1:
    def system_one(self, _state, questions):
        return {
            "answers": {
                name: {"noul": 0.1}
                for name in questions
            }
        }


class FakeUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for fragment in ("Hello", " world"):
            item = {"choices": [{
                "delta": {"content": fragment},
                "finish_reason": None,
            }]}
            self.wfile.write(("data: " + json.dumps(item) + "\n\n").encode())
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_proxy_checks_every_streamed_chat_and_rejects_bypass(tmp_path):
    with ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream) as backend:
        upstream_thread = _start(backend)
        with ResidentGGUFProxy(
            ("127.0.0.1", 0),
            upstream=f"http://127.0.0.1:{backend.server_port}",
            judge=D1Sidecar(decision_model=FakeD1()),
            log_path=tmp_path / "monitor.jsonl",
            interval_chars=3,
        ) as proxy:
            proxy_thread = _start(proxy)
            addr = f"http://127.0.0.1:{proxy.server_port}"
            with urlopen(addr + "/healthz", timeout=5) as resp:
                assert resp.status == 200
            payload = {"model": "local", "stream": False,
                       "messages": [{"role": "user", "content": "hello"}]}
            request = Request(
                addr + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as caught:
                urlopen(request, timeout=5)
            assert caught.value.code == 400
            payload["stream"] = True
            request = Request(
                addr + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as resp:
                raw = resp.read().decode()
                assert "Hello" in raw and " world" in raw
            lines = (tmp_path / "monitor.jsonl").read_text().splitlines()
            assert len(lines) == 1
            row = json.loads(lines[0])
            assert row["checked_chunks"] >= 2
            assert row["probabilities"]["looping"] == 0.1
            proxy.shutdown()
            proxy_thread.join(timeout=5)
        backend.shutdown()
        upstream_thread.join(timeout=5)
