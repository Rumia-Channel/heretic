# SPDX-License-Identifier: AGPL-3.0-or-later
"""Always-resident d1-2B monitoring of llama.cpp OpenAI-compatible SSE.

Use this when the generator is a GGUF model served by llama-server, instead
of Heretic's Transformers Model. The monitor loads d1 once and judges every
generation *during* streaming and again at the end. No inference backend
is silently substituted for GGUF.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .sidecar_d1 import D1Sidecar, D1Verdict


@dataclass
class StreamMonitorResult:
    response: str
    finish_reason: str | None
    checked_chunks: int
    stopped_for_loop: bool
    final_verdict: D1Verdict
    token_repetition_hint: bool

    def asdict(self) -> dict[str, Any]:
        out = asdict(self)
        out["final_flags_at_0_75"] = self.final_verdict.flags(0.75)
        return out


def _text_loop_hint(text: str, width: int = 16) -> bool:
    """Cheap character n-gram hint. Not a ground-truth loop detector."""
    if len(text) < max(128, width * 8):
        return False
    windows = (text[i : i + width] for i in range(len(text) - width + 1))
    window_list = list(windows)
    repeated_fraction = 1.0 - len(set(window_list)) / len(window_list)
    return repeated_fraction > 0.45


def monitor_sse_stream(
    events: Iterable[bytes],
    *,
    prompt: str,
    sidecar: D1Sidecar,
    check_every_chars: int = 256,
    loop_threshold: float = 0.9,
    stop_on_loop: bool = False,
    on_text: Callable[[str], None] | None = None,
) -> StreamMonitorResult:
    """Consume OpenAI-style SSE with resident d1 checks on partial responses.

    \`stop_on_loop\` closes consumption on a confirmed probabilistic loop flag,
    but does not guarantee that the remote GGUF server immediately cancels its
    background generation. It is OFF by default.

    Malformed JSON/events raise instead of silently dropping a monitoring
    failure. A final check is performed even on empty responses.
    """
    if check_every_chars < 1:
        raise ValueError("check_every_chars must be positive")
    if not 0.0 < loop_threshold < 1.0:
        raise ValueError("loop_threshold must lie strictly between 0 and 1")
    chunks: list[str] = []
    length = 0
    last_checked = 0
    checks = 0
    finish_reason: str | None = None
    stopped = False

    # Ensure the judge is already warm before the first generated token.
    sidecar._load()

    for raw_line in events:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        choices = event.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError("SSE delta.content must be a string")
        if not content:
            continue
        chunks.append(content)
        length += len(content)
        if on_text is not None:
            on_text(content)
        if length - last_checked >= check_every_chars:
            partial = "".join(chunks)
            verdict = sidecar.assess(
                prompt=prompt,
                response=partial,
                token_repetition=_text_loop_hint(partial),
                hit_max_length=False,
                eos_seen=False,
            )
            checks += 1
            last_checked = length
            if stop_on_loop and verdict.looping >= loop_threshold:
                stopped = True
                break

    response = "".join(chunks)
    final = sidecar.assess(
        prompt=prompt,
        response=response,
        token_repetition=_text_loop_hint(response),
        hit_max_length=(finish_reason == "length") or stopped,
        eos_seen=finish_reason == "stop",
    )
    checks += 1
    return StreamMonitorResult(
        response=response,
        finish_reason=finish_reason,
        checked_chunks=checks,
        stopped_for_loop=stopped,
        final_verdict=final,
        token_repetition_hint=_text_loop_hint(response),
    )


def stream_gguf_chat(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    sidecar: D1Sidecar,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    check_every_chars: int = 256,
    stop_on_loop: bool = False,
    loop_threshold: float = 0.9,
    timeout: float = 300.0,
    on_text: Callable[[str], None] | None = None,
) -> StreamMonitorResult:
    """Run one GGUF response from llama-server, monitoring it while streaming."""
    from urllib.request import Request, urlopen

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:", "https://")):
        raise ValueError("Use localhost or an authenticated HTTPS endpoint")
    if not model:
        raise ValueError("Model identifier must be nonempty")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return monitor_sse_stream(
            response, prompt=prompt, sidecar=sidecar,
            check_every_chars=check_every_chars,
            loop_threshold=loop_threshold,
            stop_on_loop=stop_on_loop,
            on_text=on_text,
        )
