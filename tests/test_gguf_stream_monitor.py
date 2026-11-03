# SPDX-License-Identifier: AGPL-3.0-or-later
"""SSE monitoring tests, using a fake d1 (no checkpoint / llama-server)."""

import json

import pytest

from heretic.gguf_stream_monitor import monitor_sse_stream
from heretic.sidecar_d1 import D1Verdict


class FakeSidecar:
    def __init__(self, looping=0.01):
        self.looping = looping
        self.loaded = 0
        self.calls = []

    def _load(self):
        self.loaded += 1

    def assess(self, **kwargs):
        self.calls.append(kwargs)
        return D1Verdict(0.01, self.looping, 0.02, 0.03)


def _event(content="", finish=None):
    choice = {"delta": {"content": content}, "finish_reason": finish}
    return ("data: " + json.dumps({"choices": [choice]}) + "\n\n").encode("utf-8")


def test_every_generation_checked_during_stream_and_at_end():
    judge = FakeSidecar()
    text_chunks = []
    data = [_event("aaa"), _event("bbb"), _event("ccc"), _event(finish="stop")]
    result = monitor_sse_stream(
        data, prompt="test question", sidecar=judge, check_every_chars=4,
        on_text=text_chunks.append,
    )
    assert result.response == "aaabbbccc"
    assert result.finish_reason == "stop"
    assert result.checked_chunks == 2
    assert judge.loaded == 1
    assert len(judge.calls) == 2
    assert text_chunks == ["aaa", "bbb", "ccc"]


def test_empty_generation_still_checked():
    judge = FakeSidecar()
    result = monitor_sse_stream(
        [_event(finish="stop"), b"data: [DONE]\n"],
        prompt="question", sidecar=judge,
    )
    assert result.response == ""
    assert result.checked_chunks == 1
    assert len(judge.calls) == 1


def test_can_stop_early_on_detected_loop():
    judge = FakeSidecar(looping=0.98)
    result = monitor_sse_stream(
        [_event("hello"), _event("world"), _event(finish="stop")],
        prompt="question", sidecar=judge, check_every_chars=4,
        stop_on_loop=True,
    )
    assert result.stopped_for_loop
    assert result.response == "hello"
    assert result.finish_reason is None


def test_d1_failure_is_not_silently_ignored():
    class FailingJudge(FakeSidecar):
        def assess(self, **kwargs):
            raise RuntimeError("local judge unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        monitor_sse_stream(
            [_event("bad")], prompt="q", sidecar=FailingJudge(),
            check_every_chars=2,
        )


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        monitor_sse_stream(
            [], prompt="q", sidecar=FakeSidecar(), check_every_chars=interval
        )
