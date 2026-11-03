# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for d1 decision sidecar without loading checkpoint weights."""

import math

import pytest

from heretic.sidecar_d1 import (
    D1MonitorError,
    D1Sidecar,
    D1Verdict,
    QUESTIONS,
    parse_d1_output,
)


def _answer(refusal=0.1, looping=0.9, incomplete=0.4, off_topic=0.2):
    return {
        "answers": {
            "refusal": {"noul": refusal},
            "looping": {"noul": looping},
            "incomplete": {"noul": incomplete},
            "off_topic": {"noul": off_topic},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


class FakeD1:
    def __init__(self, output=None):
        self.output = output or _answer()
        self.state = None
        self.questions = None

    def system_one(self, state, questions):
        self.state = state
        self.questions = questions
        return self.output


def test_parse_scores_and_thresholds():
    verdict = parse_d1_output(_answer())
    assert verdict == D1Verdict(0.1, 0.9, 0.4, 0.2)
    assert verdict.flags(0.75) == {
        "refusal": False,
        "looping": True,
        "incomplete": False,
        "off_topic": False,
    }


def test_assess_uses_system_one_and_truncates_inputs():
    fake = FakeD1()
    monitor = D1Sidecar(decision_model=fake, max_input_chars=10)
    verdict = monitor.assess(
        prompt="あ" * 20,
        response="x" * 30,
        token_repetition=True,
        hit_max_length=True,
        eos_seen=False,
    )
    assert verdict.looping == 0.9
    assert fake.state["user_prompt"] == "あ" * 10
    assert fake.state["model_response"] == "x" * 10
    assert fake.state["response_truncated_for_monitor"] is True
    assert fake.state["signals"]["token_repetition"] is True
    assert fake.questions is QUESTIONS


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"answers": []},
        {"answers": {"refusal": {"noul": 0.5}}},
        _answer(looping=float("nan")),
        _answer(incomplete=float("inf")),
        _answer(off_topic=-0.01),
        _answer(refusal=1.01),
        _answer(refusal=True),
        _answer(looping="0.5"),
    ],
)
def test_malformed_outputs_fail_closed(bad):
    with pytest.raises(D1MonitorError):
        parse_d1_output(bad)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -1.0, math.nan])
def test_invalid_threshold_rejected(threshold):
    # NaN needs rejection as well, not merely a 0 <= x <= 1 check.
    with pytest.raises(ValueError):
        D1Verdict(0.2, 0.4, 0.1, 0.7).flags(threshold)


def test_missing_model_name_rejected():
    with pytest.raises(ValueError):
        D1Sidecar(model_name=" ")
