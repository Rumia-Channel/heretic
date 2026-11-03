# SPDX-License-Identifier: AGPL-3.0-or-later
"""No-GPU tests for the MI300X evaluation/gating wrapper."""

import json

import pytest

from gemma4_qat_capture.compare_rocm_evals import compare
from gemma4_qat_capture.eval_rocm_sidecar import (
    _wait_for_judge,
    evaluate,
    load_one_line_prompts,
)


def _record(prompt_hash, refusal=0.1, looping=0.03):
    return {
        "prompt_sha256": prompt_hash,
        "checked_chunks": 2,
        "probabilities": {
            "refusal": refusal,
            "looping": looping,
            "incomplete": 0.02,
            "off_topic": 0.01,
        },
        "finish_reason": "stop",
    }


def test_wait_for_judge_requires_a_real_decision_log(tmp_path):
    import hashlib

    log = tmp_path / "log.jsonl"
    good_hash = hashlib.sha256("テスト".encode("utf-8")).hexdigest()
    log.write_text(json.dumps(_record(good_hash)) + "\n", encoding="utf-8")
    consumed = set()
    decision = _wait_for_judge(log, good_hash, consumed, timeout=0.3)
    assert decision["probabilities"]["refusal"] == 0.1
    assert consumed == {0}
    with pytest.raises(TimeoutError):
        _wait_for_judge(log, good_hash, consumed, timeout=0.01)


def test_judge_error_never_counts_as_success(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"prompt_sha256": "abc", "error": "model OOM"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="model OOM"):
        _wait_for_judge(log, "abc", set(), timeout=0.1)


def test_evaluator_refuses_missing_proxy_decisions(monkeypatch, tmp_path):
    import gemma4_qat_capture.eval_rocm_sidecar as ev

    good, bad = tmp_path / "good.txt", tmp_path / "bad.txt"
    good.write_text("天気の説明\n", encoding="utf-8")
    bad.write_text("要求への応答\n", encoding="utf-8")
    monkeypatch.setattr(
        ev, "send_streamed_prompt",
        lambda *args, **kwargs: {
            "response_chars": 30,
            "empty_response": False,
            "client_finish_reason": "stop",
        },
    )
    def missing(*args, **kwargs):
        raise TimeoutError("No matching d1 decision")
    monkeypatch.setattr(ev, "_wait_for_judge", missing)
    with pytest.raises(TimeoutError):
        evaluate(
            "http://127.0.0.1:18090", "local",
            {"good": good, "bad": bad},
            tmp_path / "report.json",
            tmp_path / "monitor.jsonl",
        )
    assert not (tmp_path / "report.json").exists()
    assert (tmp_path / "report.json.partial.json").is_file()


def _report(prompt_hash, group, *, loop=False, refusal=False):
    metrics = {
        "total": 1,
        "refusal": int(refusal),
        "looping": int(loop),
        "incomplete": 0,
        "off_topic": 0,
        "empty": 0,
        "hit_length": 0,
    }
    return {
        "status": "d1_diagnostics_only",
        "threshold": 0.75,
        "max_tokens": 384,
        "summary": {
            "good": metrics if group == "good" else {**metrics, "looping": 0},
            "bad": {**metrics, "refusal": 1} if group == "good" else metrics,
        },
        "rows": [
            {"group": "good", "index": 0, "prompt_sha256": prompt_hash},
            {"group": "bad", "index": 0, "prompt_sha256": prompt_hash},
        ],
    }


def test_comparison_rejects_changed_test_prompts(tmp_path):
    b = _report("before", "good")
    c = _report("after", "good")
    base, candidate = tmp_path / "a.json", tmp_path / "b.json"
    base.write_text(json.dumps(b), encoding="utf-8")
    candidate.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="Prompt changed"):
        compare(base, candidate)


def test_comparison_flags_new_good_loops(tmp_path):
    base = _report("same", "good", loop=False)
    candidate = _report("same", "good", loop=True)
    b, c = tmp_path / "base.json", tmp_path / "candidate.json"
    b.write_text(json.dumps(base), encoding="utf-8")
    c.write_text(json.dumps(candidate), encoding="utf-8")
    result = compare(b, c)
    assert result["status"] == "regression_detected"
    assert result["delta_candidate_minus_base"]["good"]["looping"] == 1
    assert result["regressions"]


def test_prompt_loader_nonempty(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("\n\nリスト\n", encoding="utf-8")
    assert load_one_line_prompts(path) == ["リスト"]
