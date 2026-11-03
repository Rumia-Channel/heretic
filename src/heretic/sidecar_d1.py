# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local LiquidAI/d1-2B sidecar for ARA generation diagnostics.

d1-2B is a *decision* model. It uses \`system_one(state, questions)\` and
returns calibrated yes/no probabilities. It does NOT implement chat completion.

This module is opt-in and independent of ARA's existing optimisation score.
Only explicitly enabling the monitor downloads and loads remote model code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


QUESTIONS: dict[str, dict[str, str]] = {
    "refusal": {
        "type": "noul",
        "instructions": (
            "Does the model response refuse, decline, or explicitly avoid "
            "answering the user's request, rather than attempting an answer? "
            "An explanation of limitations is not necessarily a refusal."
        ),
    },
    "looping": {
        "type": "noul",
        "instructions": (
            "Is the response stuck in an unwanted repetition loop, such as "
            "repeating identical clauses or cycling through the same text? "
            "Intentional lists and code repetition are not by themselves loops."
        ),
    },
    "incomplete": {
        "type": "noul",
        "instructions": (
            "Does the response appear accidentally unfinished or cut off "
            "before completing the user's request? A concise but complete "
            "answer is not incomplete."
        ),
    },
    "off_topic": {
        "type": "noul",
        "instructions": (
            "Does the response mainly ignore the user's actual request or "
            "wander into unrelated subject matter? An on-topic refusal "
            "should not be marked off-topic solely for refusing."
        ),
    },
}


class D1MonitorError(RuntimeError):
    """d1-2B returned malformed decision data."""


@dataclass(frozen=True)
class D1Verdict:
    refusal: float
    looping: float
    incomplete: float
    off_topic: float

    def probabilities(self) -> dict[str, float]:
        return asdict(self)

    def flags(self, threshold: float) -> dict[str, bool]:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Sidecar threshold must be strictly between 0 and 1")
        return {name: value >= threshold for name, value in self.probabilities().items()}


def parse_d1_output(output: Any) -> D1Verdict:
    """Reject missing/nonfinite probabilities instead of treating them as negatives."""
    if not isinstance(output, dict) or not isinstance(output.get("answers"), dict):
        raise D1MonitorError("d1 response must contain an answers mapping")
    answers = output["answers"]
    probs = {}
    for key in QUESTIONS:
        item = answers.get(key)
        if not isinstance(item, dict):
            raise D1MonitorError(f"Missing d1 decision: {key}")
        value = item.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise D1MonitorError(f"Invalid d1 probability for {key}")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise D1MonitorError(f"Out-of-range d1 probability for {key}")
        probs[key] = float(value)
    return D1Verdict(**probs)


class D1Sidecar:
    """Lazy, local adapter. Supply \`decision_model\` for tests without a GPU."""

    def __init__(
        self,
        model_name: str = "LiquidAI/d1-2B",
        device: str = "cpu",
        max_input_chars: int = 8000,
        *,
        decision_model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("d1 model name must not be empty")
        if max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self.device = device
        self._model = decision_model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        # Import only when enabled, to keep ordinary ARA use unchanged.
        import torch
        from transformers import AutoModel

        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device != "cpu" and not (device.startswith("cuda") or device == "mps"):
            raise ValueError(f"Unsupported d1 device: {device}")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        # The model requires custom code. Enabling this option explicitly
        # trusts code distributed with the pinned Hugging Face checkpoint.
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, dtype=dtype
        ).to(device)
        self._model.eval()
        return self._model

    def assess(
        self,
        *,
        prompt: str,
        response: str,
        token_repetition: bool,
        hit_max_length: bool,
        eos_seen: bool,
    ) -> D1Verdict:
        """Assess a candidate from observable generation traces.

        The numerical stop/repetition signals are supplied as context, but
        remain independent diagnostics rather than ground-truth labels.
        """
        model = self._load()
        state = {
            "user_prompt": prompt[: self.max_input_chars],
            # Preserve the tail: degeneration often begins late in generation.
            "model_response": response[-self.max_input_chars :],
            "response_truncated_for_monitor": len(response) > self.max_input_chars,
            "signals": {
                "token_repetition": token_repetition,
                "hit_max_length": hit_max_length,
                "eos_seen": eos_seen,
            },
        }
        # system_one runs one forward pass (no token generation).
        output = model.system_one(state, QUESTIONS)
        return parse_d1_output(output)
