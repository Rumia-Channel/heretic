# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval
import torch.nn.functional as F
from dataclasses import dataclass, field
from lm_eval.models.huggingface import HFLM
from torch import Tensor

from .config import Settings
from .model import Model, ResponseRecord
from .utils import (
    Prompt,
    has_periodic_suffix,
    load_prompts,
    print,
    repeated_ngram_fraction,
)


# Per-prompt evaluation results, kept separate so that different failure
# modes (refusal, empty response, repetition, missing EOS) are never
# conflated into a single count.
@dataclass
class ResponseStats:
    refusals: int = 0
    empty: int = 0
    repetitive: int = 0
    hit_max_length: int = 0
    records: list[ResponseRecord] = field(default_factory=list)

# A response counts as repetitive when more than half of its 8-grams are
# duplicates, or when it ends in a short periodic loop. These thresholds
# are deliberately conservative; the detector flags candidates for human
# review, it does not define ground truth.
REPETITION_NGRAM_FRACTION = 0.5


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        if not settings.use_piqa:
            print()
            print(
                f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
            )
            self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
            print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

            print("* Obtaining first-token probability distributions...")
            self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        self.base_stats = self.evaluate_responses()
        self.base_refusals = self.base_stats.refusals
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def evaluate_responses(self) -> ResponseStats:
        stats = ResponseStats()

        records = self.model.get_response_records_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        for prompt, record in zip(self.bad_prompts, records):
            is_refusal = self.is_refusal(record.text)
            if is_refusal:
                stats.refusals += 1
            if not record.text.strip():
                stats.empty += 1
            is_repetitive = (
                repeated_ngram_fraction(record.token_ids)
                >= REPETITION_NGRAM_FRACTION
                or has_periodic_suffix(record.token_ids)
            )
            if is_repetitive:
                stats.repetitive += 1
            if record.hit_max_length:
                stats.hit_max_length += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                response = record.text
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )
                if record.hit_max_length:
                    print("[yellow]Response reached the maximum length without EOS.[/]")
                if is_repetitive:
                    print("[yellow]Response shows token-level repetition.[/]")

        stats.records = records

        if self.settings.print_responses:
            print()

        return stats

    def get_score(self) -> tuple[tuple[float, float], float, ResponseStats]:
        if self.settings.use_piqa:
            print("  * Running PIQA benchmark...")
            hflm = HFLM(
                pretrained=self.model.model,  # ty:ignore[invalid-argument-type]
                tokenizer=self.model.tokenizer,  # ty:ignore[invalid-argument-type]
                batch_size="auto",
            )
            results = lm_eval.simple_evaluate(
                model=hflm,
                tasks=["piqa"],
            )
            piqa_acc_norm: float = results["results"]["piqa"]["acc_norm,none"]
            print(f"  * PIQA acc_norm: [bold]{piqa_acc_norm:.4f}[/]")
        else:
            print("  * Obtaining first-token probability distributions...")
            logprobs = self.model.get_logprobs_batched(self.good_prompts)
            kl_divergence = F.kl_div(
                logprobs,
                self.base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
            print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Evaluating model responses...")
        stats = self.evaluate_responses()
        print(f"  * Refusals: [bold]{stats.refusals}[/]/{len(self.bad_prompts)}")
        if stats.empty:
            print(f"  * Empty responses: [bold]{stats.empty}[/]")
        if stats.repetitive:
            print(f"  * Repetitive responses: [bold]{stats.repetitive}[/]")
        if stats.hit_max_length:
            print(
                f"  * Reached max length without EOS: [bold]{stats.hit_max_length}[/]"
            )

        refusals_score = (
            stats.refusals / self.base_refusals
            if self.base_refusals > 0
            else float(stats.refusals)
        )

        if self.settings.use_piqa:
            score = (
                -piqa_acc_norm,
                refusals_score,
            )

            return score, -piqa_acc_norm, stats
        else:
            kl_divergence_scale = self.settings.kl_divergence_scale
            kl_divergence_target = self.settings.kl_divergence_target

            if kl_divergence >= kl_divergence_target:
                kld_score = kl_divergence / kl_divergence_scale
            else:
                kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

            score = (
                kld_score,
                refusals_score,
            )

            return score, kl_divergence, stats
