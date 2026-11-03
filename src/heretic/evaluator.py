# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import math

import lm_eval
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from lm_eval.models.huggingface import HFLM
from torch import Tensor

from .config import Settings
from .model import Model, ResponseRecord
from .sidecar_d1 import D1Sidecar
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
    d1_evaluated: int = 0
    d1_errors: int = 0
    d1_refusal_flags: int = 0
    d1_loop_flags: int = 0
    d1_incomplete_flags: int = 0
    d1_off_topic_flags: int = 0
    d1_good_evaluated: int = 0
    d1_good_errors: int = 0
    d1_good_loop_flags: int = 0
    d1_good_incomplete_flags: int = 0
    d1_good_off_topic_flags: int = 0
    records: list[ResponseRecord] = field(default_factory=list)
    # Measured first-token KL divergence, when computed. Under PIQA this is
    # only populated when record_kl_with_piqa is enabled.
    kl_divergence: float | None = None
    # KL divergence over the last kl_context_positions prompt positions,
    # measured under teacher forcing on the same prefix for both models.
    context_kl_divergence: float | None = None


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
        self.evaluation_index = 0
        self.d1_sidecar = (
            D1Sidecar(
                model_name=settings.d1_sidecar_model,
                device=settings.d1_sidecar_device,
                max_input_chars=settings.d1_sidecar_max_input_chars,
            )
            if settings.d1_sidecar_enabled
            else None
        )
        # Always-on: fail early if the resident judge cannot be started.
        # Keep it loaded across all ARA trials; no sampling or lazy unload.
        if self.d1_sidecar is not None:
            print("[cyan]Loading resident LiquidAI/d1-2B sidecar...[/]")
            self.d1_sidecar._load()

        if (
            not settings.use_piqa
            or settings.record_kl_with_piqa
            or settings.kl_context_positions > 0
            or settings.d1_sidecar_enabled
        ):
            print()
            print(
                f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
            )
            self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
            print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

            if settings.kl_context_positions > 0:
                print(
                    f"* Obtaining context probability distributions "
                    f"(last {settings.kl_context_positions} positions)..."
                )
                self.base_context_logprobs = model.get_context_logprobs_batched(
                    self.good_prompts,
                    settings.kl_context_positions,
                )

            if not settings.use_piqa or settings.record_kl_with_piqa:
                print("* Obtaining first-token probability distributions...")
                self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Evaluating model responses...")
        self.base_stats = self.evaluate_responses()
        self._monitor_good_generations(self.base_stats)
        self.base_refusals = self.base_stats.refusals
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def matching_refusal_markers(self, response: str) -> list[str]:
        # Empty responses count as refusals without matching any marker.
        if not response.strip():
            return []

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        return [
            marker
            for marker in self.settings.refusal_markers
            if marker.lower() in response
        ]

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        return len(self.matching_refusal_markers(response)) > 0

    def evaluate_responses(self) -> ResponseStats:
        stats = ResponseStats()

        records = self.model.get_response_records_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        # Always-on: assess EVERY generated response. No sampling.
        d1_results: list[dict | None] = []

        for index, (prompt, record) in enumerate(zip(self.bad_prompts, records)):
            markers = self.matching_refusal_markers(record.text)
            record.matched_markers = markers
            is_refusal = not record.text.strip() or len(markers) > 0
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

            d1_result: dict | None = None
            if self.d1_sidecar is not None:
                try:
                    verdict = self.d1_sidecar.assess(
                        prompt=prompt.user,
                        response=record.text,
                        token_repetition=is_repetitive,
                        hit_max_length=record.hit_max_length,
                        eos_seen=record.eos_seen,
                    )
                    flags = verdict.flags(self.settings.d1_sidecar_threshold)
                    d1_result = {
                        "model": self.settings.d1_sidecar_model,
                        "probabilities": verdict.probabilities(),
                        "flags": flags,
                        "threshold": self.settings.d1_sidecar_threshold,
                    }
                    stats.d1_evaluated += 1
                    stats.d1_refusal_flags += int(flags["refusal"])
                    stats.d1_loop_flags += int(flags["looping"])
                    stats.d1_incomplete_flags += int(flags["incomplete"])
                    stats.d1_off_topic_flags += int(flags["off_topic"])
                except Exception as exc:
                    stats.d1_errors += 1
                    d1_result = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "model": self.settings.d1_sidecar_model,
                    }
                    if self.settings.d1_sidecar_strict:
                        raise RuntimeError(
                            f"d1 sidecar failed on prompt index {index}"
                        ) from exc
                    print(
                        f"[yellow]d1 monitor unavailable for prompt {index}: "
                        f"{type(exc).__name__}: {exc}[/]"
                    )
            d1_results.append(d1_result)

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
                if markers:
                    print(f"[grey50]Matched refusal markers: {markers}[/]")
                if record.hit_max_length:
                    print("[yellow]Response reached the maximum length without EOS.[/]")
                if is_repetitive:
                    print("[yellow]Response shows token-level repetition.[/]")

        stats.records = records

        if self.settings.evaluation_log_path:
            self.evaluation_index += 1
            with open(self.settings.evaluation_log_path, "a", encoding="utf-8") as f:
                for prompt, record, d1_result in zip(
                    self.bad_prompts, records, d1_results
                ):
                    f.write(
                        json.dumps(
                            {
                                "evaluation_index": self.evaluation_index,
                                "prompt_group": "bad",
                                "system_prompt": prompt.system,
                                "prompt": prompt.user,
                                "response": record.text,
                                "token_count": len(record.token_ids),
                                "eos_seen": record.eos_seen,
                                "hit_max_length": record.hit_max_length,
                                "matched_markers": record.matched_markers,
                                "d1_sidecar": d1_result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if self.settings.print_responses:
            print()

        return stats

    def _monitor_good_generations(self, stats: ResponseStats) -> None:
        """Check every ordinary-use generation with resident d1.

        This runs in addition to the original KL-only quality metric. It does
        not alter Optuna's objective; it makes generation degeneration visible.
        """
        if self.d1_sidecar is None:
            return
        records = self.model.get_response_records_batched(
            self.good_prompts, skip_special_tokens=True,
        )
        if len(records) != len(self.good_prompts):
            raise RuntimeError("Good-prompt generation count mismatch")
        log_path = self.settings.evaluation_log_path
        for prompt, record in zip(self.good_prompts, records):
            is_repetitive = (
                repeated_ngram_fraction(record.token_ids)
                >= REPETITION_NGRAM_FRACTION
                or has_periodic_suffix(record.token_ids)
            )
            result: dict
            try:
                verdict = self.d1_sidecar.assess(
                    prompt=prompt.user,
                    response=record.text,
                    token_repetition=is_repetitive,
                    hit_max_length=record.hit_max_length,
                    eos_seen=record.eos_seen,
                )
                flags = verdict.flags(self.settings.d1_sidecar_threshold)
                result = {
                    "model": self.settings.d1_sidecar_model,
                    "probabilities": verdict.probabilities(),
                    "flags": flags,
                    "threshold": self.settings.d1_sidecar_threshold,
                }
                stats.d1_good_evaluated += 1
                stats.d1_good_loop_flags += int(flags["looping"])
                stats.d1_good_incomplete_flags += int(flags["incomplete"])
                stats.d1_good_off_topic_flags += int(flags["off_topic"])
            except Exception as exc:
                stats.d1_good_errors += 1
                result = {
                    "model": self.settings.d1_sidecar_model,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if self.settings.d1_sidecar_strict:
                    raise RuntimeError(
                        "Resident d1 judge failed on an ordinary-use response"
                    ) from exc
                print(f"[yellow]d1 ordinary-response monitoring failed: {exc}[/]")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "evaluation_index": self.evaluation_index,
                                "prompt_group": "good",
                                "system_prompt": prompt.system,
                                "prompt": prompt.user,
                                "response": record.text,
                                "token_count": len(record.token_ids),
                                "eos_seen": record.eos_seen,
                                "hit_max_length": record.hit_max_length,
                                "token_repetition": is_repetitive,
                                "d1_sidecar": result,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
        print(
            f"  * d1 ordinary-use checks: {stats.d1_good_evaluated} passed, "
            f"{stats.d1_good_errors} errors, "
            f"{stats.d1_good_loop_flags} looping, "
            f"{stats.d1_good_incomplete_flags} incomplete"
        )

    def measure_kl_divergence(self) -> float:
        logprobs = self.model.get_logprobs_batched(self.good_prompts)

        # Guard against non-finite logprobs before the KL calculation.
        if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
            logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=-1e10)

        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()

        # Guard against a non-finite KL divergence result.
        if not math.isfinite(kl_divergence):
            kl_divergence = 1e10

        return kl_divergence

    def get_score(self) -> tuple[tuple[float, float], float, ResponseStats]:
        kl_divergence: float | None = None

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

            if self.settings.record_kl_with_piqa:
                print("  * Obtaining first-token probability distributions...")
                kl_divergence = self.measure_kl_divergence()
                print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")
        else:
            print("  * Obtaining first-token probability distributions...")
            kl_divergence = self.measure_kl_divergence()
            print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Evaluating model responses...")
        stats = self.evaluate_responses()
        self._monitor_good_generations(stats)
        stats.kl_divergence = kl_divergence
        print(f"  * Refusals: [bold]{stats.refusals}[/]/{len(self.bad_prompts)}")
        if stats.empty:
            print(f"  * Empty responses: [bold]{stats.empty}[/]")
        if stats.repetitive:
            print(f"  * Repetitive responses: [bold]{stats.repetitive}[/]")
        if stats.hit_max_length:
            print(
                f"  * Reached max length without EOS: [bold]{stats.hit_max_length}[/]"
            )
        if self.d1_sidecar is not None:
            print(
                f"  * d1 sidecar (all generations): {stats.d1_evaluated} assessed, "
                f"{stats.d1_errors} errors, {stats.d1_refusal_flags} refusals, "
                f"{stats.d1_loop_flags} loops, "
                f"{stats.d1_incomplete_flags} incomplete, "
                f"{stats.d1_off_topic_flags} off-topic"
            )

        if self.settings.kl_context_positions > 0:
            context_logprobs = self.model.get_context_logprobs_batched(
                self.good_prompts,
                self.settings.kl_context_positions,
            )
            stats.context_kl_divergence = F.kl_div(
                context_logprobs,
                self.base_context_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
            print(
                f"  * Context KL divergence: "
                f"[bold]{stats.context_kl_divergence:.4f}[/]"
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
            # kl_divergence is always measured in this branch.
            assert kl_divergence is not None

            kl_divergence_scale = self.settings.kl_divergence_scale
            kl_divergence_target = self.settings.kl_divergence_target

            if self.settings.smooth_kl_objective:
                # Continuous variant: the refusal-scaled target acts as a floor,
                # so the objective keeps tracking the measured KL divergence
                # on both sides of the target instead of jumping between
                # two unrelated expressions.
                kld_score = (
                    max(kl_divergence, refusals_score * kl_divergence_target)
                    / kl_divergence_scale
                )
            elif kl_divergence >= kl_divergence_target:
                kld_score = kl_divergence / kl_divergence_scale
            else:
                kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

            score = (
                kld_score,
                refusals_score,
            )

            return score, kl_divergence, stats
