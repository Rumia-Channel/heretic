# QAT-preserving ARA: implementation plan (draft)

Target: `ara` branch of Heretic, initially the official
`google/gemma-4-26B-A4B-it-qat-q4_0-gguf` and corresponding
`-qat-q4_0-unquantized` checkpoint.

## Goal and non-goals

The goal is to **measure and minimize quality loss in the final Q4_0
deployment format after a weight intervention**. A small loss on full-precision
module outputs is not sufficient evidence that the Q4_0 model remains usable.

Do not claim that QAT quality is preserved without model-wide, held-out
evaluations. Do not silently apply GGUF rules to non-quantized tensors.
Do not replace the supplied official QAT checkpoint with a non-QAT model.

The first patch adds only:
- A reference Q4_0 round-trip and identity-STE fake quantizer.
- A kNN boundary fix when sparsely routed MoE experts have fewer observations
  than the configured number of neighbors.
- Focused unit tests.

**The reference fake quantizer is NOT wired into ARA optimization yet.**
It is not a GGUF reader/writer, and no model was trained or evaluated with it.

## Workstream 0 — obtain trustworthy baselines (blocking)

1. Pin model revisions, model license, `llama.cpp` version, `transformers`
   version, tokenizer, chat template, generation parameters and evaluation
   datasets. Keep the exact model hashes in evaluation metadata.
2. Evaluate the **official Q4_0 GGUF** under llama.cpp.
3. Convert the official **QAT-unquantized** checkpoint to GGUF **without
   editing any tensor**, using the same conversion and quantization recipe.
4. Compare the no-op conversion with the official GGUF on identical tasks.
   Report both accuracy and generation traces. Differences may arise from
   tensor mapping, layer layouts, special tensor types, rounding or kernels.
5. Refuse to start model-wide QAT optimization until the no-op baseline is
   compatible and discrepancies are explained.

Baseline matrix:
- Official QAT Q4_0 GGUF (reference).
- No-op re-quantization from QAT-unquantized.
- Original QAT-unquantized inference (control).
- Future edited QAT-unquantized and edited Q4_0 GGUF (candidates).

## Workstream 1 — verify quantizer fidelity (blocking)

The `src/heretic/quantization.py` reference is based on ggml's
`quantize_row_q4_0_ref`:

- 32-element blocks along the last dimension.
- Select the **signed** maximum-absolute-magnitude element.
- Compute `d = signed_max / -8`.
- Calculate nibble codes from the **unrounded** float32 scale with the ggml
  truncation rule, then clip to `0..15`.
- Store `d` in fp16 and decode using `(q-8)*stored_d`.

Before using it in a training objective:
1. Compare bytes/nibbles and dequantized values against compiled ggml on
   random, tie, zero, subnormal, mixed-sign and extreme blocks.
2. Check the **actual GGUF tensor orientation and per-tensor quantization
   types**; the Python function assumes contiguous 32-wide blocks on the last
   dimension of the supplied tensor.
3. Verify the QAT-unquantized weights are mapped into the same tensor order
   as the GGUF converter.
4. Measure memory overhead for full-weight and LoRA candidates.
5. Explicitly mark which tensors are not Q4_0 and preserve their original
   treatment. Do not fake-quantize norm/embedding tensors by default.
6. Only after fidelity checks, expose an **opt-in** `q4_0_fake` mode, with
   a clear incompatibility error for unsupported layers or quantizers.

The identity-STE backward is only a surrogate; it is not the derivative of
Q4_0 rounding. Start with small deterministic tests and compare Adam-type
first-order optimization against the current L-BFGS strategy. Retain a
full-precision master copy of only the edited tensor(s). Never report a
numerical improvement from a proxy loss as a deployment quality gain.

## Workstream 2 — model quality and stability (blocking)

Keep quality auditing separate from the existing refusal-oriented objective.

Record **independently**, per prompt and per candidate:
- Raw generated token IDs, prompt ID, exact prompt/template and model revision.
- EOS or other stop reason, generated length, context length and max tokens.
- Token n-gram repetition and periodic suffix repetition.
- Empty output, unfinished output, task coverage and answer correctness.
- First-token KL and **teacher-forced multi-position** KL against the same
  baseline prefixes.
- Benchmark scores on normal use cases: reasoning, math, code, Japanese,
  instruction following, long answers and structured outputs.
- Latency and memory, separately from quality.

Use held-out tasks for the final quality gate; calibrate repeat detectors
against human-reviewed false positives (lists, poetry, code and templates).
Judge a candidate against predeclared tolerances; failures revert the edit.
Do not conflate a suppressed repetition symptom with recovered correctness.
Capture counterexamples; report the denominator for every metric.

## Workstream 3 — Gemma 4 MoE introspection

Gemma 4 can pack expert matrices into shared parameters. A module-level
`nn.Linear` hook may not see an individual expert.

1. Detect packed expert formats explicitly and refuse silent partial editing.
2. Capture expert routing, input/output shapes, effective tensor orientation,
   number of observed routed tokens and missing experts.
3. Enforce finite values; skip truly unobserved experts with recorded reasons.
4. Test sparsely routed experts and `k > number of observed examples`.
5. Validate both individual expert changes and **whole-model routing changes**.
6. Keep the first proof of concept on a supported dense projection before
   attempting packed experts.

## Workstream 4 — candidate editing, rollback and export

Two distinct candidates should be compared:
- Official quantized base **frozen** + separately applied LoRA delta.
- Edited high-precision QAT master weights with Q4_0 fake quantization in
  the forward path, then export to an actually runnable GGUF.

Protect every candidate with finite-loss, finite-gradient, finite-weight,
parameter-delta and activation-shift diagnostics. Save an unmodified source
snapshot and record the exact source revision. For model-wide evaluation,
restore the baseline, then apply the candidate deterministically.

Export and test against **the actual llama.cpp runtime**. Do not assume that
bitsandbytes NF4, GGUF Q4_0 and GPTQ 4-bit are interchangeable.

## Completion criteria

The feature is NOT complete until:
- A no-op QAT -> Q4_0 export reproduces the deployment baseline to an
  explicitly declared tolerance.
- The fake quantizer agrees with ggml on encoded blocks and output.
- Both failure and non-failure test cases are recorded for MoE.
- Normal-use quality regressions are measured on held-out tasks and stay
  within predeclared bounds for an adopted candidate.
- Repetition, truncation and empty replies are visible as separate flags.
- A failed edit can be fully rolled back.
- A final artifact loads and runs correctly in llama.cpp.

## References

- Google Gemma 4 QAT checkpoints:
  https://ai.google.dev/gemma/docs/core
- GGML Q4_0 implementation:
  https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c
- GGML Q4_0 representation:
  https://github.com/ggml-org/llama.cpp/discussions/17393
- Repository analysis:
  ARA_TECHNICAL_REPORT.md
