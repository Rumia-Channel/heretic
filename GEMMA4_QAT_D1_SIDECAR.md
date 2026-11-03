# Gemma 4 26B A4B QAT Q4_0 + Liquid AI d1-2B sidecar

Development branch: `ara-gemma4-26BA4B-qat`.

## Target architecture

The intended starting point is Google's **already quantized, official QAT Q4_0 GGUF**, not a BF16 base model followed by an unrelated quantizer.

```text
official QAT Q4_0 GGUF (frozen and version-pinned)
   |
   +-- runtime inference / data collection [NOT YET INTEGRATED]
   |       |
   |       +-- local d1-2B decision sidecar  [NOW WIRED INTO HF EVALUATOR]
   |
   +-- ARA trainable intervention over quantized base [NOT YET IMPLEMENTED]
           |
           +-- evaluate actual Q4_0 runtime, rollback on quality regression
```

**Important**: Heretic's existing PyTorch ARA code **cannot directly optimise a GGUF Q4_0 file**.
`quantization="bnb_4bit"` is bitsandbytes, NOT GGUF Q4_0.
The sidecar implementation here works on generated responses from the
existing `Evaluator` and is **not evidence** that direct-GGUF editing already works.
Never silently convert to another base, precision or quantization format.

## d1-2B is a decision model, not a chat model

The model `LiquidAI/d1-2B` uses `AutoModel` and `system_one(state, questions)`.
It returns yes/no **probabilities** (field `noul`) without generating tokens.
Do **not** call a chat-completions endpoint for this integration.

It currently checks:
- refusal
- looping
- incomplete generation
- off-topic content

Existing token-level periodic-loop detection, repetition fraction and
EOS/maximum-length flags remain independent signals.

### Windows PowerShell setup

Run in the cloned Heretic repository:

```powershell
git switch ara-gemma4-26BA4B-qat
uv sync --extra decision_sidecar --dev
```

For existing environments managed without uv, Liquid AI's model card
specifies `pip install "transformers>=5.19" torch torchvision pillow`.

In `config.toml`:

```toml
d1_sidecar_enabled = true
d1_sidecar_model = "LiquidAI/d1-2B"
d1_sidecar_device = "cpu"
d1_sidecar_threshold = 0.75
d1_sidecar_max_samples = 24
d1_sidecar_max_input_chars = 8000
d1_sidecar_strict = true
evaluation_log_path = "ara-evaluation-d1.jsonl"
max_response_length = 512
```

**Do not copy these settings as a purported direct-GGUF launch command.**
They only enable the sidecar in evaluations that already run through Heretic's
`Model`/Transformers pathway. The GGUF backend is not yet implemented.

CPU mode defaults to FP32 and may require significant system RAM; CUDA uses
BF16 and requires extra VRAM. During Gemma 4 Q4_0 experiments, reserving the GPU
for the model and running d1-2B on CPU reduces VRAM contention. The d1
checkpoint's remote model code is executed via `trust_remote_code=True`;
review it and pin a revision for reproducible deployments.

### What is recorded

With `evaluation_log_path` configured, each selected output contains:

```json
{
  "d1_sidecar": {
    "model": "LiquidAI/d1-2B",
    "probabilities": {
      "refusal": 0.05,
      "looping": 0.94,
      "incomplete": 0.10,
      "off_topic": 0.04
    },
    "flags": {
      "refusal": false,
      "looping": true,
      "incomplete": false,
      "off_topic": false
    },
    "threshold": 0.75
  }
}
```

These are **illustrative values**, not measured benchmark results.
Unsampled rows have `"d1_sidecar": null`; failed monitor calls record
`"error"`, unless `d1_sidecar_strict=true` aborts the evaluation.

The default 24 rows are spread deterministically through the evaluated
prompt list. Counts printed to the console are counts **in that subsample**,
not rates over the entire set.

The sidecar is currently **diagnostic only**. It does not silently modify ARA's
Optuna score, reject a candidate, stop live token generation or decide whether
a response is safe. Calibrate the threshold on separately annotated Japanese
responses, especially intentional repetition (lists, poems, code) and
partial answers. Flagged examples require review.

## Read-only QAT GGUF inventory

The first GGUF-specific function has now been added: `heretic.qat_gguf.inspect_qat_gguf`.
It uses the reference `gguf.GGUFReader` and reports each stored quantization
type without changing bytes. It explicitly rejects non-Q4_0 artifacts.

```powershell
uv sync --extra qat_gguf
uv run --extra qat_gguf python scripts/inspect_qat_gguf.py --model "C:\\Models\\gemma-4-26B-A4B-qat-q4_0.gguf" --sha256
```

Replace the example filename with the **actual** local GGUF path. `--sha256`
reads the entire file; omit it when you only need the tensor inventory.

This inventory is **not** a GGUF training or inference backend. It is the
first validation step needed before wiring ARA to Q4_0 tensor blocks.

## Direct Q4_0 work remaining (blocking)

1. **Completed partially:** read-only GGUF tensor inventory and optional SHA-256
   hashing. **Still needed:** baseline GGUF inference, packed-MoE detection,
   tensor-level patch eligibility checks and a pinned artifact manifest.
2. Decide how ARA's trainable intervention is represented while keeping all
   **original Q4_0 storage bytes frozen** (e.g. separate adapter, or targeted
   dequantize/edit/re-encode with an exact Q4_0 kernel).
3. If a tensor is patched directly, validate encoded Q4_0 blocks and exact
   inference in llama.cpp, and keep a byte-for-byte rollback copy.
4. Integrate GGUF generation traces with the same local d1 sidecar. Later,
   optionally inspect fixed-size **generation chunks** to detect loops early.
   Post-generation auditing is implemented here; live stopping is not.
5. Compare edited candidate against the **original** official GGUF for
   generation stability, task correctness, empty output, EOS, repetition and
   refusal. Maintain independent held-out normal-use evaluation data.

The existing `src/heretic/quantization.py` reference Q4_0 fake quantizer is
a component-level simulation, **not a GGUF reader/writer or complete QAT
training pathway**.

## Source

- [LiquidAI/d1-2B model card](https://huggingface.co/LiquidAI/d1-2B)
- [Official QAT GGUF](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
- [QAT unquantized comparison checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized)
