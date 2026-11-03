# Gemma 4 26B A4B: direct QAT Q4_0 ARA

This is **Gemma 4-specific**. Generic Q4_0 GGUF byte editing, STE
optimizer, batch writer, and resident d1 decision monitor live on `ara`,
and are imported into this branch via PR #4. Do **not** merge the files
in `gemma4_qat_capture/` into generic `ara`.

## Model and scope

Base: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`.

The current experimental path modifies **attention output matrices only**,
directly in the already-quantized GGUF:

1. llama.cpp loads the unmodified Google official GGUF.
2. The C++ `cb_eval` callback captures `kqv_out-<layer>` at the **last
   prompt token** using the *embedded Google chat template*.
3. Python validates that the GGUF contains exactly the corresponding
   `blk.N.attn_output.weight` in **Q4_0**, and that the capture width
   equals the matrix's input dimension.
4. `good_y` and `bad_y` are reconstructed from the **original Q4_0 bytes**
   and the captured inputs; they are the **raw linear projection outputs**,
   *not* post-scaling, post-normalization or post-residual activations.
5. Generic `ara` Q4_0 STE ARA optimizes the selected matrices. The original
   official GGUF remains unmodified.
6. Generic batch patcher writes the selected new 18-byte Q4_0 blocks into
   **one new GGUF**, leaving all other bytes untouched.
7. **Required manually at present:** test original vs candidate with actual
   llama.cpp inference, the always-resident d1-2B sidecar, and held-out
   normal-use benchmarks before publishing.

This path is **not yet a validated full-model ARA implementation**.
In particular, Gemma 4 26B A4B's packed MoE projections and routing have
not been captured, optimized or GPU-integration-tested. There is also no
claim that post-quantization QAT quality is perfectly retained.

## Prerequisites

- Linux GPU server recommended; Windows PowerShell works for Python/uv
  commands, but compiling llama.cpp requires a C++ compiler and CMake.
- A local official Google Gemma 4 26B A4B QAT Q4_0 GGUF.
- A pinned checkout of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
  The CI compile reference is
  `fac3a894fc04901c3e5abaff457e024c6b52e04f`; pinning avoids
  callback API drift.
- Python v3.10+ with the project's `uv` dependencies.

Install:

```bash
uv sync --extra qat_gguf --extra decision_sidecar
uv pip install -p .venv/bin/python datasets
```

### 1. Build the Gemma-specific collector

```bash
cmake -S gemma4_qat_capture -B build-gemma4 \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gemma4 --target gemma4-qat-capture -j
```

The independent CMake project links against the pinned llama.cpp
`llama` and `llama-common` targets. The C++ collector reads the canonical
chat template from the original GGUF rather than hardcoding delimiters.

### 2. Export ARA training prompts

```bash
uv run python gemma4_qat_capture/prepare_prompts.py \
  --out runs/prompts-ja --limit 128
```

Inputs come from the existing two Japanese HF datasets, `train` split
only. The `test` split is reserved for normal-use and refusal evaluation.
Each prompt occupies one physical line in the capture input.

### 3. Collect + optimise + write NEW Q4_0 GGUF

```bash
uv run python gemma4_qat_capture/run_qat_ara.py \
  --model /models/gemma-4-26B-A4B-it-qat-q4_0.gguf \
  --collector ./build-gemma4/gemma4-qat-capture \
  --good runs/prompts-ja/good.txt \
  --bad runs/prompts-ja/bad.txt \
  --out runs/first-ara-q4 \
  --layers 0,1,2 \
  --gpu-layers 999 --batch 256 --ctx 8192 \
  --device cuda:0 --steps 15
```

This generates:
- `runs/first-ara-q4/captured/`: original GGUF runtime
  `kqv_out` activations, raw float32 and a shape/count manifest.
- `runs/first-ara-q4/io/`: checked ARA module I/O and source
  SHA-256 identity.
- `runs/first-ara-q4/candidate.gguf`: candidate Q4_0 GGUF,
  created in one batch transaction; no source file overwrite.
- `runs/first-ara-q4/run.json`: provenance and optimization losses.

If any requested tensor is NOT Q4_0, the pipeline aborts rather than
silently changing format or using a different checkpoint. Always inspect
the actual GGUF tensor types before choosing layers.

**Windows PowerShell example** (adapt model/collector paths):

```powershell
uv run python .\gemma4_qat_capture\run_qat_ara.py `
  --model "D:\Models\gemma-4-26B-A4B-it-qat-q4_0.gguf" `
  --collector ".\build-gemma4\Release\gemma4-qat-capture.exe" `
  --good ".\runs\prompts-ja\good.txt" `
  --bad ".\runs\prompts-ja\bad.txt" `
  --out ".\runs\first-ara-q4" `
  --layers "0,1,2" --gpu-layers 999 --device cuda:0
```

### 4. Evaluate the original GGUF AND the candidate

The d1-2B judge must be resident for the **entire** evaluation. The
`scripts/serve_gguf_d1_proxy.py` tool is shared and lives on `ara`.
For example, use two GPUs and bind the upstream llama-server to localhost:

```bash
llama-server -m /models/gemma-4-26B-A4B-it-qat-q4_0.gguf \
  --alias local --host 127.0.0.1 --port 8080 --n-gpu-layers 999
uv run python scripts/serve_gguf_d1_proxy.py \
  --upstream http://127.0.0.1:8080 \
  --judge-device cuda:1 --port 8090 \
  --log runs/baseline-d1.jsonl
```

For the candidate, stop the upstream model, restart llama-server with
`runs/first-ara-q4/candidate.gguf`, and keep the judge resident.
Do not change chat templates, prompt formatting, generation hyperparameters
or test data between runs.

**The proxy only monitors requests routed through port 8090.** Requests
sent directly to llama-server bypass it. Judge thresholds are provisional,
so also compare task accuracy, EOS/empty responses and independent
quantized next-token divergence.

## Current validity boundaries

- The callback captures **last-token attention `kqv_out`**. This is
  the correct pre-projection input, but it is not a full forward-hook
  framework for arbitrary Gemma 4 experts. Full packed MoE still needs
  expert routing, scaled projections and correct per-expert I/O.
- `good_y`/`bad_y` are reconstructed raw-matmul outputs, **not** the
  actual attention sublayer's final output after `wo_s` and normalization.
- Q4_0 gradients use STE and a layer-local surrogate; actual inference
  is needed to decide whether the result is beneficial.
- The source and compiled reference `ggml` Q4_0 encoders need a
  conformance test to guarantee identical rounding on the target hardware.
- A CPU C++ compilation job and synthetic Python tests have been added;
  **real 26B A4B GGUF GPU execution remains unverified**.
- Do not describe this as fully production-ready until
  baseline/candidate measurements and rollback have passed.
