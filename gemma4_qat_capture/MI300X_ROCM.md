# AMD Developer Cloud MI300X — QAT Q4_0 ARA + resident d1-2B

This is an **experimental**, Gemma 4 **26B A4B-specific** entrypoint.
The generic Q4_0 writer/optimizer and resident d1 monitoring library remain
on the shared `ara` branch. This launcher and its datasets/model mapping
remain ONLY on `ara-gemma4-26BA4B-qat`.

## One command

On Ubuntu Linux with ROCm and a working **ROCm PyTorch+torchvision**
installation (for example, AMD's ROCm PyTorch developer image):

```bash
git fetch origin
git switch ara-gemma4-26BA4B-qat
bash gemma4_qat_capture/run_mi300x_rocm.sh
```

That single Bash script handles project Python dependencies, pins and builds
a llama.cpp HIP backend for **MI300X gfx942**, downloads the original
Google QAT Q4_0 GGUF when missing, exports separate Japanese
train/test-split prompts, creates one Q4_0 candidate, and compares all
held-out generations with resident `LiquidAI/d1-2B`.

You need a working ROCm installation (hipconfig, rocminfo), build tools
(CMake, C++ compiler, git, curl), and preinstalled ROCm versions of
torch and torchvision. If uv is absent, the launcher bootstraps pinned uv 0.13.7 from the official Astral installer. It **fails early** if Python sees a CUDA/CPU wheel
instead of a ROCm torch wheel; it does not silently install another
backend. HF_TOKEN may be required for the official model license/access.

### Process topology (default: 3 MI300X GPUs)

```text
MI300X GPU 0       MI300X GPU 1           MI300X GPU 2
-------------      -----------------      -----------------------
official GGUF      official GGUF           LiquidAI/d1-2B
cb_eval capture    llama-server            always resident
Q4_0 STE ARA  <--> d1-proxy -> test prompts -> d1 decisions
new GGUF write     test-stream eval       refusal/looping/etc
   [ARA + baseline evaluation run concurrently]
        |
        v
candidate GGUF ---> SAME llama-server port --> SAME resident d1
        |
        v
comparison.json (failed/no-op/regression = nonzero exit status)
```

All decisions are generated through the proxy, not through direct
llama-server calls. A proxy log verdict is REQUIRED for each response; if
d1 fails/OOMs, the evaluation stops rather than treating it as success.
The d1 process remains alive across the official/candidate server restart.
By default it checks every 256 generated characters and once at the end.
It is a post-hoc/streaming monitor, NOT a differentiable loss and NOT an
automatic weight-update controller.

On Linux the launcher isolates each worker using
`ROCR_VISIBLE_DEVICES=N`, so the ROCm PyTorch device within that process
is called `cuda:0`. This naming is normal for AMD ROCm PyTorch.
The script requires 3 distinct devices by default.

For a 2-GPU instance, sharing the original GGUF inference GPU with ARA
is possible but can cause OOM/contention:

```bash
GPU_ARA=0 GPU_INFER=0 GPU_JUDGE=1 ALLOW_SHARED_GPU=1 \
  bash gemma4_qat_capture/run_mi300x_rocm.sh
```

For a separate local official model path:

```bash
GGUF_PATH=/data/models/gemma-4-26B_q4_0-it.gguf \
  bash gemma4_qat_capture/run_mi300x_rocm.sh
```

### Parameters

| Environment | Default | Description |
|---|---|---|
| `GPU_ARA` / `GPU_INFER` / `GPU_JUDGE` | 0 / 1 / 2 | Physical MI300X IDs |
| `RUN_DIR` | `runs/mi300x-UTC-timestamp` | Per-run output directory; no overwrite |
| `GGUF_PATH` | `models/gemma-4-26B_q4_0-it.gguf` | Exact Google official QAT Q4_0 |
| `LLAMA_CPP_DIR` | `.deps/llama.cpp` | Pinned llama.cpp source |
| `LAYERS` | `auto` | Comma-separated Attention indices or auto |
| `LAYER_COUNT` | 3 | Number of Q4_0 attention layers for auto selection |
| `TRAIN_LIMIT` | 128 per group | Train-split ARA activation prompts |
| `EVAL_LIMIT` | 24 per group | Held-out test-split d1 prompts (smoke testing, NOT adequate final benchmark) |
| `ARA_STEPS` | 15 | STE ARA optimizer steps per selected layer |
| `ARA_LR` | 0.0005 | Optimizer learning rate |
| `ARA_MAX_ABS_DELTA` | 0.05 | Maximum per-weight update |
| `EVAL_MAX_TOKENS` | 384 | Generation limit for BOTH evaluations |
| `JUDGE_THRESHOLD` | 0.75 | Provisional d1 yes/no cutoff |
| `ROCM_PYTHON` | python3 | System Python with compatible ROCm torch+torchvision |

### Outputs

The created run directory contains:

- `source.json`: original GGUF inventory and SHA256.
- `train-sources.json`, `test-sources.json`: separate HF dataset splits.
- `ara/captured/`: REAL llama.cpp `kqv_out` float32 input vectors.
- `ara/io/`: verified per-layer input/output data from the same GGUF.
- `ara/candidate.gguf`: new GGUF with selected Q4_0 weight blocks edited;
  the original file stays untouched.
- `baseline.json` and `candidate.json`: per-prompt decision probabilities.
- `baseline-d1.jsonl` / `candidate-d1.jsonl`: streaming monitor logs.
- `comparison.json`: refusal improvement and looping/empty/EOS regression
  checks; `regression_detected` aborts with nonzero exit status.
- `ara.log`, `llama-server.log`, `d1-sidecar.log`: troubleshooting logs.

The script aborts if all encoded Q4_0 blocks are unchanged or if any
d1 evaluation is missing. The **candidate is still experimental** even
if the d1 comparison has no flags. A separate independently scored
math/code/Japanese-writing evaluation, token-level KL, and long-context
checks are needed before publishing.

### Technical boundaries

1. The Gemma 4 capture currently targets `blk.N.attn_output.weight`
   matrices; **packed MoE Experts are NOT covered**.
2. ARA is a local, quantization-aware STE objective; no exact gradient
   through ggml's discrete 4-bit integer codes is claimed.
3. ARA optimization runs on **one** MI300X, concurrently with baseline
   model inference and d1 judgements on other MI300Xs. Selected ARA
   layers are not themselves distributed across multiple workers.
4. d1 results are not continuously generated for the candidate while its
   weights are still being optimized; this requires intermediate
   checkpoint generation. The baseline is monitored concurrently, then
   the finished candidate is monitored with the same resident d1.
5. The d1 Hugging Face model executes its custom checkpoint code via
   `trust_remote_code=True`; review it and pin a revision for
   security-sensitive deployments. Benchmarks on AMD MI325X are not
   proof of compatibility on your exact MI300X image.
6. No real AMD Developer Cloud MI300X build or model-wide end-to-end
   acceptance test has been run in this chat. CI includes a CPU-only
   llama.cpp build and mocked decision checks, but not ROCm integration.

If the launcher fails, check the run logs and consult
[GEMMA4_QAT_ACTIVATION_PIPELINE.md](../GEMMA4_QAT_ACTIVATION_PIPELINE.md).
