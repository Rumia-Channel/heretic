# Direct Q4_0 GGUF editing in ARA

This shared feature is developed on **\`ara\`** and then brought into
\`ara-gemma4-26BA4B-qat\`. The intended baseline is Google's **original,
already-quantized QAT Q4_0 GGUF**. Nothing in these tools requires converting
the whole checkpoint to BF16 or replacing it with a bitsandbytes/NF4 model.

> Status: Q4_0 tensor byte editing and Q4_0-aware *layer-local* ARA
> optimization are implemented. Automated per-layer activation extraction
> from llama.cpp/Gemma 4 packed MoE, model-wide optimization and real
> checkpoint validation are **not** implemented yet.

## How Q4_0 is edited

Each Q4_0 block occupies 18 bytes: a signed FP16 scale and 16 bytes of
paired 4-bit codes encoding 32 weights. The change is calculated on
**dequantized values of that tensor as stored in the GGUF**, then the changed
blocks are re-encoded with the Q4_0 reference formula.

- **Untouched blocks are kept byte-identical**, including original Google QAT
  quantization decisions.
- Other tensors and metadata are copied unchanged, including tensors stored
  in F16/F32 or another quantization type.
- Only explicit Q4_0 tensor names are accepted; unsupported or mismatched
  tensor types are rejected.
- Source GGUF is opened read-only; an output is created under a different
  path in a no-overwrite transaction.
- The delta's \`.npy\` shape must equal the **GGUF reader shape** (reversed
  tensor dimensions). No automatic transpose from PyTorch is guessed.
- Memory use is limited to a tensor or configured block window rather than
  dequantizing the whole 26B checkpoint.

### Windows PowerShell — prerequisites

\`\`\`powershell
git fetch origin
git switch ara-gemma4-26BA4B-qat
uv sync --extra qat_gguf --extra decision_sidecar --dev
\`\`\`

For the unmodified official GGUF (adjust the filename):

\`\`\`powershell
$model = "D:\Models\gemma-4-26B-A4B-it-qat-q4_0.gguf"
uv run python scripts/inspect_qat_gguf.py --model $model --sha256
\`\`\`

The inventory reports which tensors are actually Q4_0. Record the source
hash as the immutable baseline.

### Apply a precomputed delta to ONE tensor

Given a C-contiguous float32 \`.npy\` with the exact tensor shape:

\`\`\`powershell
uv run python scripts/patch_gguf_q4_0.py \`
  --model $model \`
  --output "D:\Models\gemma-ara-candidate.gguf" \`
  --tensor "blk.0.attn_output.weight" \`
  --delta ".\deltas\blk0.npy" \`
  --max-abs-delta 0.25
\`\`\`

The tensor name is an example only. Always read actual names from the GGUF.
A delta must come from a valid ARA procedure or another justified experiment.
The command **only applies the delta**; it does not collect activations.

### Multiple tensors — ONE model copy

Create a file \`patches.json\`, with entries such as:

\`\`\`json
[
  {"tensor": "blk.0.attn_output.weight", "delta": "deltas/blk0.npy", "scale": 1.0},
  {"tensor": "blk.1.attn_output.weight", "delta": "deltas/blk1.npy", "scale": 1.0}
]
\`\`\`

Paths in this manifest are resolved relative to its directory.

\`\`\`powershell
uv run python scripts/patch_gguf_q4_0_manifest.py \`
  --model $model \`
  --output "D:\Models\gemma-ara-multi.gguf" \`
  --manifest ".\patches.json"
\`\`\`

The whole model is copied once. Changes are applied only to the selected
encoded payload offsets and then atomically published as a new GGUF.

### Q4_0-aware local ARA optimizer

This optimizer starts from the dequantized values of a **selected original
Q4_0 tensor** and performs ARA's good-output preservation / nearest-neighbor
bad-output steering, with Q4_0 fake quantization in each forward pass.

It requires a pre-collected \`.npz\` archive with four float arrays:

| key | shape | meaning |
|---|---|---|
| \`good_x\` | \`[N_good, in_features]\` | Input to this exact Q4_0 projection |
| \`good_y\` | \`[N_good, out_features]\` | **Original** GGUF projection output |
| \`bad_x\` | \`[N_bad, in_features]\` | Input to the same projection |
| \`bad_y\` | \`[N_bad, out_features]\` | **Original** GGUF projection output |

The arrays must be collected from the **same model revision, tensor and
quantization format**. The ARA objective assumes the selected module
behaves as \`y = x @ W.T\` without hidden bias or nonlinearity.

\`\`\`powershell
uv run python scripts/ara_gguf_q4_0.py \`
  --model $model \`
  --output "D:\Models\gemma-ara-local.gguf" \`
  --tensor "blk.0.attn_output.weight" \`
  --module-io ".\io\blk0.npz" \`
  --device cuda:0 \`
  --steps 15
\`\`\`

**IMPORTANT**: Heretic's existing \`get_module_io()\` uses Transformers
forward hooks, not llama.cpp's GGUF runtime. The collection of these module
arrays from the real QAT GGUF and correctly mapping Gemma 4 packed-MoE
experts are **blocking development tasks**. Do not make up activations from
another checkpoint to bypass this limitation.

## Always-on LiquidAI/d1-2B monitoring

d1-2B is a decision model (\`system_one\`), not a chat-completions LLM.
Run it on a separate GPU and keep it loaded. The **resident monitoring
proxy** is used for all chat completion calls in the deployment path.

Launch llama.cpp's original QAT model bound to localhost, for example:

\`\`\`powershell
llama-server -m $model --alias local --host 127.0.0.1 --port 8080 -ngl 999
\`\`\`

On a different terminal, start the resident d1 monitoring proxy:

\`\`\`powershell
uv run python scripts/serve_gguf_d1_proxy.py \`
  --upstream "http://127.0.0.1:8080" \`
  --port 8090 \`
  --judge-device cuda:1 \`
  --interval-chars 256 \`
  --log ".\logs\gguf-d1.jsonl"
\`\`\`

Send requests to **\`http://127.0.0.1:8090/v1/chat/completions\`**, not
to the upstream port 8080. The proxy requires \`"stream": true\`, forwards
the streaming output, scores it periodically, checks the full response on
completion, and records four probabilities (refusal, looping, incomplete,
off-topic). Use \`--stop-on-loop\` to optionally abort a streaming request
if the loop probability reaches \`--loop-threshold\` (default 0.9).

The proxy starts the d1 model **before** accepting requests, keeps it
resident, and serializes inference calls. It binds to localhost by default.
For remote clients, put authenticated TLS infrastructure in front of the
proxy and make the **upstream port inaccessible** to those clients.
Requests that bypass the proxy cannot be monitored.

\`--save-text\` enables prompt/response logging. Without it, the logs contain
prompt hashes, timestamps, stop reasons and d1 scores instead of full text.
Protect logs when evaluating sensitive prompts.

This proxy is an **audit and control path**, not evidence of QAT capability
preservation. Calibrate d1 probabilities on Japanese answers and verify
model quality against the untouched official GGUF.

## Blocking issues before production use

1. Verify Python-encoded Q4_0 blocks byte-for-byte against compiled ggml.
2. Capture input/output activations directly from the official GGUF runtime,
   including packed expert routing and exact tensor-order mapping.
3. Combine multi-layer ARA optimization with a model-wide inference loop
   instead of only a layer-local surrogate loss.
4. Test the edited candidate in llama.cpp and compare generation quality,
   looping, empty output, instruction following and refusal across held-out
   Japanese prompts. Sidecar thresholds are provisional.
5. Verify rollback and output loading on the real 26B GGUF under GPU load.
6. Pin source checkpoint, model code and runtime revisions; record model hash.
7. Run GPU integration tests. CPU mock/unit tests alone are insufficient.
