#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# AMD MI300X/ROCm: run official QAT Q4_0 GGUF ARA while a resident
# LiquidAI/d1-2B judges the ORIGINAL GGUF concurrently. Then judge candidate.
# This produces a CANDIDATE, not a proven quality-preserving final model.
set -Eeuo pipefail
umask 077

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_PATH="${GGUF_PATH:-$ROOT/models/gemma-4-26B_q4_0-it.gguf}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/mi300x-$(date -u +%Y%m%dT%H%M%SZ)}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/.deps/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-fac3a894fc04901c3e5abaff457e024c6b52e04f}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gemma4-mi300x}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv-rocm-mi300x}"
ROCM_PYTHON="${ROCM_PYTHON:-python3}"
GPU_ARA="${GPU_ARA:-0}"
GPU_INFER="${GPU_INFER:-1}"
GPU_JUDGE="${GPU_JUDGE:-2}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
LAYERS="${LAYERS:-auto}"
LAYER_COUNT="${LAYER_COUNT:-3}"
TRAIN_LIMIT="${TRAIN_LIMIT:-128}"
EVAL_LIMIT="${EVAL_LIMIT:-24}"
ARA_STEPS="${ARA_STEPS:-15}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-384}"
JUDGE_THRESHOLD="${JUDGE_THRESHOLD:-0.75}"
JUDGE_INTERVAL_CHARS="${JUDGE_INTERVAL_CHARS:-256}"
INFER_PORT="${INFER_PORT:-18080}"
PROXY_PORT="${PROXY_PORT:-18090}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUILD_JOBS="${BUILD_JOBS:-8}"
server_pid=""
proxy_pid=""
baseline_pid=""

log() { printf '[MI300X-ARA] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing executable: $1"; }
cleanup() {
  local result=$?
  trap - EXIT INT TERM
  for pid in "$baseline_pid" "$proxy_pid" "$server_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "$baseline_pid" "$proxy_pid" "$server_pid"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  if ((result)); then
    log "FAILED (exit=$result). Logs: $RUN_DIR"
    log "Original QAT GGUF is unchanged. Do not deploy an unverified candidate."
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# On Linux ROCm use ROCr's device mask exclusively. Each worker then sees
# its selected MI300X as CUDA-style device cuda:0 in ROCm PyTorch.
on_gpu() {
  local gpu="$1"; shift
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="$gpu" "$@"
}
wait_ready() {
  local url="$1" pid="$2" role="$3"
  for ((i=0; i<240; ++i)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      log "$role ready"; return
    fi
    kill -0 "$pid" 2>/dev/null || die "$role exited before ready"
    sleep 2
  done
  die "$role health check timed out"
}
for exe in git cmake curl uv hipconfig rocminfo "$ROCM_PYTHON" find realpath; do need "$exe"; done
for value in "$GPU_ARA" "$GPU_INFER" "$GPU_JUDGE" "$LAYER_COUNT" \
  "$TRAIN_LIMIT" "$EVAL_LIMIT" "$ARA_STEPS" "$EVAL_MAX_TOKENS" \
  "$INFER_PORT" "$PROXY_PORT" "$CONTEXT_SIZE" "$BATCH_SIZE"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "Expected non-negative integer: $value"
done
((LAYER_COUNT>0 && TRAIN_LIMIT>0 && EVAL_LIMIT>0 && ARA_STEPS>0)) ||
  die "Counts must be positive"
(( INFER_PORT != PROXY_PORT )) || die "Inference and proxy ports must differ"
[[ ! -e "$RUN_DIR" ]] || die "RUN_DIR already exists: $RUN_DIR"
if [[ "$GPU_ARA" == "$GPU_INFER" || "$GPU_ARA" == "$GPU_JUDGE" ||
      "$GPU_INFER" == "$GPU_JUDGE" ]]; then
  [[ "$ALLOW_SHARED_GPU" == "1" ]] ||
    die "Set three distinct GPU IDs, or explicitly ALLOW_SHARED_GPU=1"
  log "WARNING: shared MI300X can cause OOM/contention"
fi

log "Checking AMD ROCm PyTorch (no silent CUDA wheel fallback)"
"$ROCM_PYTHON" - <<'PY'
import torch
assert torch.version.hip and torch.cuda.is_available(), (
    "Use an AMD ROCm PyTorch environment with visible MI300X GPUs"
)
print("ROCm", torch.version.hip, "GPU count", torch.cuda.device_count())
PY
if [[ ! -e "$VENV_DIR/bin/python" ]]; then
  uv venv --system-site-packages --python "$ROCM_PYTHON" "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
uv pip install --python "$PYTHON" --no-deps -e "$ROOT"
uv pip install --python "$PYTHON" "gguf" "datasets>=4" \
  "transformers>=5.19" "huggingface-hub" "pillow" "numpy>=2.2,<3"
"$PYTHON" - <<'PY'
import torch
assert torch.version.hip and torch.cuda.is_available(), "ROCm torch missing inside venv"
try:
    import torchvision
except Exception as exc:
    raise RuntimeError(
        "ROCm-matched torchvision is required; use AMD's ROCm PyTorch image"
    ) from exc
print("torch", torch.__version__, "HIP", torch.version.hip)
PY
for gpu in "$GPU_ARA" "$GPU_INFER" "$GPU_JUDGE"; do
  on_gpu "$gpu" "$PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 1, "ROCR_VISIBLE_DEVICES must expose one GPU"
arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
assert "gfx942" in arch, "Expected MI300X gfx942, got " + repr(arch)
print("MI300X", torch.cuda.get_device_name(0), arch)
PY
done

# Official Google QAT Q4_0 artifact (not QAT-unquantized / BF16).
if [[ ! -s "$GGUF_PATH" ]]; then
  log "Downloading official Google 26B A4B QAT Q4_0 GGUF (~14.4 GB)"
  mkdir -p "$(dirname "$GGUF_PATH")"
  "$PYTHON" - "$GGUF_PATH" <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import sys
path = Path(sys.argv[1]).resolve()
name = "gemma-4-26B_q4_0-it.gguf"
if path.name != name:
    raise SystemExit("Auto-download needs GGUF_PATH ending in " + name)
print(hf_hub_download(
    repo_id="google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
    filename=name, local_dir=str(path.parent)))
PY
fi
[[ -s "$GGUF_PATH" ]] || die "Missing official GGUF"
GGUF_PATH="$(realpath "$GGUF_PATH")"

# Pin the llama.cpp source because cb_eval callback interfaces may change.
if [[ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_REF"
fi
current_ref="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
[[ "$current_ref" == "$LLAMA_CPP_REF" ]] ||
  die "llama.cpp checkout must be pinned to $LLAMA_CPP_REF (found $current_ref)"
[[ -z "$(git -C "$LLAMA_CPP_DIR" status --porcelain)" ]] ||
  die "llama.cpp checkout is dirty"

log "Compiling Gemma 4 capture + llama-server with gfx942 HIP backend"
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S "$ROOT/gemma4_qat_capture" -B "$BUILD_DIR" \
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" \
  --target gemma4-qat-capture llama-server --parallel "$BUILD_JOBS"
collector="$(find "$BUILD_DIR" -type f -name gemma4-qat-capture -executable -print -quit)"
server="$(find "$BUILD_DIR" -type f -name llama-server -executable -print -quit)"
[[ -n "$collector" && -n "$server" ]] || die "Missing built llama.cpp binaries"

if [[ "$LAYERS" == "auto" ]]; then
  LAYERS="$("$PYTHON" - "$GGUF_PATH" "$LAYER_COUNT" <<'PY'
from gguf import GGUFReader
import sys, re
reader = GGUFReader(sys.argv[1])
rx = re.compile(r"blk\.(\d+)\.attn_output\.weight")
found = sorted(int(match.group(1)) for tensor in reader.tensors
               if tensor.tensor_type.name == "Q4_0"
               if (match := rx.fullmatch(tensor.name)))
limit = int(sys.argv[2])
if len(found) < limit:
    raise SystemExit("Official GGUF lacks requested count of Q4_0 attention layers")
print(",".join(map(str, found[:limit])))
PY
)"
fi
log "ARA layers: $LAYERS"
mkdir -p "$RUN_DIR"
printf '%s\n' "$GGUF_PATH" >"$RUN_DIR/original-model.txt"
printf '%s\n' "$LLAMA_CPP_REF" >"$RUN_DIR/llama-cpp-ref.txt"
"$PYTHON" "$ROOT/scripts/inspect_qat_gguf.py" \
  --model "$GGUF_PATH" --sha256 >"$RUN_DIR/source.json"
"$PYTHON" "$ROOT/gemma4_qat_capture/prepare_prompts.py" \
  --out "$RUN_DIR/train" --split train --limit "$TRAIN_LIMIT" \
  >"$RUN_DIR/train-sources.json"
"$PYTHON" "$ROOT/gemma4_qat_capture/prepare_prompts.py" \
  --out "$RUN_DIR/test" --split test --limit "$EVAL_LIMIT" \
  >"$RUN_DIR/test-sources.json"

start_server() {
  on_gpu "$GPU_INFER" "$server" \
    --model "$1" --alias local --host 127.0.0.1 \
    --port "$INFER_PORT" --ctx-size "$CONTEXT_SIZE" -ngl 999 \
    >"$RUN_DIR/llama-server.log" 2>&1 &
  server_pid=$!
  wait_ready "http://127.0.0.1:$INFER_PORT/health" "$server_pid" "GGUF llama-server"
}

start_server "$GGUF_PATH"
log "Loading resident d1-2B on GPU $GPU_JUDGE"
on_gpu "$GPU_JUDGE" "$PYTHON" "$ROOT/scripts/serve_gguf_d1_proxy.py" \
  --upstream "http://127.0.0.1:$INFER_PORT" \
  --port "$PROXY_PORT" --judge-device cuda:0 \
  --interval-chars "$JUDGE_INTERVAL_CHARS" \
  --log "$RUN_DIR/d1-live.jsonl" \
  >"$RUN_DIR/d1-sidecar.log" 2>&1 &
proxy_pid=$!
wait_ready "http://127.0.0.1:$PROXY_PORT/healthz" "$proxy_pid" "resident d1 proxy"

log "Parallel execution: held-out baseline+d1 on GPUs $GPU_INFER/$GPU_JUDGE and ARA on $GPU_ARA"
"$PYTHON" "$ROOT/gemma4_qat_capture/eval_rocm_sidecar.py" \
  --proxy "http://127.0.0.1:$PROXY_PORT" --model local \
  --good "$RUN_DIR/test/good.txt" --bad "$RUN_DIR/test/bad.txt" \
  --proxy-log "$RUN_DIR/d1-live.jsonl" --out "$RUN_DIR/baseline.json" \
  --threshold "$JUDGE_THRESHOLD" --max-tokens "$EVAL_MAX_TOKENS" \
  >"$RUN_DIR/baseline.log" 2>&1 &
baseline_pid=$!

on_gpu "$GPU_ARA" "$PYTHON" "$ROOT/gemma4_qat_capture/run_qat_ara.py" \
  --model "$GGUF_PATH" --collector "$collector" \
  --good "$RUN_DIR/train/good.txt" --bad "$RUN_DIR/train/bad.txt" \
  --out "$RUN_DIR/ara" --layers "$LAYERS" \
  --gpu-layers 999 --batch "$BATCH_SIZE" --ctx "$CONTEXT_SIZE" \
  --device cuda:0 --steps "$ARA_STEPS" >"$RUN_DIR/ara.log" 2>&1

wait "$baseline_pid"
baseline_pid=""
[[ -s "$RUN_DIR/baseline.json" && -s "$RUN_DIR/ara/candidate.gguf" ]] ||
  die "Incomplete baseline or ARA candidate"

log "Evaluating Q4_0 candidate without reloading resident d1-2B"
kill "$server_pid"
wait "$server_pid" 2>/dev/null || true
server_pid=""
mv "$RUN_DIR/d1-live.jsonl" "$RUN_DIR/baseline-d1.jsonl"
start_server "$RUN_DIR/ara/candidate.gguf"
kill -0 "$proxy_pid" 2>/dev/null || die "Resident d1 monitor exited"
"$PYTHON" "$ROOT/gemma4_qat_capture/eval_rocm_sidecar.py" \
  --proxy "http://127.0.0.1:$PROXY_PORT" --model local \
  --good "$RUN_DIR/test/good.txt" --bad "$RUN_DIR/test/bad.txt" \
  --proxy-log "$RUN_DIR/d1-live.jsonl" --out "$RUN_DIR/candidate.json" \
  --threshold "$JUDGE_THRESHOLD" --max-tokens "$EVAL_MAX_TOKENS" \
  >"$RUN_DIR/candidate.log" 2>&1
[[ -s "$RUN_DIR/d1-live.jsonl" ]] || die "Candidate d1 log missing"
mv "$RUN_DIR/d1-live.jsonl" "$RUN_DIR/candidate-d1.jsonl"
"$PYTHON" "$ROOT/gemma4_qat_capture/compare_rocm_evals.py" \
  --base "$RUN_DIR/baseline.json" --candidate "$RUN_DIR/candidate.json" \
  --out "$RUN_DIR/comparison.json" >"$RUN_DIR/comparison.log"
log "Candidate: $RUN_DIR/ara/candidate.gguf"
log "Comparison: $RUN_DIR/comparison.json"
log "Experimental only: independent math/code and KL evaluations still required."
