#!/usr/bin/env bash
# Serve Qwen3.8-27B, the open judge closest to GPT-5, as an OpenAI-compatible endpoint
# (http://localhost:${PORT}/v1) with vLLM, under the name expected by `pink grade --judge qwen3.8-27b`.
#
#   bash scripts/serve_qwen38_judge.sh             # two 48 GB GPUs (TP=2)
#   DP=2 bash scripts/serve_qwen38_judge.sh        # four GPUs: two TP=2 replicas behind one endpoint
#
#   pink grade --judge qwen3.8-27b --input transcriptions.jsonl --out grades.csv --workers 64   # 64 per replica
#
# Notes (vLLM 0.19.1, 4x L40S):
# * BF16 weights (~56 GB) do not fit one 48 GB GPU, hence TP=2.
# * The preset keeps thinking on; the judge reasons for a few thousand tokens before the JSON (16k context).
# * On GPUs connected only via PCIe, NCCL can hang at start-up; NCCL_P2P_DISABLE=1 avoids it.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
TP="${TP:-2}"
DP="${DP:-1}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-64}"   # concurrent requests per replica
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec vllm serve "${MODEL}" \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_SEQS}" \
  --port "${PORT}"
