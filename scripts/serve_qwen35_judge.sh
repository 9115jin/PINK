#!/usr/bin/env bash
# Serve the light open-source judge, Qwen3.5-9B, as an OpenAI-compatible endpoint
# (http://localhost:${PORT}/v1) with vLLM, under the name expected by `pink grade --judge qwen3.5-9b`.
#
#   bash scripts/serve_qwen35_judge.sh             # one GPU (24-48 GB)
#   DP=4 bash scripts/serve_qwen35_judge.sh        # four GPUs, one replica each (data parallel)
#
#   pink grade --judge qwen3.5-9b --input transcriptions.jsonl --out grades.csv --workers 64   # 64 per replica
#
# Notes (vLLM 0.19.1, 4x L40S):
# * The preset keeps thinking on. The judge reasons for a few thousand tokens before the JSON, so the
#   context is 16k. Without thinking it scores almost only 0 or 20 per component and agrees less with GPT-5.
# * Throughput: with DP=4 on 4x L40S, the 2,244 FERMAT oracle solutions were graded in 58 min.
# * On GPUs connected only via PCIe, NCCL can hang at start-up; NCCL_P2P_DISABLE=1 avoids it.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DP="${DP:-1}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-64}"   # concurrent requests per replica
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec vllm serve "${MODEL}" \
  --served-model-name Qwen/Qwen3.5-9B \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_SEQS}" \
  --port "${PORT}"
