#!/usr/bin/env bash
# Serve the paper's open-source judge, Qwen3-Next-80B-A3B-Thinking, as an
# OpenAI-compatible endpoint (http://localhost:${PORT}/v1) with vLLM, under the
# name expected by `pink grade --judge qwen3-next-80b`.
#
#   bash scripts/serve_qwen3_judge.sh                      # 8x 80GB GPUs, BF16 weights
#   MODEL=Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 TP=4 \
#     bash scripts/serve_qwen3_judge.sh                    # 4x 48GB GPUs (e.g. L40S), official FP8 weights
#
#   pink grade --judge qwen3-next-80b --input manifest.jsonl --out oracle_grades.csv
#
# Notes (vllm 0.13.0):
# * The judge's reasoning trace often exceeds 7k tokens. With --max-model-len 8192, about a third
#   of the responses were cut off before the JSON (pink retries them, up to 5 times). With 16384,
#   39 of 40 test gradings succeeded on the first attempt (4x L40S, FP8). If you run out of
#   memory, lower MAX_LEN to 8192 (the setting used for the paper's runs on 8x H100).
# * On GPUs connected only via PCIe (e.g. L40S), NCCL can hang at start-up; this is avoided by
#   NCCL_P2P_DISABLE=1 (set below unless you override it).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Thinking}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.75}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec vllm serve "${MODEL}" \
  --served-model-name Qwen/Qwen3-Next-80B-A3B-Thinking \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs 8 \
  --port "${PORT}"
