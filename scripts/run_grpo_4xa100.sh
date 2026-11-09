#!/usr/bin/env bash
# Stage 2: GRPO (RLVR) with a vLLM generation server on the last GPU, training on the others.
# Usage: bash scripts/run_grpo_4xa100.sh [extra TRL args]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/grpo_qwen3_8b.yaml}
MODEL=${MODEL:-outputs/sft-qwen3-8b-merged}
OUT=${OUT:-outputs/grpo-qwen3-8b}
VLLM_GPU=${VLLM_GPU:-3}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2}
NPROC=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/logs
CUDA_VISIBLE_DEVICES=$VLLM_GPU nohup trl vllm-serve --model "$MODEL" --max_model_len 10240 \
  --gpu_memory_utilization 0.9 > outputs/logs/vllm_serve.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
echo "waiting for vLLM server (pid $VLLM_PID)..."
for i in $(seq 1 120); do
  curl -sf http://localhost:8000/health/ >/dev/null 2>&1 && break
  sleep 5
done

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes "$NPROC" \
  train/grpo.py --config "$CFG" --model_name_or_path "$MODEL" --output_dir "$OUT" "$@"

python train/merge.py --base "$MODEL" --adapter "$OUT" --out "${OUT}-merged"
echo "GRPO done: adapter=$OUT merged=${OUT}-merged"