#!/usr/bin/env bash
# Stage 1: LoRA SFT on 4xA100 (ZeRO-2 DDP), then merge the adapter for GRPO / vLLM serving.
# Usage: bash scripts/run_sft_4xa100.sh [extra TRL args, e.g. --model_name_or_path Qwen/Qwen3.5-9B --loss_type dft]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/sft_qwen3_8b.yaml}
OUT=${OUT:-outputs/sft-qwen3-8b}
BASE=${BASE:-Qwen/Qwen3-8B}
NGPU=${NGPU:-4}
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes "$NGPU" \
  train/sft.py --config "$CFG" --output_dir "$OUT" --model_name_or_path "$BASE" "$@"

python train/merge.py --base "$BASE" --adapter "$OUT" --out "${OUT}-merged"
echo "SFT done: adapter=$OUT merged=${OUT}-merged"