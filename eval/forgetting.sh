#!/usr/bin/env bash
# Forgetting check: general benchmarks before/after fine-tuning (lm-eval + vLLM backend).
# Usage: bash eval/forgetting.sh <model_path> [tp=1]
# Compare against the base model; a drop > ~1-2 points means: add replay data, lower LR/epochs, or merge with --scale < 1.
set -euo pipefail
MODEL=$1; TP=${2:-1}
NAME=$(basename "$MODEL")
TASKS=${TASKS:-ifeval,gsm8k,mmlu,arc_challenge,hellaswag}
# Multilingual (check names with: lm_eval --tasks list | grep -i -E "global_mmlu|belebele"):
#   TASKS="$TASKS,global_mmlu_full_he,global_mmlu_full_fa,belebele_heb_Hebr,belebele_pes_Arab"
lm_eval --model vllm \
  --model_args pretrained="$MODEL",tensor_parallel_size="$TP",dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192 \
  --tasks "$TASKS" --batch_size auto --apply_chat_template --fewshot_as_multiturn \
  --output_path "outputs/forgetting/$NAME"