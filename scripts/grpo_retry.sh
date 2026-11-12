#!/bin/bash
# GRPO run that resumes from the latest checkpoint after an (8 GB) OOM crash: scripts/grpo_retry.sh <train args>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/grpo-qwen3-4b
for i in 1 2 3 4 5 6 7 8; do
  CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1)
  python train/grpo_unsloth.py --out $OUT "$@" ${CK:+--resume $CK} >> grpo_full.log 2>&1 && break
done
