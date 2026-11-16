#!/bin/bash
# Unsloth 4-bit QLoRA SFT (train/sft_unsloth.py); defaults replicate outputs/sft-cot-qwen3-4b. Final adapter + ONE final checkpoint
# (optimizer/scheduler/state), no intermediate checkpoints. Skips if the adapter exists (unless --max-steps smoke).
#   sft_qlora.sh --machine M --run R [--max-steps 3 --sft-out outputs/smoke-sft] [--sft-max-len 6144] [--sft-lr ...]
source "$(dirname "$0")/lib.sh"; init "$@"
SFT_OUT=${SFT_OUT}; B=$(basename $SFT_OUT)
[ -z "$MAX_STEPS" ] && [ -f $SFT_OUT/adapter_model.safetensors ] && { log sft INFO "skip (adapter exists: $SFT_OUT)"; exit 0; }
if [ "$SFT_MAX_LEN" = auto ]; then   # cover the data: max rendered token length rounded up to 512
  SFT_MAX_LEN=$(py $ENV_UNSLOTH scripts/pipeline/token_len.py $SFT_BASE $SFT_TRAIN "${SFT_CHAT_KW:-{\}}" | tee $A100_DIR/token_len.txt | awk '/^max_len/{print $2}')
  log sft-tokens INFO "$(grep -v max_len $A100_DIR/token_len.txt | tr '\n' ' ') -> max_len=$SFT_MAX_LEN"
fi
F="--model $SFT_BASE --load_in_4bit --train $SFT_TRAIN --out $SFT_OUT --max_len $SFT_MAX_LEN --r $SFT_R --epochs $SFT_EPOCHS --lr $SFT_LR --bs $SFT_BS --ga $SFT_GA --val_frac $SFT_VAL_FRAC --log_steps 2 --disable_tqdm --save_state"
[ "$ENABLE_THINKING" = 1 ] && F="$F --enable_thinking"
[ -n "$SFT_INIT_ADAPTER" ] && F="$F --init_adapter $SFT_INIT_ADAPTER"
[ -n "$SFT_WARMUP" ] && F="$F --warmup_ratio $SFT_WARMUP"
[ -n "$MAX_STEPS" ] && F="$F --max_steps $MAX_STEPS --longest_first 16"
N=${MAX_STEPS:-64}
rm -f $A100_DIR/.vram
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> $A100_DIR/.vram; sleep 15; done ) & VP=$!
( sleep 60; while true; do python3 scripts/pipeline/sft_eta.py $SFT_OUT/progress/metrics.jsonl $N | while read l; do log sft-train INFO "$l"; done; sleep 300; done ) & EP=$!
stage sft-${MAX_STEPS:+smoke-}$B py $ENV_UNSLOTH train/sft_unsloth.py $F; rc=$?
kill $VP $EP 2>/dev/null
log sft-$B INFO "peak_vram_MiB=$(sort -n $A100_DIR/.vram 2>/dev/null | tail -1) progress=$SFT_OUT/progress/metrics.jsonl"
exit $rc
