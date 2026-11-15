#!/bin/bash
# 4-bit QLoRA SFT with Unsloth, replicating outputs/sft-cot-qwen3-4b (r=32, alpha=32, 7 proj, 2 epochs=64 steps, lr 1e-4,
# eff. batch 16 = bs 1 x ga 16, thinking on, val_frac 0.08). Writes final adapter + ONE final checkpoint (optimizer/scheduler/state).
#   OUT=outputs/sft-cot-qwen3.8-27b [BASE=unsloth/Qwen3.8-27B-unsloth-bnb-4bit] [MAX_LEN=auto] [MAX_STEPS=3 (smoke)]
cd ${REPO:-/NLP/relations_extraction_llm_finetune}
source scripts/a100/lib.sh
export HF_HOME=${HF_HOME:-/data} HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-outputs/sft-cot-qwen3.8-27b}; BASE=${BASE:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}
TRAIN=${TRAIN:-data/synth_hefa/sft_cot_train.jsonl}
if [ -z "$MAX_LEN" ]; then   # cover the data: max token length rounded up to 512
  MAX_LEN=$(py re-unsloth scripts/a100/token_len.py $BASE $TRAIN | tee $A100_DIR/token_len.txt | awk '/^max_len/{print $2}')
  log sft-tokens INFO "$(tr '\n' ' ' < $A100_DIR/token_len.txt) -> MAX_LEN=$MAX_LEN"
fi
[ -z "$MAX_STEPS" ] && [ -f $OUT/adapter_model.safetensors ] && { log sft INFO "skip (adapter exists: $OUT)"; exit 0; }
[ -n "$MAX_STEPS" ] && STEPS="--max_steps $MAX_STEPS"
rm -f $A100_DIR/.vram
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> $A100_DIR/.vram; sleep 15; done ) & VP=$!
( sleep 60; while true; do python3 scripts/a100/sft_eta.py $OUT/progress/metrics.jsonl $((${MAX_STEPS:-64})) | while read l; do log sft-train INFO "$l"; done; sleep 300; done ) & EP=$!
stage sft-${MAX_STEPS:+smoke-}$(basename $OUT) py re-unsloth train/sft_unsloth.py --model $BASE --load_in_4bit --train $TRAIN --out $OUT \
  --max_len $MAX_LEN --r 32 --epochs 2 --lr 1e-4 --bs 1 --ga 16 --enable_thinking --val_frac 0.08 --log_steps 2 \
  --disable_tqdm --save_state $STEPS
rc=$?; kill $VP $EP 2>/dev/null
peak=$(sort -n $A100_DIR/.vram 2>/dev/null | tail -1)
log sft-$(basename $OUT) INFO "peak_vram_MiB=${peak:-?} progress=$OUT/progress/metrics.jsonl"
exit $rc
