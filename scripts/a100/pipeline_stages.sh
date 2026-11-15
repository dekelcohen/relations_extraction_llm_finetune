#!/bin/bash
# Runs inside the container: setup -> download -> smoke -> base eval -> SFT smoke -> SFT -> ft eval. Idempotent; stops at first FAIL.
#   env: FP8=Qwen/Qwen3.8-27B-FP8  BNB=unsloth/Qwen3.8-27B-unsloth-bnb-4bit  STAGES="setup download smoke eval sftsmoke sft fteval"
cd ${REPO:-/NLP/relations_extraction_llm_finetune}
source scripts/a100/lib.sh
FP8=${FP8:-Qwen/Qwen3.8-27B-FP8}; BNB=${BNB:-unsloth/Qwen3.8-27B-unsloth-bnb-4bit}
STAGES=${STAGES:-"setup download smoke eval sftsmoke sft fteval"}
export A100_DIR
for s in $STAGES; do
  case $s in
    setup)    stage setup bash scripts/a100/setup_env.sh all || exit 1 ;;
    download) stage download bash scripts/a100/download.sh $FP8 $BNB || exit 1 ;;
    smoke)    NAME=qwen3.8-27b MODEL=$FP8 LIMIT=3 SETS=pilot bash scripts/a100/eval_model.sh || exit 1 ;;
    eval)     NAME=qwen3.8-27b MODEL=$FP8 bash scripts/a100/eval_model.sh || exit 1 ;;
    sftsmoke) MAX_STEPS=3 OUT=outputs/smoke-sft bash scripts/a100/sft_qlora.sh || exit 1; rm -rf outputs/smoke-sft ;;
    sft)      bash scripts/a100/sft_qlora.sh || exit 1 ;;
    fteval)   # LoRA trained on the bnb-4bit base: try the FP8 base first, fall back to the bnb-4bit base
              L=outputs/sft-cot-qwen3.8-27b
              NAME=qwen3.8-27b-ft MODEL=$FP8 LORA=$L bash scripts/a100/eval_model.sh \
                || NAME=qwen3.8-27b-ft MODEL=$BNB QUANT=bitsandbytes LORA=$L bash scripts/a100/eval_model.sh || exit 1 ;;
  esac
done
log pipeline DONE "stages: $STAGES"
