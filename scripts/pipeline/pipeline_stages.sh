#!/bin/bash
# Stages (config var STAGES / --stages "a b"): setup download smoke base_eval sft_smoke sft ft_eval. Idempotent; stops at first FAIL.
#   pipeline_stages.sh --machine a100 --run qwen3.8-27b [--stages "base_eval sft"] [--limit N] [--detach]
source "$(dirname "$0")/lib.sh"; init "$@"
S="scripts/pipeline"; A="--machine $MACHINE --run $RUN"; export NO_REEXEC=1; export IN_CONTAINER=${IN_CONTAINER:-1}
for s in $STAGES; do
  case $s in
    setup)     stage setup bash $S/setup_env.sh $A || exit 1 ;;
    download)  stage download bash $S/download.sh $A || exit 1 ;;
    smoke)     bash $S/eval_model.sh $A --limit 3 --eval-sets pilot || exit 1 ;;
    base_eval) bash $S/eval_model.sh $A || exit 1 ;;
    sft_smoke) bash $S/sft_qlora.sh $A --max-steps 3 --sft-out outputs/smoke-sft || exit 1; rm -rf outputs/smoke-sft ;;
    sft)       bash $S/sft_qlora.sh $A || exit 1 ;;
    ft_eval)   bash $S/eval_model.sh $A --name $FT_NAME --model $FT_MODEL --quant "$FT_QUANT" --lora $SFT_OUT \
                 || { [ -n "$FT_FALLBACK_MODEL" ] && bash $S/eval_model.sh $A --name $FT_NAME --model $FT_FALLBACK_MODEL --quant "$FT_FALLBACK_QUANT" --lora $SFT_OUT; } || exit 1 ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
log pipeline DONE "stages: $STAGES"
