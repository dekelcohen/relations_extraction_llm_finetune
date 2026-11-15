#!/bin/bash
# Stages: setup[_vllm|_unsloth] download[_eval|_sft] smoke base_eval sft_smoke sft ft_eval. Idempotent; stops at first FAIL.
#   pipeline_stages.sh --machine a100 --run qwen3.8-27b [--stages "base_eval sft"] [--limit N] [--detach]
source "$(dirname "$0")/lib.sh"; init "$@"
S="scripts/pipeline"; A="--machine $MACHINE --run $RUN"; export NO_REEXEC=1; export IN_CONTAINER=${IN_CONTAINER:-1}
run_stage() {
  case $1 in
    setup)          stage setup bash $S/setup_env.sh $A ;;
    setup_vllm)     stage setup_vllm bash $S/setup_env.sh $A --which vllm ;;
    setup_unsloth)  stage setup_unsloth bash $S/setup_env.sh $A --which unsloth ;;
    download)       stage download bash $S/download.sh $A ;;
    download_eval)  stage download_eval bash $S/download.sh $A --download "$MODEL" ;;
    download_sft)   stage download_sft bash $S/download.sh $A --download "$SFT_BASE" ;;
    smoke)     bash $S/eval_model.sh $A --limit 3 --eval-sets pilot ;;
    base_eval) bash $S/eval_model.sh $A ;;
    sft_smoke) bash $S/sft_qlora.sh $A --max-steps 3 --sft-out outputs/smoke-sft; r=$?; rm -rf outputs/smoke-sft; return $r ;;
    sft)       bash $S/sft_qlora.sh $A ;;
    ft_eval)   bash $S/eval_model.sh $A --name $FT_NAME --model $FT_MODEL --quant "$FT_QUANT" --lora $SFT_OUT \
                 || { [ -n "$FT_FALLBACK_MODEL" ] && bash $S/eval_model.sh $A --name $FT_NAME --model $FT_FALLBACK_MODEL --quant "$FT_FALLBACK_QUANT" --lora $SFT_OUT; } ;;
    *) echo "unknown stage $1"; return 2 ;;
  esac
}
run_track() { for s in $1; do run_stage $s || return 1; done; }
# STAGES: groups separated by ';' run in order; tracks inside a group separated by '|' run in parallel (e.g. GPU eval || CPU/network setup)
IFS=';' read -ra GROUPS_ <<< "$STAGES"
for g in "${GROUPS_[@]}"; do
  IFS='|' read -ra TR <<< "$g"; pids=(); fail=0
  [ ${#TR[@]} -gt 1 ] && log parallel INFO "tracks: $g"
  for t in "${TR[@]}"; do run_track "$t" & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || fail=1; done
  [ $fail = 1 ] && { log pipeline FAIL "group: $g"; exit 1; }
done
log pipeline DONE "stages: $STAGES"
