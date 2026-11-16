#!/bin/bash
# Stages: ab_<tag> (INT4 A/B on 50 pilot prompts) retry_<seed|medium> (re-generate max_tokens hits of int4 preds, eval/retry_hits.py) setup[_vllm|_unsloth] download[_eval|_sft] smoke base_eval[_<base>] sft_smoke sft ft_eval[_<base>]; bases = EVAL_BASES in the run config. Idempotent; stops at first FAIL.
#   pipeline_stages.sh --machine a100 --run qwen3.8-27b [--stages "base_eval sft"] [--limit N] [--detach]
source "$(dirname "$0")/lib.sh"; init "$@"
S="scripts/pipeline"; A="--machine $MACHINE --run $RUN"; export NO_REEXEC=1; export IN_CONTAINER=${IN_CONTAINER:-1}
bm_() { local k=${1^^}; k=${k//-/_}; local m=BASE_${k}_MODEL q=BASE_${k}_QUANT e=BASE_${k}_ENGINE_KW g=BASE_${k}_GPU_MEM s=BASE_${k}_MAX_NUM_SEQS t=BASE_${k}_MAX_TOKENS l=BASE_${k}_MAX_MODEL_LEN c=BASE_${k}_CHAT_KW
  BM=${!m}; BQ=${!q}; BO=(); [ -n "${!c}" ] && BO+=(--chat-kw "${!c}"); [ -n "${!e}" ] && BO+=(--engine-kw "${!e}"); [ -n "${!g}" ] && BO+=(--gpu-mem "${!g}"); [ -n "${!s}" ] && BO+=(--max-num-seqs "${!s}"); [ -n "${!t}" ] && BO+=(--max-tokens "${!t}"); [ -n "${!l}" ] && BO+=(--max-model-len "${!l}"); }  # EVAL_BASES entry -> model/quant + per-base engine overrides
run_stage() {
  case $1 in
    setup)          stage setup bash $S/setup_env.sh $A ;;
    setup_vllm)     stage setup_vllm bash $S/setup_env.sh $A --which vllm ;;
    setup_unsloth)  stage setup_unsloth bash $S/setup_env.sh $A --which unsloth ;;
    download)       stage download bash $S/download.sh $A ;;
    download_eval)  stage download_eval bash $S/download.sh $A ;;
    download_sft)   stage download_sft bash $S/download.sh $A --download "$SFT_BASE" ;;
    smoke)     b=${EVAL_BASES%% *}; bm_ $b; bash $S/eval_model.sh $A --name $EVAL_NAME-$b --model $BM --quant "$BQ" "${BO[@]}" --limit ${LIMIT:-3} --eval-sets pilot ;;
    base_eval)      for b in $EVAL_BASES; do run_stage base_eval_$b || return 1; done ;;
    base_eval_*)    b=${1#base_eval_}; bm_ $b; bash $S/eval_model.sh $A --name $EVAL_NAME-$b --model $BM --quant "$BQ" "${BO[@]}" ;;
    ft_eval)        for b in $EVAL_BASES; do run_stage ft_eval_$b || return 1; done ;;
    ft_eval_*)      b=${1#ft_eval_}; bm_ $b; bash $S/eval_model.sh $A --name $FT_NAME-$b --model $BM --quant "$BQ" "${BO[@]}" --lora $SFT_OUT ;;
    ftlim_*)        b=${1#ftlim_}; bm_ $b; bash $S/eval_model.sh $A --name $FT_NAME-$b-lim --model $BM --quant "$BQ" "${BO[@]}" --lora $SFT_OUT --limit ${LIMIT:-50} --eval-sets "${LIMIT_SETS:-grpo_dev}" ;;  # ft on the first N prompts -> smoke.<FT_NAME>-<b>-lim.<set>.jsonl ("-lim" bypasses the EVAL_BASES skip)
    ab_*)           bm_ int4; bash $S/eval_model.sh $A --name smoke-int4-${1#ab_} --model $BM --quant "$BQ" "${BO[@]}" --limit ${LIMIT:-50} --eval-sets pilot ${AB_MAX_TOKENS:+--max-tokens $AB_MAX_TOKENS --max-model-len $AB_MAX_MODEL_LEN} ${AB_CHAT_KW:+--chat-kw "$AB_CHAT_KW"} ;;  # A/B speed+quality on the first 50 pilot prompts (AB_CHAT_KW / --engine-kw etc. override)
    retry_*)        t=${1#retry_}; bm_ int4; CK=(); [ $t = medium ] && CK=(--chat-kw '{"reasoning_effort": "medium"}')   # hit-limit retry (base+ft int4 preds, one engine), seed 1
                    bash $S/eval_model.sh $A --name $FT_NAME-int4 --names "$EVAL_NAME-int4 $FT_NAME-int4" --model $BM --quant "$BQ" "${BO[@]}" --lora $SFT_OUT --retry $t --seed 1 "${CK[@]}" ;;
    sft_smoke) bash $S/sft_qlora.sh $A --max-steps ${SMOKE_STEPS:-3} --sft-out outputs/smoke-sft; r=$?; rm -rf outputs/smoke-sft; return $r ;;
    sft)       bash $S/sft_qlora.sh $A ;;
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
