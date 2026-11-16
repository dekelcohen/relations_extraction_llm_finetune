#!/bin/bash
# eval/predict.py on pilot / grpo_dev / test_long with the thinking settings of the config. Skips finished sets (metrics file exists) and resumes unfinished ones per prompt.
#   eval_model.sh --machine M --run R [--name N] [--model X] [--quant bitsandbytes] [--lora dir] [--limit 3] [--eval-sets "pilot"] [--outdir outputs/eval]
# outputs: pilot -> $NAME-think.preds.jsonl ; others -> $NAME.<set>.preds.jsonl (+ .metrics.json)
source "$(dirname "$0")/lib.sh"; init "$@"
NAME=${NAME:-$EVAL_NAME}; OUTDIR=${OUTDIR:-outputs/eval}; mkdir -p $OUTDIR
# pipeline evals (<EVAL_NAME|FT_NAME>-<base>): skip bases removed from EVAL_BASES since the pipeline started (config is re-read here)
for p in $EVAL_NAME $FT_NAME; do b=${NAME#$p-}
  [ "$b" != "$NAME" ] && [[ $b != *-* ]] && [[ " $EVAL_BASES " != *" $b "* ]] && { log eval-$NAME INFO "skip (base $b not in EVAL_BASES)"; exit 0; }
done
declare -A DATA=([pilot]=data/pilot/export/pilot_all.jsonl [grpo_dev]=data/stage_2_RL_GRPO/grpo_dev.jsonl [test_long]=data/test_long/test_long.jsonl [hard_inss]=data/hard_inss/dev.jsonl)
# one process / one engine load for all unfinished sets
D=(); O=()
for n in ${NAMES:-$NAME}; do for s in $EVAL_SETS; do
  if [ $s = pilot ]; then out=$OUTDIR/$n-think.preds.jsonl; else out=$OUTDIR/$n.$s.preds.jsonl; fi
  if [ -n "$RETRY" ]; then   # --retry TAG [--names "a b"]: hit-limit retry of finished preds -> <stem>.retry-TAG.jsonl (eval/retry_hits.py)
    [ -f $out.metrics.json ] && D+=(--data ${DATA[$s]}) O+=(--out $out); continue   # retry_hits.py skips chunks already retried
  fi
  [ -n "$LIMIT" ] && out=$OUTDIR/smoke.$NAME.$s.jsonl
  [ -z "$LIMIT" ] && [ -f $out.metrics.json ] && { log eval-$NAME INFO "skip $s (done)"; continue; }
  D+=(--data ${DATA[$s]}); O+=(--out $out)
done; done
[ ${#D[@]} -eq 0 ] && exit 0
F="--temperature $TEMPERATURE --top_p $TOP_P --top_k $TOP_K --max_tokens $MAX_TOKENS --max_model_len $MAX_MODEL_LEN --gpu_mem $GPU_MEM"
[ "$ENABLE_THINKING" = 1 ] && F="$F --enable_thinking"
[ -n "$MAX_NUM_SEQS" ] && F="$F --max_num_seqs $MAX_NUM_SEQS"
[ "$ENFORCE_EAGER" = 1 ] && F="$F --enforce_eager"
[ -n "$KV_CACHE_DTYPE" ] && F="$F --kv_cache_dtype $KV_CACHE_DTYPE"
[ -n "$QUANT" ] && F="$F --quantization $QUANT"
[ -n "$LORA" ] && F="$F --lora $LORA --max_lora_rank $MAX_LORA_RANK"
[ -n "$LIMIT" ] && F="$F --limit $LIMIT"
[ -n "$SEED" ] && F="$F --seed $SEED"
CK=(); [ -n "$CHAT_KW" ] && CK=(--chat_kw "$CHAT_KW")   # JSON: keep as one argv word (no re-splitting)
# predict.py logs engine load, progress/throughput/ETA (every $PROGRESS_EVERY s) and per-set results to pipeline.log;
# finished prompts go to <out>.partial.jsonl, so a restart resumes mid-set
SCRIPT=(eval/predict.py); [ -n "$RETRY" ] && SCRIPT=(eval/retry_hits.py --tag $RETRY)
stage ${RETRY:+retry-$RETRY-}eval-$NAME py $ENV_VLLM "${SCRIPT[@]}" --model $MODEL "${D[@]}" "${O[@]}" $F "${CK[@]}" ${ENGINE_KW:+--engine_kw "$ENGINE_KW"}
