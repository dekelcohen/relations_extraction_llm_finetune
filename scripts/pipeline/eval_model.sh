#!/bin/bash
# eval/predict.py on pilot / grpo_dev / test_long with the thinking settings of the config. Skips finished sets (metrics file exists).
#   eval_model.sh --machine M --run R [--name N] [--model X] [--quant bitsandbytes] [--lora dir] [--limit 3] [--eval-sets "pilot"] [--outdir outputs/eval]
# outputs: pilot -> $NAME-think.preds.jsonl ; others -> $NAME.<set>.preds.jsonl (+ .metrics.json)
source "$(dirname "$0")/lib.sh"; init "$@"
NAME=${NAME:-$EVAL_NAME}; OUTDIR=${OUTDIR:-outputs/eval}; mkdir -p $OUTDIR
declare -A DATA=([pilot]=data/pilot/export/pilot_all.jsonl [grpo_dev]=data/stage_2_RL_GRPO/grpo_dev.jsonl [test_long]=data/test_long/test_long.jsonl)
for s in $EVAL_SETS; do
  if [ $s = pilot ]; then out=$OUTDIR/$NAME-think.preds.jsonl; else out=$OUTDIR/$NAME.$s.preds.jsonl; fi
  [ -n "$LIMIT" ] && out=$OUTDIR/smoke.$NAME.$s.jsonl
  [ -z "$LIMIT" ] && [ -f $out.metrics.json ] && { log eval-$NAME-$s INFO "skip (done)"; continue; }
  F="--temperature $TEMPERATURE --top_p $TOP_P --top_k $TOP_K --max_tokens $MAX_TOKENS --max_model_len $MAX_MODEL_LEN --gpu_mem $GPU_MEM"
  [ "$ENABLE_THINKING" = 1 ] && F="$F --enable_thinking"
  [ -n "$MAX_NUM_SEQS" ] && F="$F --max_num_seqs $MAX_NUM_SEQS"
  [ "$ENFORCE_EAGER" = 1 ] && F="$F --enforce_eager"
  [ -n "$KV_CACHE_DTYPE" ] && F="$F --kv_cache_dtype $KV_CACHE_DTYPE"
  [ -n "$QUANT" ] && F="$F --quantization $QUANT"
  [ -n "$LORA" ] && F="$F --lora $LORA --max_lora_rank $MAX_LORA_RANK"
  [ -n "$LIMIT" ] && F="$F --limit $LIMIT"
  t0=$SECONDS
  stage eval-$NAME-$s py $ENV_VLLM eval/predict.py --model $MODEL --data ${DATA[$s]} --out $out $F || exit 1
  f1=$(python3 - "$out.metrics.json" <<'P'
import json,sys
r=json.load(open(sys.argv[1]))["report"]["all"]; print("slot_f1=%s n=%s" % (r.get("slot_f1"), r.get("n")))
P
)
  log eval-$NAME-$s INFO "$f1 wall=$((SECONDS-t0))s out=$out"
done
