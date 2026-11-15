#!/bin/bash
# Run eval/predict.py (thinking settings of the Qwen3-4B base run) on pilot, grpo_dev, test_long. Skips finished sets.
#   NAME=qwen3.8-27b MODEL=Qwen/Qwen3.8-27B-FP8 [LORA=dir] [QUANT=bitsandbytes] [LIMIT=3] [SETS="pilot grpo_dev test_long"]
#   [OUTDIR=outputs/eval] [MAX_TOKENS=5120 MAX_LEN=8192 GPU_MEM=0.92 MAX_SEQS=16 EXTRA="..."]
# outputs: pilot -> $NAME-think.preds.jsonl ; others -> $NAME.<set>.preds.jsonl
cd ${REPO:-/NLP/relations_extraction_llm_finetune}
source scripts/a100/lib.sh
: "${NAME:?}" "${MODEL:?}"
export HF_HOME=${HF_HOME:-/data} HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING
OUTDIR=${OUTDIR:-outputs/eval}; SETS=${SETS:-pilot grpo_dev test_long}
declare -A DATA=([pilot]=data/pilot/export/pilot_all.jsonl [grpo_dev]=data/stage_2_RL_GRPO/grpo_dev.jsonl [test_long]=data/test_long/test_long.jsonl)
mkdir -p $OUTDIR
for s in $SETS; do
  if [ $s = pilot ]; then out=$OUTDIR/$NAME-think.preds.jsonl; else out=$OUTDIR/$NAME.$s.preds.jsonl; fi
  [ -f $out.metrics.json ] && [ -z "$LIMIT" ] && { log eval-$NAME-$s INFO "skip (done)"; continue; }
  FLAGS="--enable_thinking --temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens ${MAX_TOKENS:-5120} --max_model_len ${MAX_LEN:-8192} --gpu_mem ${GPU_MEM:-0.92} --max_num_seqs ${MAX_SEQS:-16} --enforce_eager"
  [ -n "$QUANT" ] && FLAGS="$FLAGS --quantization $QUANT"
  [ -n "$LORA" ] && FLAGS="$FLAGS --lora $LORA --max_lora_rank 32"
  [ -n "$LIMIT" ] && FLAGS="$FLAGS --limit $LIMIT" && out=$OUTDIR/smoke.$NAME.$s.jsonl
  t0=$SECONDS
  stage eval-$NAME-$s py re-vllm eval/predict.py --model $MODEL --data ${DATA[$s]} --out $out $FLAGS $EXTRA || exit 1
  f1=$(python3 - "$out.metrics.json" <<'P'
import json,sys
m=json.load(open(sys.argv[1])); r=m["report"]["all"]; print("slot_f1=%s n=%s" % (r.get("slot_f1"), r.get("n")))
P
)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
  log eval-$NAME-$s INFO "$f1 wall=$((SECONDS-t0))s out=$out"
done
