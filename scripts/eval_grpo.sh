#!/bin/bash
# Stage-2 evals (inside the container): grpo_dev for SFT + 2 GRPO adapters -> pick best by dev slot-F1 -> pilot + test_long.
# Same thinking settings as qwen3-4b-ft-think.
cd /NLP/relations_extraction_llm_finetune
E=outputs/eval
EV="--enable_thinking --temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 5120 --max_model_len 8192 --gpu_mem 0.9 --enforce_eager --model Qwen/Qwen3-4B-AWQ --max_lora_rank 32"
run() { # name lora data
  [ -f $E/$1.preds.jsonl ] || python eval/predict.py $EV --lora $2 --data $3 --out $E/$1.preds.jsonl > $E/$1.log 2>&1; }
DEV=data/stage_2_RL_GRPO/grpo_dev.jsonl
run sft.grpo_dev outputs/sft-cot-qwen3-4b $DEV
run grpo-final.grpo_dev outputs/grpo-qwen3-4b $DEV
run grpo-mid.grpo_dev outputs/grpo-qwen3-4b/checkpoint-10 $DEV
BEST=$(python - <<'P'
import json
c={"outputs/grpo-qwen3-4b":"grpo-final","outputs/grpo-qwen3-4b/checkpoint-10":"grpo-mid"}
f=lambda n: json.load(open(f"outputs/eval/{n}.grpo_dev.preds.jsonl.metrics.json"))["report"]["all"]["slot_f1"]
print(max(c, key=lambda k: f(c[k])))
P
)
echo $BEST > $E/grpo_best.txt
run qwen3-4b-grpo-think $BEST data/pilot/export/pilot_all.jsonl
run sft.test_long outputs/sft-cot-qwen3-4b data/test_long/test_long.jsonl
run grpo.test_long $BEST data/test_long/test_long.jsonl
touch $E/GRPO_EVAL_DONE
