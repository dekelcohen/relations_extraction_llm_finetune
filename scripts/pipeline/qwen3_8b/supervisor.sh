#!/bin/bash
# Host supervisor for the Qwen3-8B experiment (GPU0 only, container relx8b-dev). HF_TOKEN comes from the environment only (never a
# file/argument) and is forwarded by name to the steps that need it. Idempotent: finished steps leave $ST/<step>.ok and are skipped.
# DRY_RUN=1: container/GPU/config/helper checks only (no download, no token needed).
P=/mnt/dekel/relx-qwen35-9b; W=$P/qwen3-8b-run; L=$W/logs; ST=$W/state
C=relx8b-dev; GPU=GPU-e7e4f74b-c511-3094-59c2-e67a390876f6; M=a100x4-dekel8b; R1=qwen3-8b; R2=qwen3-8b-stage2
BASE=unsloth/Qwen3-8B-unsloth-bnb-4bit; PY=/data/envs/re-vllm/bin/python; S=scripts/pipeline
DEADLINE_H=${DEADLINE_H:-40}; DRY_RUN=${DRY_RUN:-0}
[ $DRY_RUN = 1 ] && { ST=$W/state-dry; L=$W/logs-dry; }
mkdir -p $L $ST
say() { echo "[$(date '+%F %T')] $*" >> $L/orch.log; }
exec 9>$W/run.lock; flock -n 9 || { echo "already running"; exit 1; }
echo $$ > $W/run.pid
[ $DRY_RUN = 1 ] || : "${HF_TOKEN:?HF_TOKEN must be exported}"
cleanup() { rc=$?; say "cleanup start rc=$rc"; kill $WD $VP 2>/dev/null
  docker stop -t 30 $C >> $L/orch.log 2>&1
  sleep 5; say "post-stop VRAM GPU0: $(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader) ; container: $(docker inspect -f '{{.State.Status}}' $C)"
  say "supervisor exit rc=$rc"; }
trap cleanup EXIT; trap 'exit 143' TERM INT
( exec 9>&-; sleep $((DEADLINE_H*3600)); say "DEADLINE ${DEADLINE_H}h reached; stopping container"; docker stop -t 30 $C >> $L/orch.log 2>&1 ) & WD=$!
( exec 9>&-; while true; do echo "$(date +%T),$(nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> $L/vram.csv; sleep 60; done ) & VP=$!

D()  { docker exec -e IN_CONTAINER=1 -w /data/repo $C "$@"; }
DT() { docker exec -e IN_CONTAINER=1 -e HF_TOKEN -w /data/repo $C "$@"; }   # HF_TOKEN by name only
PS() { D bash $S/pipeline_stages.sh --machine $M --run $1 --stages "$2"; }
EV() { local t=$1 r=$2; shift 2; D timeout $t bash $S/eval_model.sh --machine $M --run $r "$@"; }
step() { local n=$1; shift; [ -f $ST/$n.ok ] && { say "STEP $n skipped (done)"; return 0; }
  say "STEP $n START"; local t=$SECONDS; "$@" >> $L/$n.log 2>&1; local rc=$?; say "STEP $n rc=$rc elapsed=$((SECONDS-t))s"
  [ $rc = 0 ] && touch $ST/$n.ok; return $rc; }
req() { step "$@" || { say "FATAL: required step $1 failed; stopping"; exit 1; }; }
opt() { step "$@" || say "OPTIONAL step $1 failed (nonfatal; retried at the end)"; return 0; }
gate() { D $PY $S/smoke_gate.py "$@"; }   # prints SMOKE_STATS to the step log

smoke_base() {  # crash = fatal; overthinking (low closure / invalid JSON) gets one bounded longer attempt, then is only logged
  EV 1800 $R1 --name qwen3-8b-bnb4 --model $BASE --quant bitsandbytes --limit 4 --eval-sets pilot --hf-batch-size 4 || return 1
  gate outputs/eval/smoke.qwen3-8b-bnb4.pilot.jsonl.metrics.json --min-json 0.25 --min-closed 0.5 && return 0
  echo "base thinking smoke missed the gate at max_tokens 8192; one bounded attempt at 12288"
  EV 2400 $R1 --name qwen3-8b-bnb4-long --model $BASE --quant bitsandbytes --limit 4 --eval-sets pilot --hf-batch-size 4 \
    --max-tokens 12288 --max-model-len 20480 || return 1
  gate outputs/eval/smoke.qwen3-8b-bnb4-long.pilot.jsonl.metrics.json --min-json 0.25 --min-closed 0.5 \
    || echo "WARN: base model overthinks / misses JSON even at 12288 tokens (logged, not blocking)"
  return 0
}
adapter_smoke() {  # <name> <adapter dir> <run> gate-args...: loader errors (zero LoRA B) raise inside predict.py -> fatal
  local name=$1 lora=$2 run=$3; shift 3
  EV 2400 $run --name $name --model $BASE --quant bitsandbytes --lora $lora --limit 6 --eval-sets pilot --hf-batch-size 6 || return 1
  gate outputs/eval/smoke.$name.pilot.jsonl.metrics.json "$@"
}
upload() { DT $PY scripts/hf_sync.py push $1 --repo $2 --final-model; }
collect() {
  mkdir -p $W/results
  D bash -c 'tar czf outputs/qwen3-8b-results.tgz --exclude="*.partial.jsonl" --exclude="smoke.*" outputs/eval/qwen3-8b* outputs/pipeline/a100x4-dekel8b hf_artifacts.json 2>/dev/null; for f in outputs/eval/qwen3-8b*.metrics.json; do echo $f; '$PY' -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({\"all\": d[\"report\"][\"all\"], \"gen\": d.get(\"generation\")}))" $f; done' > $L/final_metrics.txt 2>&1
  cp $W/repo/outputs/qwen3-8b-results.tgz $W/results/ 2>/dev/null; return 0
}

say "supervisor started pid=$$ deadline=${DEADLINE_H}h dry=$DRY_RUN"
bash $W/repo/$S/qwen3_8b/container.sh ensure >> $L/container.log 2>&1 || { say "FATAL: container ensure failed"; exit 1; }
say "container $C ready on $GPU"

if [ $DRY_RUN = 1 ]; then
  req dry_config D bash $S/exec.sh --machine $M --run $R1 "echo config-ok"
  req dry_py D $PY -m py_compile $S/smoke_gate.py $S/hf_auth_check.py scripts/hf_sync.py eval/predict.py train/sft_unsloth.py
  req dry_data D bash -c 'wc -l data/sft_stage2/train_selfdistill.jsonl data/synth_hefa/sft_cot_train.jsonl data/pilot/export/pilot_all.jsonl data/test_long/test_long.jsonl data/hard_inss/dev.jsonl'
  req dry_imports D bash -c "CC=/data/envs/re-vllm/bin/gcc $PY -c 'import unsloth, peft, bitsandbytes, trl, transformers; print(\"imports ok\", transformers.__version__)'"
  say "DRY RUN OK"; exit 0
fi

req auth DT $PY $S/hf_auth_check.py
req download PS $R1 download_sft
req smoke_base_think smoke_base
req smoke_train_s1 PS $R1 sft_smoke
say "GATE 1 PASSED: base thinking smoke + stage-1 training smoke"
req train_s1 PS $R1 sft
opt upload_s1 upload outputs/sft-cot-qwen3-8b Shani123/re-qwen3-8b-sft-cot
req smoke_adapter_s1 adapter_smoke qwen3-8b-ft-bnb4-s1 outputs/sft-cot-qwen3-8b $R1 --min-json 0.5 --min-closed 0.5
say "GATE 2 PASSED: stage-1 adapter smoke"
req smoke_train_s2 PS $R2 sft_smoke
req train_s2 PS $R2 sft
opt upload_s2 upload outputs/sft-cot-qwen3-8b-stage2 Shani123/re-qwen3-8b-sft-cot-stage2
req smoke_adapter_s2 adapter_smoke qwen3-8b-ft-stage2-bnb4-s2 outputs/sft-cot-qwen3-8b-stage2 $R2 --min-json 0.5 --min-closed 0.5 --min-f1 0.4
say "GATE 3 PASSED: stage-2 adapter smoke"
req eval_s2 PS $R2 ft_eval
req eval_s1 PS $R1 ft_eval
req eval_base PS $R1 base_eval
opt upload_s1 upload outputs/sft-cot-qwen3-8b Shani123/re-qwen3-8b-sft-cot
opt upload_s2 upload outputs/sft-cot-qwen3-8b-stage2 Shani123/re-qwen3-8b-sft-cot-stage2
collect
say "ALL DONE: full sequence complete (results: $W/results/qwen3-8b-results.tgz)"
