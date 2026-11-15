#!/bin/bash
# Local helper for the A100 host. Password comes from the environment only:  export SSHPASS=...
#   remote.sh status | logs [stage-log-name] | start ["stages"] | pull-results | sh "<cmd>"
H=${A100_HOST:-user@10.53.160.181}
NLP=${A100_NLP:-/mnt/tv_developers/projects/vla/NLP}
R=$NLP/relations_extraction_llm_finetune
: "${SSHPASS:?export SSHPASS first}"
rs() { sshpass -e ssh -o StrictHostKeyChecking=no $H "$@"; }
case $1 in
  status) rs "tail -n ${N:-15} $R/outputs/a100/pipeline.log; cat $R/outputs/a100/status.json; echo; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" ;;
  logs)   rs "tail -n ${N:-40} $R/outputs/a100/${2:-pipeline}.log" ;;
  start)  rs "bash $R/scripts/a100/run_pipeline.sh '$2'" ;;
  pull-results)
    cd "$(dirname "$0")/../.." && sshpass -e rsync -az -e "ssh -o StrictHostKeyChecking=no" \
      --include='*/' --include='qwen3.8-27b*' --include='smoke.*' --exclude='*' $H:$R/outputs/eval/ outputs/eval/ &&
    sshpass -e rsync -az -e "ssh -o StrictHostKeyChecking=no" --exclude='.vram' --exclude='*.out' $H:$R/outputs/a100/ outputs/a100/
    # adapter only (no optimizer state): outputs/sft-cot-qwen3.8-27b -> then scripts/hf_sync.py push
    sshpass -e rsync -az -e "ssh -o StrictHostKeyChecking=no" --exclude='checkpoint-*' $H:$R/outputs/sft-cot-qwen3.8-27b/ outputs/sft-cot-qwen3.8-27b/ 2>/dev/null || true ;;
  sh) rs "${@:2}" ;;
  *) sed -n 2,4p "$0" ;;
esac
