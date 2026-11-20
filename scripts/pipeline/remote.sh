#!/bin/bash
# Generic helper (run on your workstation) for a machine in configs/machines. Remote machines: ssh password only from the env: export SSHPASS=...
#   remote.sh --machine a100 status | logs [name] | start [pipeline args...] | pull-results | sh "<cmd>" | wait [--timeout 570] | watchdog [--idle-min N] [--deadline-h H] | watchdog-stop
# A machine with empty SSH_HOST (local3070) runs the same commands locally.
cd "$(dirname "$0")/../.."
[ "$1" = --machine ] && M=$2 && shift 2 || { sed -n 2,5p "$0"; exit 1; }
. configs/machines/$M.env; CMD=$1; shift
if [ -n "$SSH_HOST" ]; then : "${SSHPASS:?export SSHPASS first}"
  rs() { sshpass -e ssh -o StrictHostKeyChecking=no $SSH_HOST "$@"; }; RR=$HOST_REPO
  rsync_() { sshpass -e rsync -az -e "ssh -o StrictHostKeyChecking=no" "$@"; }; SRC=$SSH_HOST:
else rs() { bash -c "$*"; }; RR=$HOST_REPO; rsync_() { rsync -az "$@"; }; SRC=; fi
P=$RR/outputs/pipeline/$M
case $CMD in
  status) rs "tail -n ${N:-15} $P/pipeline.log; cat $P/status.json; echo; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" ;;
  logs)   rs "tail -n ${N:-40} $P/${1:-pipeline}.log" ;;
  start)  rs "cd $RR && git pull -q --ff-only; bash scripts/pipeline/run_pipeline.sh --machine $M $(printf "%q " "$@")" ;;
  sh)     rs "$@" ;;
  watchdog) rs "cd $RR && git pull -q --ff-only; setsid nohup bash scripts/pipeline/gpu_watchdog.sh --machine $M $(printf "%q " "$@") >> $P/watchdog.out 2>&1 < /dev/null & sleep 1; cat $P/watchdog.pid" ;;
  wait)   rs "cd $RR && bash scripts/pipeline/wait_event.sh --machine $M $(printf "%q " "$@")" ;;   # first new DONE/FAIL/ALERT line or TIMEOUT+status
  watchdog-stop) rs "touch $P/watchdog.stop" ;;
  pull-results)
    [ -z "$SSH_HOST" ] && { echo "local machine: nothing to pull"; exit 0; }
    rsync_ --include='*/' --include='qwen3*' --include='smoke.*' --exclude='*' $SRC$RR/outputs/eval/ outputs/eval/
    rsync_ --exclude='.vram' --exclude='*.out' $SRC$P/ outputs/pipeline/$M/
    for d in $(rs "ls -d $RR/outputs/sft-cot-qwen3.8* $RR/outputs/sft-cot-qwen3.5-9b* 2>/dev/null"); do   # adapter only, no optimizer state
      rsync_ --exclude='checkpoint-*' $SRC$d/ outputs/$(basename $d)/; done ;;
  *) sed -n 2,5p "$0" ;;
esac
