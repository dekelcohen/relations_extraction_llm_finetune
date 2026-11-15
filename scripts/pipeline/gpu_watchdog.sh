#!/bin/bash
# GPU idle watchdog. Runs ON THE HOST (not in the container) as its own process, so it survives pipeline/container crashes.
#   bash scripts/pipeline/gpu_watchdog.sh --machine a100 [--idle-min 10] [--poll-s 30] [--util-max 2] [--gpu 0] [--deadline-h 5]
# --deadline-h: hard wall-clock budget; when reached logs 'STAGE watchdog DEADLINE' and docker-stops OUR container (frees GPU).
# Idle = util <= UTIL_MAX and memory.used not growing (model load from NFS shows util 0 but rising memory).
# After IDLE_MIN idle minutes, appends to pipeline.log (agents' wait loops grep it):
#   [ts] STAGE watchdog ALERT gpu_idle <N>s util=.. mem=.. last=<last stage line>
# One alert per idle episode, re-armed when the GPU becomes busy. Stop: touch <pipeline dir>/watchdog.stop
cd "$(dirname "$0")/../.."
M=; DEADLINE_H=; IDLE_MIN=10; POLL_S=30; UTIL_MAX=2; GPU=0
while [ $# -gt 0 ]; do case $1 in
  --machine) M=$2 ;; --idle-min) IDLE_MIN=$2 ;; --poll-s) POLL_S=$2 ;; --util-max) UTIL_MAX=$2 ;; --gpu) GPU=$2 ;; --deadline-h) DEADLINE_H=$2 ;;
  *) sed -n 2,7p "$0"; exit 1 ;; esac; shift 2; done
: "${M:?--machine required}"
. configs/machines/$M.env
IDLE_S=$(awk "BEGIN{print int($IDLE_MIN*60)}"); END_TS=; [ -n "$DEADLINE_H" ] && END_TS=$(( $(date +%s) + $(awk "BEGIN{print int($DEADLINE_H*3600)}") ))
D=outputs/pipeline/$M; LOG=$D/pipeline.log; PID=$D/watchdog.pid; mkdir -p "$D"; rm -f "$D/watchdog.stop"
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "watchdog already running pid=$(cat "$PID")"; exit 0; fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE watchdog $*" >> "$LOG"; }
note INFO "started pid=$$ gpu=$GPU idle_min=$IDLE_MIN util_max=$UTIL_MAX deadline=${END_TS:+$(date -d @$END_TS '+%F %T')}"
idle_since=; alerted=0; prev_mem=
while [ ! -f "$D/watchdog.stop" ]; do
  q=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -z "$q" ]; then note ALERT "nvidia-smi failed"; sleep $((POLL_S * 10)); continue; fi
  util=${q%,*}; mem=${q#*,}; now=$(date +%s)
  if [ -n "$END_TS" ] && [ "$now" -ge "$END_TS" ]; then
    note DEADLINE "${DEADLINE_H}h reached: docker stop $CONTAINER"; ${DOCKER:-docker} stop "$CONTAINER" >/dev/null 2>&1; note INFO "stopped container $CONTAINER"; break
  fi
  if [ "$util" -le "$UTIL_MAX" ] && [ $((mem - ${prev_mem:-$mem})) -lt 500 ]; then
    idle_since=${idle_since:-$now}
    if [ $alerted = 0 ] && [ $((now - idle_since)) -ge $IDLE_S ]; then
      last=$(grep -v -e 'config INFO' -e 'STAGE watchdog' "$LOG" 2>/dev/null | tail -n 1 | cut -c1-300)
      note ALERT "gpu_idle $((now - idle_since))s util=$util mem=${mem}MiB last=$last"; alerted=1
    fi
  else
    [ $alerted = 1 ] && note INFO "gpu_busy again util=$util mem=${mem}MiB"
    idle_since=; alerted=0
  fi
  prev_mem=$mem; sleep "$POLL_S"
done
note INFO "stopped"
