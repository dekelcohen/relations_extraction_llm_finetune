#!/bin/bash
# GPU / pipeline watchdog. Runs ON THE HOST (not in the container) as its own process, so it survives pipeline/container crashes.
#   bash scripts/pipeline/gpu_watchdog.sh --machine a100 [--idle-min 10] [--poll-s 30] [--util-max 2] [--gpu 0] [--deadline-h 5]
# Tracks only OUR usage: GPU processes whose host pid belongs to $CONTAINER (docker top) — memory from
# nvidia-smi --query-compute-apps, SM util from nvidia-smi pmon. Foreign GPU load neither masks our idleness nor alarms.
# Alerts (pipeline.log 'STAGE watchdog ALERT <kind> ...' + machine-readable alerts.log '<ts>\t<kind>\t<detail>'):
#   gpu_idle      our util <= UTIL_MAX and our memory not growing for IDLE_MIN (setup*/download* stages exempt); once per episode
#   stage_fail    a 'STAGE <x> FAIL' line appeared (alerts.log only; the FAIL line itself is already in pipeline.log)
#   pipeline_dead a stage is still START but no pipeline process runs in the container (2 polls)
# --deadline-h: hard wall-clock budget; logs 'STAGE watchdog DEADLINE' and docker-stops OUR container. Stop: touch <dir>/watchdog.stop
cd "$(dirname "$0")/../.."
M=; DEADLINE_H=; IDLE_MIN=10; POLL_S=30; UTIL_MAX=2; GPU=0
while [ $# -gt 0 ]; do case $1 in
  --machine) M=$2 ;; --idle-min) IDLE_MIN=$2 ;; --poll-s) POLL_S=$2 ;; --util-max) UTIL_MAX=$2 ;; --gpu) GPU=$2 ;; --deadline-h) DEADLINE_H=$2 ;;
  *) sed -n 2,11p "$0"; exit 1 ;; esac; shift 2; done
: "${M:?--machine required}"
. configs/machines/$M.env; DK=${DOCKER:-docker}
IDLE_S=$(awk "BEGIN{print int($IDLE_MIN*60)}"); END_TS=; [ -n "$DEADLINE_H" ] && END_TS=$(( $(date +%s) + $(awk "BEGIN{print int($DEADLINE_H*3600)}") ))
D=outputs/pipeline/$M; LOG=$D/pipeline.log; ALOG=$D/alerts.log; PID=$D/watchdog.pid; mkdir -p "$D"; rm -f "$D/watchdog.stop"
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "watchdog already running pid=$(cat "$PID")"; exit 0; fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE watchdog $*" >> "$LOG"; }
alert_file() { printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" >> "$ALOG"; }
alert() { note ALERT "$1 $2"; alert_file "$1" "$2"; }
evacuation_guard() { python3 scripts/pipeline/evacuation_guard.py "$D" "$1" --docker "$DK" --container "$CONTAINER"; }
deadline_stop() {
  if [ "$M" = a100 ]; then evacuation_guard stop; else $DK stop "$CONTAINER" >/dev/null 2>&1; fi
}
declare -A SEEN
note INFO "started pid=$$ gpu=$GPU container=$CONTAINER idle_min=$IDLE_MIN util_max=$UTIL_MAX deadline=${END_TS:+$(date -d @$END_TS '+%F %T')}"
nl=$(wc -l < "$LOG" 2>/dev/null || echo 0); idle_since=; alerted=0; prev_mem=0; noproc=0
while [ ! -f "$D/watchdog.stop" ]; do
  now=$(date +%s)
  [ "$M" != a100 ] || evacuation_guard observe
  if [ -n "$END_TS" ] && [ "$now" -ge "$END_TS" ]; then
    alert_file deadline "${DEADLINE_H}h reached: docker stop $CONTAINER"
    note DEADLINE "${DEADLINE_H}h reached: docker stop $CONTAINER"
    if deadline_stop; then
      note INFO "container $CONTAINER stopped or already down"
    else
      alert deadline_stop_failed "$CONTAINER"
    fi
    break
  fi
  # new FAIL lines since the last poll
  n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  [ "$n" -gt "$nl" ] && tail -n +$((nl + 1)) "$LOG" | head -n $((n - nl)) | grep -E 'STAGE [^ ]+ FAIL' | while IFS= read -r l; do alert_file stage_fail "${l:0:400}"; done
  nl=$n
  # our container's processes (host pids) and whether a pipeline step is running in it
  top=$($DK top "$CONTAINER" -eo pid,args 2>/dev/null | tail -n +2)
  pids=" $(echo "$top" | awk '{print $1}' | tr '\n' ' ') "
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  q=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -z "$q" ]; then alert nvidia_smi_failed "-"; sleep $((POLL_S * 10)); continue; fi
  gutil=${q%,*}; gmem=${q#*,}
  mem=0; mine=
  for a in $apps; do p=${a%,*}; [[ $pids == *" $p "* ]] && { mem=$((mem + ${a#*,})); mine="$mine $p"; }; done
  util=0
  if [ -n "$mine" ]; then   # per-process SM% for our pids; pmon unavailable -> global util
    pm=$(nvidia-smi pmon -i "$GPU" -c 1 -s u 2>/dev/null | grep -v '^#')
    if [ -n "$pm" ]; then util=$(echo "$pm" | awk -v m="$mine " '{if (index(" " m, " " $2 " ") && $4 ~ /^[0-9]+$/) s += $4} END{print s+0}')
    else util=$gutil; fi
  fi
  # pipeline_dead: an open (START without DONE/FAIL) stage while no pipeline process runs in the container, 2 polls in a row
  open=$(awk '$3=="STAGE" && $4=="config" {delete s; delete t}  # only stages of the latest run (lib.sh logs config once per run)
    $3=="STAGE" && $4!="watchdog" && ($5=="START"||$5=="DONE"||$5=="FAIL") {s[$4]=$5; t[$4]=$1"_"$2} END{for (k in s) if (s[k]=="START") printf "%s@%s ", k, t[k]}' "$LOG" 2>/dev/null)
  if [ -n "$open" ] && ! echo "$top" | grep -qE 'pipeline_stages\.sh|eval_model\.sh|sft_qlora\.sh|setup_env\.sh|download\.sh|predict\.py|sft_unsloth\.py'; then
    noproc=$((noproc + 1)); [ $noproc -ge 2 ] && [ -z "${SEEN[$open]}" ] && { SEEN[$open]=1; alert pipeline_dead "open=${open% } container_procs=$(echo "$top" | grep -c .)"; }
  else noproc=0; fi
  # gpu_idle (ours only). Non-GPU stages (setup*/download*) are exempt.
  cur=$(grep -oE '"stage":"[^"]*","state":"START"' "$D/status.json" 2>/dev/null | cut -d'"' -f4)
  if [[ $cur =~ ^(setup|download) ]] || { [ "$util" -gt "$UTIL_MAX" ] || [ $((mem - prev_mem)) -ge 500 ]; }; then
    [ $alerted = 1 ] && note INFO "gpu_busy again util=$util mem=${mem}MiB"
    idle_since=; alerted=0
  else
    idle_since=${idle_since:-$now}
    if [ $alerted = 0 ] && [ $((now - idle_since)) -ge $IDLE_S ]; then
      last=$(grep -v -e 'STAGE config' -e 'STAGE watchdog' "$LOG" 2>/dev/null | tail -n 1 | cut -c1-250)
      alert gpu_idle "$((now - idle_since))s ours: util=$util mem=${mem}MiB pids=${mine:- none}; gpu total: util=$gutil mem=${gmem}MiB; last=$last"; alerted=1
    fi
  fi
  prev_mem=$mem; sleep "$POLL_S"
done
note INFO "stopped"
