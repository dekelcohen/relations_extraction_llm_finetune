#!/bin/bash
# Launch the Qwen3-8B supervisor detached (setsid nohup). The token is read from the exported HF_TOKEN, or with --stdin from the first
# stdin line; it is kept in the process environment only (never written to a file, argument, or log).
#   export HF_TOKEN=...; bash start.sh        |        printf '%s\n' "$TOKEN" | bash start.sh --stdin
W=/mnt/dekel/relx-qwen35-9b/qwen3-8b-run
[ "$1" = --stdin ] && IFS= read -r HF_TOKEN
: "${HF_TOKEN:?HF_TOKEN must be exported (or use --stdin)}"; export HF_TOKEN
flock -n $W/run.lock true || { echo "supervisor already running (pid $(cat $W/run.pid 2>/dev/null))"; exit 1; }
cp $W/repo/scripts/pipeline/qwen3_8b/supervisor.sh $W/supervisor.sh
mkdir -p $W/logs
setsid nohup bash $W/supervisor.sh >> $W/logs/supervisor.out 2>&1 < /dev/null &
sleep 8
pid=$(cat $W/run.pid 2>/dev/null); kill -0 "$pid" 2>/dev/null && echo "supervisor running pid=$pid log=$W/logs/orch.log" || { echo "supervisor failed to start"; tail -n 5 $W/logs/supervisor.out; exit 1; }
