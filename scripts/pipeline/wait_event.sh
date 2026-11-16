#!/bin/bash
# Run ON THE HOST (remote.sh wait). Blocks until NEW pipeline.log lines (vs. the count at start) match
# 'STAGE <x> DONE|FAIL' or 'STAGE watchdog ALERT|DEADLINE', prints them (max 5), or after --timeout prints
# 'TIMEOUT' + a 1-line status (status.json stage, GPU util/mem, last eval progress). Tiny output for agent wait loops.
#   wait_event.sh --machine a100 [--timeout 570] [--poll 10]
cd "$(dirname "$0")/../.."
M=; T=570; P=10
while [ $# -gt 0 ]; do case $1 in --machine) M=$2 ;; --timeout) T=$2 ;; --poll) P=$2 ;; *) sed -n 2,5p "$0"; exit 1 ;; esac; shift 2; done
D=outputs/pipeline/$M; L=$D/pipeline.log; n0=$(wc -l < "$L" 2>/dev/null || echo 0); end=$((SECONDS + T))
RE='STAGE [^ ]+ (DONE|FAIL)|STAGE watchdog (ALERT|DEADLINE)'
while [ $SECONDS -lt $end ]; do
  hit=$(tail -n +$((n0 + 1)) "$L" 2>/dev/null | grep -E "$RE" | head -n 5 | cut -c1-400)
  [ -n "$hit" ] && { echo "$hit"; exit 0; }
  sleep "$P"
done
st=$(sed -E 's/"time":"[^"]*"//' "$D/status.json" 2>/dev/null | cut -c1-160)
g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -n 1)
pr=$(grep 'eval-set INFO' "$L" 2>/dev/null | tail -n 1 | sed -E 's/^\[[^]]*\] STAGE eval-set INFO //' | cut -c1-200)
echo "TIMEOUT $st gpu=[$g] last=[$pr]"
