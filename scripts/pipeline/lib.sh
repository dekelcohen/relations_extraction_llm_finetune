# Sourced by every pipeline script. Usage in a script:  source lib.sh; init "$@"   (re-execs inside the container if RUNNER=docker)
#   --machine <configs/machines/NAME.env> --run <configs/runs/NAME.env> [--detach] [--key value ...]  (--key value overrides any config
#   variable, upper-cased, '-'->'_': --model X, --stages "base_eval sft", --limit 2, --gpu-mem 0.8, --sft-lr 5e-5)
LIBDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$LIBDIR/../.." && pwd)

init() {
  local script=$(basename "$0") ORIG=("$@"); MACHINE=${MACHINE:-}; RUN=${RUN:-}; DETACH=0; OVR_KEYS=()
  while [ $# -gt 0 ]; do
    case $1 in
      --machine) MACHINE=$2; shift 2 ;;
      --run) RUN=$2; shift 2 ;;
      --detach) DETACH=1; shift ;;
      --*) k=$(echo "${1#--}" | tr 'a-z-' 'A-Z_'); if [ $# -gt 1 ] && [[ $2 != --* ]]; then OVR[$k]=$2; shift 2; else OVR[$k]=1; shift; fi; OVR_KEYS+=($k) ;;
      *) POSITIONAL+=("$1"); shift ;;
    esac
  done
  : "${MACHINE:?--machine required (configs/machines/*.env)}"
  CFG_KEYS=$(grep -hoE '^[A-Z_0-9]+=' "$ROOT/configs/machines/$MACHINE.env" ${RUN:+"$ROOT/configs/runs/$RUN.env"} | tr -d '=' | tr '\n' ' ')
  set -a; . "$ROOT/configs/machines/$MACHINE.env"; [ -n "$RUN" ] && . "$ROOT/configs/runs/$RUN.env"
  for k in "${OVR_KEYS[@]}"; do export "$k=${OVR[$k]}"; done; set +a
  CFG_KEYS="$CFG_KEYS ${OVR_KEYS[*]}"
  A100_DIR=$ROOT/outputs/pipeline/$MACHINE; PIPE_LOG=$A100_DIR/pipeline.log; mkdir -p "$A100_DIR"
  if [ "$RUNNER" = docker ] && [ -z "$IN_CONTAINER" ] && [ -z "$NO_REEXEC" ]; then
    if [ "$MACHINE" = a100 ] && [[ $script == pipeline_stages.sh || ( $script == sft_qlora.sh && -z $MAX_STEPS ) ]]; then
      python3 "$LIBDIR/evacuation_guard.py" "$A100_DIR" check --run "$RUN" \
        --docker "$DOCKER" --container "$CONTAINER" || exit $?
    fi
    local env=(-e IN_CONTAINER=1); [ "$DETACH" = 1 ] && env+=(-d)
    if [ "$PUBLISH" = 1 ]; then
      env+=(-e PUBLISH=1 -e HF_TOKEN -e GH_TOKEN)
    fi
    if [ "$DETACH" = 1 ]; then
      local q=(); for a in "${ORIG[@]}"; do [ "$a" = --detach ] || q+=("$a"); done; local qs; qs=$(printf '%q ' "${q[@]}")
      exec $DOCKER exec "${env[@]}" -w "$REPO" "$CONTAINER" bash -c "bash scripts/pipeline/$script $qs >> $REPO/outputs/pipeline/$MACHINE/pipeline.out 2>&1"
    fi
    exec $DOCKER exec "${env[@]}" -w "$REPO" "$CONTAINER" bash "scripts/pipeline/$script" "${ORIG[@]}"
  fi
  cd "$ROOT"
  [ -n "$EXTRA_EXPORTS" ] && eval "$EXTRA_EXPORTS"
  export HF_HOME PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} TOKENIZERS_PARALLELISM=false
  RESOLVED=$(for k in $(echo $CFG_KEYS | tr ' ' '\n' | grep -vE '^(DOCKER|SSH_HOST|HOST_.*|EXTRA_EXPORTS)$' | sort -u); do printf '%s=%s ' $k "${!k}"; done)
  if [ -z "$PIPELINE_CONFIG_LOGGED" ]; then  # resolved config: once per run (children inherit the flag)
    echo "[$script] machine=$MACHINE run=$RUN"; echo "$RESOLVED" | tr ' ' '\n' | grep . | sed 's/^/  /'
    log config INFO "machine=$MACHINE run=$RUN $RESOLVED"
    export PIPELINE_CONFIG_LOGGED=1
  fi
  export PIPE_LOG
}
declare -A OVR; POSITIONAL=()

# log <stage> <START|DONE|FAIL|INFO> [info...]  -> one line in pipeline.log + status.json
log() {
  local st=$1 ev=$2; shift 2; local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
  if [ "$st" = config ]; then   # full resolved config -> config.log; pipeline.log keeps only stage/watchdog events
    echo "[$ts] $*" >> "$A100_DIR/config.log"; echo "[$ts] STAGE config INFO machine=$MACHINE run=$RUN (resolved config: config.log)" >> "$PIPE_LOG"; return
  fi
  echo "[$ts] STAGE $st $ev $*" >> "$PIPE_LOG"
  printf '{"stage":"%s","state":"%s","info":"%s","time":"%s"}\n' "$st" "$ev" "${*//\"/\'}" "$ts" > "$A100_DIR/status.json"
}
# stage <name> <cmd...>: START/DONE/FAIL + elapsed + error tail; detail log $A100_DIR/<name>.log without progress bars
stage() {
  local name=$1; shift; local t0=$SECONDS lf=$A100_DIR/$name.log
  log "$name" START
  local rc=0
  if [ -n "$EVACUATION_STAGE_LOG" ]; then
    ln -sfn -- "$EVACUATION_STAGE_LOG" "$lf" || return 1
    lf=$A100_DIR/$EVACUATION_STAGE_LOG
    EVACUATION_TRAIN_MARKERS=1 "$@" >> "$lf" 2>&1 || rc=$?
    printf '\nEVACUATION_STAGE_EXIT %s\n' "$rc" >> "$lf"
  else
    "$@" > >(tr '\r' '\n' | awk -v pf="$A100_DIR/progress.last" '/[0-9](it\/s|s\/it)|%\|/ {print > pf; close(pf); next} {print}' > "$lf") 2>&1 || rc=$?
  fi
  if [ "$rc" = 0 ]; then
    log "$name" DONE "elapsed=$((SECONDS-t0))s $(cat "$A100_DIR/$name.info" 2>/dev/null)"; rm -f "$A100_DIR/$name.info"
  else
    sleep 1; log "$name" FAIL "elapsed=$((SECONDS-t0))s root=$(root_cause "$lf") tail=$(tail -n 2 "$lf" | tr '\n' ' ' | cut -c1-200)"; return 1
  fi
}
# root_cause <log>: last real error line (skips tracebacks and vLLM's generic "Engine core initialization failed" wrapper)
root_cause() {
  grep -aE '[A-Za-z]+(Error|Exception)(:|$)|: error:|out of memory|Killed' "$1" | grep -avE 'Engine core initialization failed|^\s*(File |Traceback|raise )|WARNING' \
    | tail -n 1 | sed -E 's/^.*\] //' | tr -s ' ' | cut -c1-300
}
py() { local e=$1; shift; micromamba run -p "$e" python "$@"; }

publish_auth() {
  [ "$PUBLISH" = 1 ] && [ "$PUBLISH_AUTH_OK" != 1 ] || return 0
  stage auth-smoke py "$1" "$LIBDIR/publish_results.py" preflight || return 1
  export PUBLISH_AUTH_OK=1
}

publish_result() {
  local env=$1 action=$2; shift 2
  stage "publish-$action" py "$env" "$LIBDIR/publish_results.py" "$action" "$@" ||
    log "publish-$action" WARN "publication failed; retained local results for retry"
}
