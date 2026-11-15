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
    local env=(-e IN_CONTAINER=1); [ "$DETACH" = 1 ] && env+=(-d)
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
  echo "[$ts] STAGE $st $ev $*" >> "$PIPE_LOG"
  [ "$st" = config ] && return
  printf '{"stage":"%s","state":"%s","info":"%s","time":"%s"}\n' "$st" "$ev" "${*//\"/\'}" "$ts" > "$A100_DIR/status.json"
}
# stage <name> <cmd...>: START/DONE/FAIL + elapsed + error tail; detail log $A100_DIR/<name>.log without progress bars
stage() {
  local name=$1; shift; local t0=$SECONDS lf=$A100_DIR/$name.log
  log "$name" START
  if "$@" > >(tr '\r' '\n' | awk -v pf="$A100_DIR/progress.last" '/[0-9](it\/s|s\/it)|%\|/ {print > pf; close(pf); next} {print}' > "$lf") 2>&1; then
    log "$name" DONE "elapsed=$((SECONDS-t0))s $(cat "$A100_DIR/$name.info" 2>/dev/null)"; rm -f "$A100_DIR/$name.info"
  else
    sleep 1; log "$name" FAIL "elapsed=$((SECONDS-t0))s err=$(tail -n 3 "$lf" | tr '\n' ' ' | cut -c1-400)"; return 1
  fi
}
py() { local e=$1; shift; micromamba run -p "$e" python "$@"; }
