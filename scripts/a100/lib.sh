# Shared helpers (sourced). Pipeline log: one line per event; status.json = current stage.
REPO=${REPO:-/NLP/relations_extraction_llm_finetune}
A100_DIR=${A100_DIR:-$REPO/outputs/a100}
PIPE_LOG=$A100_DIR/pipeline.log
mkdir -p "$A100_DIR"

# log <STAGE> <START|DONE|FAIL|INFO> [key info...]
log() {
  local st=$1 ev=$2; shift 2
  local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] STAGE $st $ev $*" >> "$PIPE_LOG"
  printf '{"stage":"%s","state":"%s","info":"%s","time":"%s"}\n' "$st" "$ev" "${*//\"/\'}" "$ts" > "$A100_DIR/status.json"
}

# stage <name> <cmd...>: runs cmd, logs START/DONE/FAIL + elapsed + error tail (detail log: $A100_DIR/<name>.log, no progress bars)
stage() {
  local name=$1; shift
  local t0=$SECONDS lf=$A100_DIR/$name.log
  log "$name" START
  if "$@" > >(tr '\r' '\n' | grep -v -E '[0-9]+(\.[0-9]+)?(it/s|s/it)|%\|' > "$lf") 2>&1; then
    log "$name" DONE "elapsed=$((SECONDS-t0))s $(cat "$A100_DIR/$name.info" 2>/dev/null)"
  else
    sleep 1; log "$name" FAIL "elapsed=$((SECONDS-t0))s err=$(tail -n 3 "$lf" | tr '\n' ' ' | cut -c1-400)"
    return 1
  fi
}

# in-container python of an env: py <env> args...
py() { local e=$1; shift; micromamba run -p /envs/$e python "$@"; }
