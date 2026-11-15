#!/bin/bash
# Run ON THE MACHINE'S HOST: git pull, bring the compose stack up if needed, then run the stages in the container (detached by default).
#   bash scripts/pipeline/run_pipeline.sh --machine a100 --run qwen3.8-27b [--stages "..."] [--limit N] [--foreground]
NO_REEXEC=1 source "$(dirname "$0")/lib.sh"
FG=0; ARGS=(); for a in "$@"; do [ "$a" = --foreground ] && FG=1 || ARGS+=("$a"); done
PIPELINE_CONFIG_LOGGED=1 NO_REEXEC=1 init "${ARGS[@]}" >/dev/null   # config is logged once, by the container side
git pull -q --ff-only 2>/dev/null || true
cd docker
[ -f .env ] || printf "HOST_UID=$(id -u)\nHOST_GID=$(id -g)\nDATA_DIR=$HOST_DATA\nENVS_DIR=$HOST_ENVS\nNLP_DIR=$HOST_NLP\n" > .env   # paths only, no secrets
$DOCKER ps --format '{{.Names}}' | grep -qx "$CONTAINER" || $DOCKER compose -p $COMPOSE_PROJECT up -d --build >/dev/null 2>&1
cd "$ROOT"
unset NO_REEXEC; [ $FG = 0 ] && ARGS+=(--detach)
exec bash scripts/pipeline/pipeline_stages.sh "${ARGS[@]}"
