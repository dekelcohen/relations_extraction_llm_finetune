#!/bin/bash
# Run ON THE A100 HOST (detached): starts the container if needed, then runs pipeline_stages.sh inside it.
#   NLP=/mnt/tv_developers/projects/vla/NLP bash scripts/a100/run_pipeline.sh [STAGES="setup download ..."]
set -e
NLP=${NLP:-/mnt/tv_developers/projects/vla/NLP}
R=$NLP/relations_extraction_llm_finetune
cd $R && git pull -q --ff-only || true
cd docker
[ -f .env ] || printf "HOST_UID=$(id -u)\nHOST_GID=$(id -g)\nDATA_DIR=$NLP/hf_cache\nENVS_DIR=$NLP/envs\nNLP_DIR=$NLP\n" > .env  # paths only, no secrets
docker compose -p relx-eval up -d --build >/dev/null 2>&1
mkdir -p $R/outputs/a100
[ -n "$1" ] && export STAGES="$1"
docker exec -d -e STAGES="$STAGES" -e HF_HOME=/data relx-eval-dev-1 bash -c "bash scripts/a100/pipeline_stages.sh >> /NLP/relations_extraction_llm_finetune/outputs/a100/pipeline.out 2>&1" 
echo "started; follow with: scripts/a100/remote.sh status"
