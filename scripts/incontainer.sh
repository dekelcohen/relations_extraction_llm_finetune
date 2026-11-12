#!/bin/bash
# run a command inside the dev container with the training env: scripts/incontainer.sh "<cmd>"
cd "$(dirname "$0")/../docker" && sudo -n docker compose exec -T -w /NLP/relations_extraction_llm_finetune dev bash -c "export HF_HOME=/data/.cache/huggingface PATH=/data/envs/re/bin:\$PATH CC=/data/envs/re/bin/gcc VLLM_USE_FLASHINFER_SAMPLER=0; set -a; . ./.env; set +a; $1"
