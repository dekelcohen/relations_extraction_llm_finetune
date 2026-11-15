#!/bin/bash
# Download HF models into the shared cache (HF_HOME=/data -> NLP/hf_cache). Public models need no token.
#   download.sh <repo_id> [<repo_id> ...]
export HF_HOME=${HF_HOME:-/data} HF_HUB_DISABLE_PROGRESS_BARS=1
for m in "$@"; do micromamba run -p /envs/re-vllm hf download "$m" > /dev/null; done
echo "hf_cache=$(du -sh /data | cut -f1)" > ${A100_DIR:-/tmp}/download.info
