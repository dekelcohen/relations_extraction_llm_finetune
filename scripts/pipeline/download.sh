#!/bin/bash
# Download $DOWNLOAD (space-separated HF repo ids, from the run config) into $HF_HOME. Public models need no token.
source "$(dirname "$0")/lib.sh"; init "$@"
for m in $DOWNLOAD; do micromamba run -p $ENV_VLLM hf download "$m" >/dev/null || exit 1; done
echo "hf_cache=$(du -sh $HF_HOME | cut -f1)" > $A100_DIR/download.info
