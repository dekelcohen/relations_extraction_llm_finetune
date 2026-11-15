#!/bin/bash
# Create/update micromamba envs under /envs (container). Idempotent via marker files.
#   setup_env.sh [vllm|unsloth|all]   (default all)   FORCE=1 to rebuild
set -e
W=${1:-all}
mk() { # name "pip packages"
  local n=$1; shift
  if [ -f /envs/$n/.setup_ok ] && [ -z "$FORCE" ]; then echo "env $n ok (skip)"; return; fi
  [ -d /envs/$n ] || micromamba create -y -q -p /envs/$n python=3.12 pip
  micromamba run -p /envs/$n pip install -q "$@"
  touch /envs/$n/.setup_ok
}
[[ $W == vllm || $W == all ]] && mk re-vllm vllm python-dotenv requests "huggingface_hub[hf_xet]"
[[ $W == unsloth || $W == all ]] && mk re-unsloth unsloth datasets python-dotenv requests "huggingface_hub[hf_xet]"
micromamba run -p /envs/re-vllm pip cache purge >/dev/null 2>&1 || true
rm -rf /envs/.pip-cache /envs/.pkgs 2>/dev/null || true
echo "envs: $(du -sh /envs/re-vllm /envs/re-unsloth 2>/dev/null | tr '\n' ' ')" > ${A100_DIR:-/tmp}/setup.info 2>/dev/null || true
