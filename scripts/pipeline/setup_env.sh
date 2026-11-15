#!/bin/bash
# Create/update the micromamba envs ($ENV_VLLM, $ENV_UNSLOTH) and purge caches. Idempotent; --force 1 rebuilds. Skipped when MANAGE_ENVS=0.
#   setup_env.sh --machine a100 [--which vllm|unsloth|all]
source "$(dirname "$0")/lib.sh"; init "$@"
[ "$MANAGE_ENVS" = 1 ] || { echo "envs unmanaged on $MACHINE (skip)"; exit 0; }
W=${WHICH:-all}
mk() { local p=$1; shift
  if [ -f $p/.setup_ok ] && [ -z "$FORCE" ]; then echo "env $p ok (skip)"; return; fi
  [ -d $p ] || micromamba create -y -q -p $p python=3.12 pip
  micromamba run -p $p pip install -q "$@" && touch $p/.setup_ok; }
[[ $W == vllm || $W == all ]] && mk $ENV_VLLM vllm python-dotenv requests "huggingface_hub[hf_xet]"
[[ $W == unsloth || $W == all ]] && mk $ENV_UNSLOTH unsloth datasets python-dotenv requests "huggingface_hub[hf_xet]"
[ -x $ENV_VLLM/bin/gcc ] || micromamba install -y -q -p $ENV_VLLM -c conda-forge gcc gxx   # triton/vLLM JIT needs a C compiler (slim image has none)
micromamba run -p $ENV_VLLM pip cache purge >/dev/null 2>&1; rm -rf "$(dirname $ENV_VLLM)/.pip-cache" "$(dirname $ENV_VLLM)/.pkgs"
echo "envs=$(du -sh $ENV_VLLM $ENV_UNSLOTH 2>/dev/null | cut -f1 | tr '\n' ',')" > $A100_DIR/setup.info
