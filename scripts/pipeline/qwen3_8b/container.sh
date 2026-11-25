#!/bin/bash
# Create/start the dedicated Qwen3-8B container relx8b-dev (GPU0 UUID only) and verify it. No secrets involved.
#   container.sh ensure | verify | stop
P=/mnt/dekel/relx-qwen35-9b; W=$P/qwen3-8b-run; C=relx8b-dev; GPU=GPU-e7e4f74b-c511-3094-59c2-e67a390876f6
IMG=relx9b-slim:e9e6f82-dekel
case $1 in
  ensure)
    mkdir -p $W/logs $W/home/tmp $W/state
    if ! docker inspect $C >/dev/null 2>&1; then
      docker run -d --name $C --gpus "device=$GPU" --shm-size 16g --user "$(id -u):$(id -g)" \
        -v $P:/data -v $W/repo:/data/repo -v $W/logs:/data/logs8b -v $W/home:/data/home8b \
        -e MAMBA_ROOT_PREFIX=/data/envs/.mamba -e CONDA_PKGS_DIRS=/data/envs/.pkgs -e PIP_CACHE_DIR=/data/envs/.pip-cache \
        -e HF_HOME=/data/hf_cache -e TMPDIR=/data/home8b/tmp -e HOME=/data/home8b -e TORCH_HOME=/data/cache/torch \
        -e XDG_CACHE_HOME=/data/cache -e TRITON_CACHE_DIR=/data/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
        -e PYTHONUNBUFFERED=1 -e LANG=C.UTF-8 -e ENV_NAME= \
        $IMG sleep infinity || exit 1
    else
      [ "$(docker inspect -f '{{.State.Running}}' $C)" = true ] || docker start $C >/dev/null || exit 1
    fi
    bash "$0" verify ;;
  verify)
    docker exec $C bash -c 'nvidia-smi -L; id -u; ls /data/envs/re-vllm/.setup_ok /data/repo/configs/runs/qwen3-8b.env' || exit 1
    [ "$(docker exec $C nvidia-smi -L | grep -c "$GPU")" = 1 ] || { echo "container does not see GPU0"; exit 1; }
    [ "$(docker exec $C nvidia-smi -L | wc -l)" = 1 ] || { echo "container sees more than one GPU"; exit 1; }
    docker exec -e CUDA_VISIBLE_DEVICES=0 $C /data/envs/re-vllm/bin/python -c "import torch; assert torch.cuda.device_count()==1; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.mem_get_info())" ;;
  stop) docker stop -t 30 $C ;;
  *) echo "usage: container.sh ensure|verify|stop"; exit 2 ;;
esac
