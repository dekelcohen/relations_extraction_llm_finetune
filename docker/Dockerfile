# Slim GPU dev image: no CUDA/torch inside. Python envs live on the host in /envs (micromamba -p),
# CUDA libs come from the torch cu12x wheels in each env, and the driver (535 / CUDA 12.2 here) is
# injected at runtime by the NVIDIA Container Toolkit (`gpus: all`). cu121..cu128 wheels run on
# driver >= 525 via CUDA minor-version compatibility.
FROM debian:bookworm-slim

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG USERNAME=dev

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MAMBA_ROOT_PREFIX=/envs/.mamba \
    CONDA_PKGS_DIRS=/envs/.pkgs \
    PIP_CACHE_DIR=/envs/.pip-cache \
    HF_HOME=/data/.cache/huggingface \
    TORCH_HOME=/data/.cache/torch \
    ENV_NAME=

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl less procps tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=mambaorg/micromamba:2-debian12-slim /bin/micromamba /usr/local/bin/micromamba

# -l: no lastlog entry (a huge UID would otherwise create a multi-GB sparse /var/log/lastlog)
RUN groupadd -g ${HOST_GID} ${USERNAME} \
    && useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${USERNAME} \
    && mkdir -p /envs /data && chown ${HOST_UID}:${HOST_GID} /envs /data

COPY activate-env.sh /usr/local/bin/activate-env.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && echo 'source /usr/local/bin/activate-env.sh' >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /data
ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
