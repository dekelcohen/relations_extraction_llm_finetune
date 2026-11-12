# Sourced by the entrypoint and by interactive bash (so `docker compose exec` gets it too).
# ENV_NAME=foo -> activates /data/envs/foo, else /envs/foo (if it exists).
eval "$(micromamba shell hook -s bash)"
if [ -n "${ENV_NAME:-}" ]; then
    for d in "/data/envs/${ENV_NAME}" "/envs/${ENV_NAME}"; do
        if [ -d "$d/conda-meta" ]; then micromamba activate "$d"; break; fi
    done
    if [ -z "${CONDA_PREFIX:-}" ]; then
        echo "[gpu-dev] env ${ENV_NAME} not found in /data/envs or /envs. Create it with:" >&2
        echo "  micromamba create -y -p /data/envs/${ENV_NAME} python=3.12 pip" >&2
    fi
fi
