if [[ ${0##*/} == eval_model.sh && ${NAME:-} == qwen3.5-9b-ft && ${LIMIT:-} == 3 && -n ${LORA:-} ]]; then
    optional_ft1_smoke_exit() {
        local rc=$?
        if (( rc != 0 )); then
            printf '[%s] OPTIONAL smoke_ft1_eval failed (rc=%s); continuing to stage 2\n' "$(date '+%F %T')" "$rc" | tee -a "${RELX_ORCH_LOG:-/data/logs/orch.log}" >&2
            exit 0
        fi
    }
    trap optional_ft1_smoke_exit EXIT
fi
