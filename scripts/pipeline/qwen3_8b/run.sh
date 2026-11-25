#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/pipeline/qwen3_8b/run.sh [options]
Run on the host with the configured GPU container already running.
  --machine NAME          Machine config (default: local3070)
  --stages "STAGE ..."    Ordered subset (default: download train1 train2 eval_base eval_stage2 report)
  --stage2-adapter PATH   Existing adapter for eval_stage2, as seen inside the container
  --hf-batch-size N       Both evaluations' batch size (default: 1)
  --max-tokens N          Both evaluations' generation cap (default: 8192)
  --max-model-len N       Both evaluations' context cap (default: 16384)
  --foreground           Wait for completion instead of launching with nohup
  --dry-run              Print commands without starting work
EOF
}

script=$(realpath -- "${BASH_SOURCE[0]}")
cd "$(dirname "$script")/../../.."
machine=local3070
stages="download train1 train2 eval_base eval_stage2 report"
adapter=outputs/sft-cot-qwen3-8b-stage2
batch=1 max_tokens=8192 max_len=16384
foreground=0 dry_run=0
while (($#)); do
  case $1 in
    --help|-h) usage; exit 0 ;;
    --foreground) foreground=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --machine|--stages|--stage2-adapter|--hf-batch-size|--max-tokens|--max-model-len)
      if (($# < 2)) || [[ -z $2 || $2 == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2; exit 2
      fi
      case $1 in
        --machine) machine=$2 ;;
        --stages) stages=$2 ;;
        --stage2-adapter) adapter=$2 ;;
        --hf-batch-size) batch=$2 ;;
        --max-tokens) max_tokens=$2 ;;
        --max-model-len) max_len=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ ! $machine =~ ^[a-zA-Z0-9._-]+$ || ! -f configs/machines/$machine.env ]]; then
  printf 'Unknown machine: %s\n' "$machine" >&2; exit 2
fi
for n in "$batch" "$max_tokens" "$max_len"; do
  if [[ ! $n =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive integer, got %s\n' "$n" >&2; exit 2
  fi
done
read -r -a steps <<< "$stages"
if ((${#steps[@]} == 0)) || [[ $stages == *$'\n'* ]]; then
  printf 'Expected a space-separated stage list\n' >&2; exit 2
fi
for step in "${steps[@]}"; do
  case $step in
    download|train1|train2|eval_base|eval_stage2|report) ;;
    *) printf 'Unknown stage: %s\n' "$step" >&2; exit 2 ;;
  esac
done

run_dir="outputs/pipeline/$machine/qwen3-8b"
log="$run_dir/run.log"
if (( ! dry_run )); then
  mkdir -p "$run_dir"
  if (( ! foreground )); then
    setsid nohup bash "$script" --foreground --machine "$machine" --stages "$stages" \
      --stage2-adapter "$adapter" --hf-batch-size "$batch" --max-tokens "$max_tokens" \
      --max-model-len "$max_len" >> "$log" 2>&1 < /dev/null &
    printf 'Launcher PID %s; follow %s\n' "$!" "$log"
    exit 0
  fi
  exec 9>"$run_dir/run.lock"
  flock -n 9 || { printf 'Qwen3-8B pipeline already running on %s\n' "$machine" >&2; exit 1; }
fi

pipeline=(bash scripts/pipeline/pipeline_stages.sh --machine "$machine")
eval_cmd=(bash scripts/pipeline/eval_model.sh --machine "$machine"
          --model unsloth/Qwen3-8B-unsloth-bnb-4bit --quant bitsandbytes
          --hf-batch-size "$batch" --max-tokens "$max_tokens" --max-model-len "$max_len")
run() {
  if (( dry_run )); then
    printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

for step in "${steps[@]}"; do
  printf '%s START %s\n' "$(date -Is)" "$step"
  rc=0
  case $step in
    download) run "${pipeline[@]}" --run qwen3-8b --stages download_sft || rc=$? ;;
    train1) run "${pipeline[@]}" --run qwen3-8b --stages 'sft_smoke sft' || rc=$? ;;
    train2) run "${pipeline[@]}" --run qwen3-8b-stage2 --stages 'sft_smoke sft' || rc=$? ;;
    eval_base) run "${eval_cmd[@]}" --run qwen3-8b --name qwen3-8b-bnb4 || rc=$? ;;
    eval_stage2) run "${eval_cmd[@]}" --run qwen3-8b-stage2 --name qwen3-8b-ft-stage2-bnb4 --lora "$adapter" || rc=$? ;;
    report) run python3 scripts/build_qwen3_8b_report.py || rc=$? ;;
  esac
  if ((rc)); then
    printf '%s FAIL %s (exit %s)\n' "$(date -Is)" "$step" "$rc" >&2
    exit "$rc"
  fi
  printf '%s DONE %s\n' "$(date -Is)" "$step"
done
