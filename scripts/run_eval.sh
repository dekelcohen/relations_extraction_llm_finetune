#!/usr/bin/env bash
# Task eval (our synthetic test split + optional converted public data) for one or more models.
# Usage: bash scripts/run_eval.sh <model_path_or_hf_id> [test.jsonl] [--lora <adapter>] [--guided_json]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; shift
DATA=${1:-data/synth/export/grpo_test.jsonl}; [ $# -gt 0 ] && shift
NAME=$(basename "$MODEL")
python eval/predict.py --model "$MODEL" --data "$DATA" --out "outputs/eval/${NAME}.preds.jsonl" "$@"