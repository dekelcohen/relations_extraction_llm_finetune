#!/usr/bin/env bash
# Data pipeline (runs anywhere with internet; no GPU). Needs OPENROUTER_API_KEY (env or RE_ENV_FILE=<.env>)
# and llm_providers on PYTHONPATH (e.g. export PYTHONPATH=/path/to/llm_providers).
# Usage: N=500 bash scripts/gen_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-300}; OUT=${OUT:-data/synth}; LANGS=${LANGS:-he,fa,en}; GEN=${GEN:-gen}
python -m re_synth seeds --langs "$LANGS" --n $(( N / 2 + 20 )) --out "$OUT/seeds.jsonl"
python -m re_synth build --seeds "$OUT/seeds.jsonl" --out "$OUT/samples.jsonl" --langs "$LANGS" --n "$N" --model "$GEN"
python -m re_synth predict --samples "$OUT/samples.jsonl" --model weak   --out "$OUT/preds_weak.jsonl"
python -m re_synth predict --samples "$OUT/samples.jsonl" --model strong --out "$OUT/preds_strong.jsonl"
python -m re_synth score --samples "$OUT/samples.jsonl" --preds "$OUT/preds_weak.jsonl"   --out "$OUT/score_weak.json"
python -m re_synth score --samples "$OUT/samples.jsonl" --preds "$OUT/preds_strong.jsonl" --out "$OUT/score_strong.json"
python -m re_synth mine --samples "$OUT/samples.jsonl" --weak "$OUT/preds_weak.jsonl" --strong "$OUT/preds_strong.jsonl" \
  --out "$OUT/tiers.json"
python -m re_synth export --samples "$OUT/samples.jsonl" --tiers "$OUT/tiers.json" --out "$OUT/export" --n-aug 2