"""Rebuild grpo_train_filtered.jsonl from prefilter.jsonl: keep prompts with reward std > 0 whose sampled
completion fits the GRPO completion budget (HF generate is slow, so long-thinking prompts are skipped), then
oversample hard / frontier_hard x2.

python train/refilter.py --dir data/stage_2_RL_GRPO --tok outputs/sft-cot-qwen3-4b --max_len 1300
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from split_grpo import oversample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tok", default="outputs/sft-cot-qwen3-4b")
    ap.add_argument("--max_len", type=int, default=1300)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json"))
    pf = [json.loads(l) for l in open(os.path.join(a.dir, "prefilter.jsonl"), encoding="utf-8")]
    ok = {x["id"] for x in pf if x["std"] > 1e-6 and len(t.encode(x["completion"]).ids) <= a.max_len}
    rows = [json.loads(l) for l in open(os.path.join(a.dir, "grpo_train.jsonl"), encoding="utf-8")]
    keep = [r for r in rows if r["id"] in ok]
    final = oversample(keep)
    with open(os.path.join(a.dir, "grpo_train_filtered.jsonl"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tiers = {}
    for r in keep:
        tiers[r["meta"]["tier"]] = tiers.get(r["meta"]["tier"], 0) + 1
    print(json.dumps({"kept_unique": len(keep), "rows_after_oversample": len(final), "by_tier": tiers,
                      "max_len": a.max_len, "mean_reward": statistics.mean(x["mean"] for x in pf if x["id"] in ok)}))


if __name__ == "__main__":
    main()
