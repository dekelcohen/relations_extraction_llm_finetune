"""GRPO pre-filter: n rollouts per training prompt with the SFT adapter (vLLM), scored with train/rewards.py.
Drops prompts whose rewards are all equal (zero group std -> zero gradient), oversamples hard / frontier_hard x2.

python train/prefilter.py --data data/stage_2_RL_GRPO/grpo_train.jsonl --lora outputs/sft-cot-qwen3-4b \
    --out_dir data/stage_2_RL_GRPO
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--lora", default="outputs/sft-cot-qwen3-4b")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_num_seqs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_prompt_tokens", type=int, default=2500, help="GRPO trains on prompts up to this length")
    ap.add_argument("--subset", type=int, default=None, help="random subset (2/3 hard+frontier_hard, 1/3 easy)")
    ap.add_argument("--kv_cache_dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from data import read_jsonl, render_prompt
    from predict import gen_vllm
    from rewards import extraction_reward
    from split_grpo import oversample

    tok = AutoTokenizer.from_pretrained(a.lora)
    rows = read_jsonl(a.data)[: a.limit]
    prompts = [render_prompt(tok, r["prompt"], True) for r in rows]
    short = [len(tok(p, add_special_tokens=False)["input_ids"]) <= a.max_prompt_tokens for p in prompts]
    rows = [r for r, ok in zip(rows, short) if ok]
    if a.subset:
        rng = random.Random(a.seed)
        hard = [r for r in rows if r["meta"]["tier"] in ("hard", "frontier_hard")]
        easy = [r for r in rows if r["meta"]["tier"] not in ("hard", "frontier_hard")]
        rng.shuffle(hard)
        rng.shuffle(easy)
        k = a.subset * 2 // 3
        rows = hard[:k] + easy[: a.subset - min(k, len(hard))]
    prompts = [render_prompt(tok, r["prompt"], True) for r in rows]
    ns = SimpleNamespace(model=a.model, lora=a.lora, n=a.n, temperature=1.0, top_p=0.95, top_k=-1, seed=0,
                         max_tokens=a.max_tokens, tp=1, max_model_len=a.max_model_len, max_lora_rank=32,
                         gpu_mem=a.gpu_mem, quantization=None, kv_cache_dtype=a.kv_cache_dtype, max_num_seqs=a.max_num_seqs,
                         enforce_eager=True, guided_json=False)
    outs = gen_vllm(ns, prompts)
    lens = [len(tok(c, add_special_tokens=False)["input_ids"]) for o in outs for c in o]
    pf, keep = [], []
    for r, o in zip(rows, outs):
        rew = extraction_reward(o, [r["gt"]] * len(o), [r["decoys"]] * len(o), [r["text"]] * len(o),
                                [r["slot_types"]] * len(o))
        std = statistics.pstdev(rew)
        pf.append({"id": r["id"], "tier": r["meta"]["tier"], "mean": statistics.mean(rew), "std": std, "rewards": rew,
                   "completion": o[0]})
        if std > 1e-6:
            keep.append(r)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "prefilter.jsonl"), "w", encoding="utf-8") as f:
        for x in pf:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    final = oversample(keep)
    with open(os.path.join(a.out_dir, "grpo_train_filtered.jsonl"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_tier = {}
    for x in pf:
        t = by_tier.setdefault(x["tier"], {"n": 0, "kept": 0, "mean": []})
        t["n"] += 1
        t["kept"] += x["std"] > 1e-6
        t["mean"].append(x["mean"])
    stats = {"prompts": len(rows), "kept_unique": len(keep), "rows_after_oversample": len(final),
             "completion_tokens": {"mean": statistics.mean(lens), "p50": pct(lens, .5), "p95": pct(lens, .95),
                                   "max": max(lens)},
             "mean_reward": statistics.mean(x["mean"] for x in pf),
             "by_tier": {k: {"n": v["n"], "kept": v["kept"], "mean_reward": statistics.mean(v["mean"])}
                         for k, v in by_tier.items()}}
    with open(os.path.join(a.out_dir, "prefilter_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
