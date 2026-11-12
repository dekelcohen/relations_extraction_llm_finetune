"""Split stage-2 GRPO rows into train / dev by SOURCE sample id (chunks of one doc never straddle the split).

python scripts/split_grpo.py --all data/stage_2_RL_GRPO/grpo_all.jsonl --docs data/stage_2_RL_GRPO/grpo_docs.jsonl \
    --out_dir data/stage_2_RL_GRPO
  grpo_dev.jsonl   = doc-level rows (from --docs), stratified by lang x primary hard case, H19-heavy
  grpo_train.jsonl = chunked rows (from --all) of every other source sample
Tiers ambiguous / weird are dropped. `oversample` (hard / frontier_hard x2) is applied after the pre-filter.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

EXCLUDE_TIERS = {"ambiguous", "weird"}
UP_TIERS = {"hard", "frontier_hard"}


def source_id(row: dict) -> str:
    return (row.get("meta", {}).get("chunk") or {}).get("parent") or row["id"]


def _primary(row: dict) -> str:
    """Rarest-looking hard case first (H9 is on almost every row, so it is only a fallback)."""
    hc = [h for h in row["meta"].get("hard_cases", []) if h != "H9"]
    pri = ["H19", "H17", "H7c", "H7", "H4", "H1", "H12", "H2", "H11", "H3", "H5", "H10", "H6", "H15", "H16", "H14",
           "H18", "H13"]
    return next((h for h in pri if h in hc), "H9")


def pick_dev(docs: list[dict], n_long: int = 12, n_other: int = 18, seed: int = 0) -> set[str]:
    """Per language: n_long H19 docs + n_other others, round-robin over primary hard-case buckets."""
    rng = random.Random(seed)
    tier_of = {d["id"]: d["meta"].get("tier") for d in docs}
    chosen: set[str] = set()
    for lang in sorted({d["meta"]["lang"] for d in docs}):
        for is_long, k in ((True, n_long), (False, n_other)):
            buckets = defaultdict(list)
            for d in docs:
                if d["meta"]["lang"] == lang and ("H19" in d["meta"].get("hard_cases", [])) == is_long:
                    buckets[_primary(d)].append(d["id"])
            for b in buckets.values():
                b.sort()
                rng.shuffle(b)
                # frontier_hard first (popped from the end) so dev also measures the hardest tier
                b.sort(key=lambda i: tier_of[i] == "frontier_hard")
            order = sorted(buckets)
            got = 0
            while got < k and any(buckets.values()):
                for b in order:
                    if buckets[b] and got < k:
                        chosen.add(buckets[b].pop())
                        got += 1
    return chosen


def split(all_rows: list[dict], docs: list[dict], seed: int = 0, **kw) -> tuple[list[dict], list[dict]]:
    keep = lambda r: r["meta"].get("tier") not in EXCLUDE_TIERS
    dev_ids = pick_dev([d for d in docs if keep(d)], seed=seed, **kw)
    dev = [d for d in docs if d["id"] in dev_ids]
    train = [r for r in all_rows if keep(r) and source_id(r) not in dev_ids]
    return train, dev


def oversample(rows: list[dict], factor: int = 2) -> list[dict]:
    return [r for r in rows for _ in range(factor if r["meta"].get("tier") in UP_TIERS else 1)]


def _read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, dev = split(_read(a.all), _read(a.docs), a.seed)
    _write(os.path.join(a.out_dir, "grpo_train.jsonl"), train)
    _write(os.path.join(a.out_dir, "grpo_dev.jsonl"), dev)
    print(f"train rows={len(train)} sources={len({source_id(r) for r in train})} dev docs={len(dev)}")


if __name__ == "__main__":
    main()
