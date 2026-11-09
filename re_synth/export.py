"""Export samples to training files (conversational, model-agnostic; chat template applied in train/*.py).

SFT  rows: {"id", "messages": [system, user, assistant], "meta": {...}}
GRPO rows: {"id", "prompt": [system, user], "gt", "decoys", "text", "slot_types" (JSON strings), "meta"}
Augmentations (train split only): slot alias renaming, slot order shuffle, dropped definitions, k-shot examples.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from . import prompting
from .schemas import schema_of

log = logging.getLogger(__name__)


def split_of(sid: str, test_frac=0.1, val_frac=0.05) -> str:
    h = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def _variant(s: dict, rng: random.Random, pool: list[dict], augment: bool) -> tuple[list[dict], dict, dict]:
    schema = schema_of(s)
    rename = {}
    if augment and rng.random() < 0.3:
        rename = {sl.name: rng.choice(sl.aliases) for sl in schema.slots if sl.aliases and rng.random() < 0.6}
    sdict = schema.to_prompt_dict(rename, drop_defs=augment and rng.random() < 0.15)
    order = [sl.name for sl in schema.slots]
    if augment and rng.random() < 0.5:
        idx = list(range(len(order)))
        rng.shuffle(idx)
        order = [order[i] for i in idx]
        sdict["slots"] = [sdict["slots"][i] for i in idx]
    shots = []
    if augment:
        k = rng.choice([0, 0, 1, 2])
        cands = [p for p in pool if p["schema_id"] == s["schema_id"] and p["id"] != s["id"] and len(p["text"]) < 2500]
        for ex in rng.sample(cands, k=min(k, len(cands))):
            shots.append({"text": ex["text"], "output": json.loads(prompting.target_json(ex["gt"], rename, order))})
    msgs = prompting.messages(sdict, s["text"], shots)
    slot_types = {rename.get(sl.name, sl.name): sl.type for sl in schema.slots}
    gt = json.loads(prompting.target_json(s["gt"], rename, order))
    return msgs, gt, slot_types


def export(samples: list[dict], out_dir: Path, tiers: dict[str, dict] | None = None, seed: int = 0,
           n_aug: int = 1, hard_dup: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts = {}
    files = {(kind, sp): (out_dir / f"{kind}_{sp}.jsonl").open("w", encoding="utf-8")
             for kind in ("sft", "grpo") for sp in ("train", "val", "test")}
    train_pool = [s for s in samples if split_of(s["id"]) == "train"]
    try:
        for s in samples:
            sp = split_of(s["id"])
            t = (tiers or {}).get(s["id"], {}).get("tier", "unknown")
            meta = {k: s[k] for k in ("lang", "format", "schema_id", "hard_cases")} | {"tier": t}
            # GRPO: all tiers except ambiguous (no trustworthy reward); one canonical variant per sample
            if t != "ambiguous" or sp != "train":
                msgs, gt, st = _variant(s, rng, train_pool, augment=False)
                files[("grpo", sp)].write(json.dumps({
                    "id": s["id"], "prompt": msgs, "gt": json.dumps(gt, ensure_ascii=False),
                    "decoys": json.dumps(s.get("decoys", []), ensure_ascii=False), "text": s["text"],
                    "slot_types": json.dumps(st), "meta": meta}, ensure_ascii=False) + "\n")
                counts[("grpo", sp)] = counts.get(("grpo", sp), 0) + 1
            if sp == "train" and t == "ambiguous":
                continue
            reps = 1 if sp != "train" else n_aug * (hard_dup if t == "hard" else 1)
            for r in range(reps):
                augment = sp == "train" and (r > 0 or rng.random() < 0.7)
                msgs, gt, _ = _variant(s, rng, train_pool, augment=augment)
                msgs = msgs + [{"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)}]
                files[("sft", sp)].write(json.dumps({"id": f"{s['id']}#{r}", "messages": msgs, "meta": meta},
                                                    ensure_ascii=False) + "\n")
                counts[("sft", sp)] = counts.get(("sft", sp), 0) + 1
    finally:
        for f in files.values():
            f.close()
    out = {f"{k}_{sp}": v for (k, sp), v in sorted(counts.items())}
    log.info("export -> %s: %s", out_dir, out)
    return out
