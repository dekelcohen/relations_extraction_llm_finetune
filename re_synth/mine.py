"""Baselines + adversarial mining: predict with weak (OSS) and strong (frontier) models, score, assign tiers."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import chunking, prompting
from .llm import chat
from .schemas import schema_of
from .score import aggregate, parse_prediction, sample_f1, score_sample, slot_types_for

log = logging.getLogger(__name__)
TIER_F1 = 0.99


def sample_messages(s: dict) -> list[dict]:
    return prompting.messages(schema_of(s).to_prompt_dict(), s["text"])


def predict(samples: list[dict], model: str, out_path: Path, workers: int = 8, max_tokens: int = 2000,
            chunk: dict | None = None) -> None:
    """chunk: kwargs of chunking.split (max_chars/overlap/min_chars); long docs are predicted per overlapping chunk
    and the chunk outputs merged (raw = merged JSON, "chunks" = per-chunk raw outputs). None disables chunking."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open(encoding="utf-8")}
    todo = [s for s in samples if s["id"] not in done]
    log.info("predict %s: %d todo -> %s", model, len(todo), out_path)

    def one(s):
        parts = chunking.split(s["text"], **chunk) if chunk is not None else []
        if not parts:
            return chat(sample_messages(s), model, temperature=0.0, max_tokens=max_tokens), None
        sd = schema_of(s).to_prompt_dict()
        raws = [chat(prompting.messages(sd, c), model, temperature=0.0, max_tokens=max_tokens) for c in parts]
        return chunking.merge_raws(raws, slot_types_for(s["schema_id"], inline=s.get("schema")), s["text"]), raws

    with ThreadPoolExecutor(workers) as ex, out_path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(one, s): s["id"] for s in todo}
        for fu in as_completed(futs):
            try:
                raw, parts = fu.result()
            except Exception as e:
                log.warning("predict %s %s error: %s", model, futs[fu], e)
                continue
            row = {"id": futs[fu], "model": model, "raw": raw} | ({"chunks": parts} if parts else {})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def load_preds(path: Path) -> dict[str, str]:
    return {r["id"]: r["raw"] for r in map(json.loads, Path(path).open(encoding="utf-8"))}


def score_preds(samples: list[dict], preds: dict[str, str]) -> dict[str, dict]:
    out = {}
    for s in samples:
        if s["id"] not in preds:
            continue
        recs, valid = parse_prediction(preds[s["id"]])
        out[s["id"]] = score_sample(recs, valid, s["gt"], s.get("decoys", []), s["text"], slot_types_for(s["schema_id"], inline=s.get("schema")))
    return out


DETERMINISTIC_GOLD = ("H6", "H7", "H13", "H19")  # gold comes from a rendered list/table block, not from LLM-written text


def tier(weak_f1: float | None, strong_f1: float | None, deterministic: bool = False) -> str:
    if weak_f1 is None or strong_f1 is None:
        return "unknown"
    w, st = weak_f1 >= TIER_F1, strong_f1 >= TIER_F1
    if not st and deterministic:
        return "frontier_hard"  # frontier errs although gold is reliable: hardest trustworthy set
    if w and st:
        return "easy"
    if st and not w:
        return "hard"  # OSS errs, frontier succeeds: the target set
    if not st and not w:
        return "ambiguous"  # likely GT/label issue or very hard: excluded from SFT, manual review
    return "weird"  # weak right, strong wrong


def mine(samples: list[dict], weak: dict[str, dict], strong: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for s in samples:
        wf = sample_f1(weak[s["id"]]) if s["id"] in weak else None
        sf = sample_f1(strong[s["id"]]) if s["id"] in strong else None
        det = any(c in s.get("hard_cases", []) for c in DETERMINISTIC_GOLD)
        out[s["id"]] = {"tier": tier(wf, sf, det), "weak_f1": wf, "strong_f1": sf}
    return out


def group_report(samples: list[dict], scores: dict[str, dict], keys=("lang", "format", "schema_id")) -> dict:
    rep = {"all": aggregate([scores[s["id"]] for s in samples if s["id"] in scores])}
    for k in keys:
        vals = sorted({s[k] for s in samples})
        for v in vals:
            rep[f"{k}={v}"] = aggregate([scores[s["id"]] for s in samples if s[k] == v and s["id"] in scores])
    hcs = sorted({h for s in samples for h in s["hard_cases"]})
    for h in hcs:
        rep[f"hc={h}"] = aggregate([scores[s["id"]] for s in samples if h in s["hard_cases"] and s["id"] in scores])
    return rep
