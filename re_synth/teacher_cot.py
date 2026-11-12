"""Teacher chain-of-thought SFT data (thinking-mode targets) + verification in one call.

A strong teacher solves each synthetic sample *blind* (no GT), writing its reasoning in <think>...</think> before
the JSON. If its answer matches the by-construction GT (sample F1 >= TIER_F1) the sample is kept and the SFT target
is "<think>\\n{teacher reasoning}\\n</think>\\n\\n{GT json}" (Qwen3 thinking-mode format); otherwise the sample is
dropped as ambiguous / unverified. Samples whose text near-duplicates a reference set (e.g. the pilot testset) are
dropped too.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompting
from .llm import chat
from .mine import TIER_F1
from .schemas import schema_of
from .score import parse_prediction, sample_f1, score_sample, slot_types_for

log = logging.getLogger(__name__)

COT_INSTRUCTION = (
    "\n\nBefore the JSON, reason step by step inside <think>...</think> (in English, concise, at most ~250 words): "
    "list each candidate main entity of the requested type, then for every slot quote the supporting text and say "
    "why nearby look-alike values (other entities, earlier/other dates, previous locations, etc.) are rejected. "
    "After </think>, output only the JSON object.")


def cot_messages(s: dict) -> list[dict]:
    msgs = prompting.messages(schema_of(s).to_prompt_dict(), s["text"])
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + COT_INSTRUCTION}
    return msgs


def split_cot(raw: str) -> tuple[str | None, str]:
    m = re.search(r"<think>(.*?)</think>", raw or "", flags=re.S)
    if not m:
        return None, raw or ""
    return m.group(1).strip(), raw[m.end():].strip()


def _ngrams(t: str, n: int = 8) -> set[str]:
    t = re.sub(r"\s+", " ", t)
    return {t[i:i + n] for i in range(0, max(0, len(t) - n + 1), 3)}


def max_overlap(text: str, ref_grams: list[set[str]]) -> float:
    g = _ngrams(text)
    if not g:
        return 0.0
    return max((len(g & r) / len(g) for r in ref_grams), default=0.0)


def run(samples: list[dict], model: str, out_dir: Path, ref_texts: list[str] | None = None, workers: int = 8,
        dedup_thr: float = 0.5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "teacher_raw.jsonl"
    done = {}
    if raw_path.exists():
        done = {r["id"]: r["raw"] for r in map(json.loads, raw_path.open(encoding="utf-8"))}
    todo = [s for s in samples if s["id"] not in done]
    log.info("teacher-cot %s: %d todo (%d cached) -> %s", model, len(todo), len(done), out_dir)
    with ThreadPoolExecutor(workers) as ex, raw_path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(chat, cot_messages(s), model, 0.0, 6000): s["id"] for s in todo}
        for fu in as_completed(futs):
            try:
                raw = fu.result()
            except Exception as e:
                log.warning("teacher %s error: %s", futs[fu], e)
                continue
            done[futs[fu]] = raw
            f.write(json.dumps({"id": futs[fu], "model": model, "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()

    ref_grams = [_ngrams(t) for t in (ref_texts or [])]
    stats = {"n": len(samples), "no_output": 0, "no_think": 0, "mismatch": 0, "near_dup": 0, "kept": 0}
    keep, verdicts = [], {}
    for s in samples:
        raw = done.get(s["id"])
        if raw is None:
            stats["no_output"] += 1
            continue
        cot, answer = split_cot(raw)
        recs, valid = parse_prediction(answer)
        m = score_sample(recs, valid, s["gt"], s.get("decoys", []), s["text"],
                         slot_types_for(s["schema_id"], inline=s.get("schema")))
        f1 = sample_f1(m)
        ov = max_overlap(s["text"], ref_grams) if ref_grams else 0.0
        verdicts[s["id"]] = {"f1": round(f1, 4), "has_think": cot is not None, "ref_overlap": round(ov, 3)}
        if not cot:
            stats["no_think"] += 1
        elif f1 < TIER_F1:
            stats["mismatch"] += 1
        elif ov >= dedup_thr:
            stats["near_dup"] += 1
        else:
            stats["kept"] += 1
            gt = prompting.target_json(s["gt"], slot_order=[sl.name for sl in schema_of(s).slots])
            msgs = prompting.messages(schema_of(s).to_prompt_dict(), s["text"])
            keep.append({"id": s["id"], "messages": msgs + [
                {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n\n{gt}"}],
                "meta": {k: s[k] for k in ("lang", "format", "schema_id", "hard_cases")} | {"teacher": model}})
    with (out_dir / "sft_cot_train.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    (out_dir / "teacher_verdicts.json").write_text(json.dumps({"stats": stats, "per_sample": verdicts}, indent=1))
    log.info("teacher-cot stats: %s", stats)
    return stats
