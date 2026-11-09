"""Shared data helpers for train/eval: load our jsonl exports and render chat templates."""
from __future__ import annotations

import json
import logging
import random

log = logging.getLogger(__name__)


def read_jsonl(paths: str | list[str]) -> list[dict]:
    paths = [paths] if isinstance(paths, str) else paths
    rows = []
    for p in paths:
        for part in p.split(","):
            with open(part.strip(), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def render_prompt(tok, messages: list[dict], enable_thinking: bool = False) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=enable_thinking)
    except TypeError:  # templates without enable_thinking
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def end_token(tok) -> str:
    """Text that ends an assistant turn for this template (Qwen: <|im_end|>, Llama-3: <|eot_id|>)."""
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "YYY"}],
                                    tokenize=False)
    tail = probe.split("YYY", 1)[-1]
    for cand in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", tok.eos_token or ""):
        if cand and cand in tail:
            return cand
    return tok.eos_token or ""


def to_prompt_completion(rows: list[dict], tok, enable_thinking=False) -> list[dict]:
    """[{messages:[...,assistant]}] -> [{prompt: str, completion: str}] (completion ends with the turn-end token)."""
    end = end_token(tok)
    out = []
    for r in rows:
        msgs = r["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        out.append({"prompt": render_prompt(tok, msgs[:-1], enable_thinking),
                    "completion": msgs[-1]["content"] + end})
    return out


def load_replay(name: str, n: int, seed: int = 0, split: str = "train") -> list[dict]:
    """General instruction data mixed into SFT to limit forgetting (e.g. allenai/tulu-3-sft-mixture,
    HuggingFaceTB/smoltalk). Takes the last assistant turn as the completion."""
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    rows = []
    for ex in ds:
        msgs = ex.get("messages") or ex.get("conversations")
        if not msgs or msgs[-1].get("role") != "assistant" or len(json.dumps(msgs)) > 12000:
            continue
        rows.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
        if len(rows) >= n:
            break
    log.info("replay: %d rows from %s", len(rows), name)
    return rows


def mix(task: list[dict], replay: list[dict], seed: int = 0) -> list[dict]:
    rows = task + replay
    random.Random(seed).shuffle(rows)
    return rows
