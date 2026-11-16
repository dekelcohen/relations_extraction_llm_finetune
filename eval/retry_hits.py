"""Hit-limit retry: re-generate ONLY the prompts (doc chunks) of finished preds that hit max_tokens, i.e. whose
thinking never closed (no </think>; mostly overthinking at the template's default reasoning_effort=xhigh).
Writes <stem>.retry-<tag>.jsonl next to each preds file ({id, chunk, text, hit}); the main preds are never touched.
Re-running appends only chunks not yet in the retry file.
Base and LoRA preds files share one engine (the row's "lora" field picks the adapter).

  # generate (GPU; same args as predict.py, --out = existing preds files):
  python eval/retry_hits.py --tag seed --seed 1 --model M --lora ADAPTER --data D1 --out P1 [--data D2 --out P2 ...] [predict.py flags]
  # score only (no GPU): F1 as-is / with retried chunks substituted / non-hit docs only, + how many retries still hit
  python eval/retry_hits.py --score --tag seed --data D1 --out P1 ... [--tag medium]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P  # noqa: E402
from predict import _pipe_log, chunk_prompt, chunking, read_jsonl, render_prompt, score_rows  # noqa: E402


def hit(t: str) -> bool:
    """max_tokens hit: thinking never closed, or the answer after </think> was cut mid-JSON."""
    if "</think>" not in t:
        return True
    ans = t.rsplit("</think>", 1)[1].strip().removesuffix("```").strip()
    return bool(ans) and ans[-1] not in "}]"


def retry_path(preds: str, tag: str) -> str:
    return preds.removesuffix(".preds.jsonl") + f".retry-{tag}.jsonl"


def load(a, data, preds):
    """-> rows, {id: [chunk texts]}, [(row, chunk_idx, chunk)] of hit chunks, lora of the preds file"""
    rows = read_jsonl(data)
    pr = {d["id"]: d for d in read_jsonl(preds)}
    rows = [r for r in rows if r["id"] in pr]
    texts, todo = {}, []
    for r in rows:
        parts = [] if a.no_chunk else chunking.split(r["text"], a.chunk_chars, a.chunk_overlap, a.chunk_min)
        parts = parts or [None]
        t = pr[r["id"]].get("chunks") or [pr[r["id"]]["raw"]]
        assert len(t) == len(parts), f"{preds} {r['id']}: {len(t)} chunk preds vs {len(parts)} chunks"
        texts[r["id"]] = t
        todo += [(r, j, c) for j, (c, x) in enumerate(zip(parts, t)) if hit(x)]
    return rows, texts, todo, next(iter(pr.values())).get("lora")


def merged(r, chunks):
    return chunks[0] if len(chunks) == 1 else chunking.merge_raws(chunks, json.loads(r["slot_types"]), r["text"])


def f1(rows, texts):
    return score_rows(rows, {r["id"]: merged(r, texts[r["id"]]) for r in rows})[1]["all"]


def score(a, data, preds, tag):
    rows, texts, todo, _ = load(a, data, preds)
    rp = retry_path(preds, tag)
    if not os.path.exists(rp):
        return None
    sub = {i: list(t) for i, t in texts.items()}
    still = n = 0
    for d in read_jsonl(rp):
        sub[d["id"]][d["chunk"]] = d["text"]; n += 1; still += hit(d["text"])
    hit_ids = {r["id"] for r, _, _ in todo}
    clean = [r for r in rows if r["id"] not in hit_ids]
    asis, subst = f1(rows, texts), f1(rows, sub)
    return {"preds": os.path.basename(preds), "tag": tag, "docs": len(rows), "hit_chunks": len(todo), "retried": n,
            "still_hit": still, "f1_asis": asis.get("slot_f1"), "f1_subst": subst.get("slot_f1"),
            "json_valid_asis": asis.get("json_valid"), "json_valid_subst": subst.get("json_valid"),
            "nonhit_docs": len(clean), "f1_nonhit": f1(clean, texts).get("slot_f1") if clean else None}


def main():
    ap = P.build_parser()
    ap.add_argument("--tag", action="append", required=True, help="retry variant name (repeatable with --score)")
    ap.add_argument("--score", action="store_true", help="score existing retry files only (no generation)")
    a = ap.parse_args()
    if not a.score:
        from transformers import AutoTokenizer
        assert len(a.tag) == 1, "generate one --tag per run"
        groups = {}  # lora -> [(preds, row, chunk_idx, chunk)]
        for data, preds in zip(a.data, a.out):
            _, _, todo, lora = load(a, data, preds)
            rp = retry_path(preds, a.tag[0])  # resume: only chunks not retried yet
            have = {(d["id"], d["chunk"]) for d in read_jsonl(rp)} if os.path.exists(rp) else set()
            todo = [x for x in todo if (x[0]["id"], x[1]) not in have]
            groups.setdefault(lora, []).append((preds, todo))
        for lora in sorted(groups, key=lambda x: x is None):  # LoRA group first: the engine is created with enable_lora
            a.lora = lora or None
            tok = AutoTokenizer.from_pretrained(lora or a.model)
            items = [(p, r, j, c) for p, todo in groups[lora] for r, j, c in todo]
            prompts = [render_prompt(tok, chunk_prompt(r, c), a.enable_thinking, **json.loads(a.chat_kw or "{}"))
                       for _, r, _, c in items]
            outs = P.gen_vllm(a, prompts) if prompts else []
            for preds, _ in groups[lora]:
                with open(retry_path(preds, a.tag[0]), "a", encoding="utf-8") as f:
                    for (p, r, j, _), t in zip(items, outs):
                        if p == preds:
                            f.write(json.dumps({"id": r["id"], "chunk": j, "text": t, "hit": hit(t), "lora": lora,
                                                "tag": a.tag[0], "seed": a.seed, "chat_kw": a.chat_kw}, ensure_ascii=False) + "\n")
    res = [s for data, preds in zip(a.data, a.out) for t in a.tag if (s := score(a, data, preds, t))]
    for s in res:
        _pipe_log("retry " + " ".join(f"{k}={v}" for k, v in s.items()))
        print(json.dumps(s))


if __name__ == "__main__":
    main()
