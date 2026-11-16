"""Eval-format source rows (id,prompt,gt,decoys,text,slot_types,meta) for every unique doc of data/sft_stage2/train.jsonl,
so base-INT4-medium can re-solve them (eval/predict.py, same chunking as eval). -> data/sft_stage2/selfdistill_src.jsonl
Row <-> source matching is by the user prompt text (the train row's prompt must equal the source row's prompt)."""
import json
from pathlib import Path

D = Path("data")
rd = lambda p: [json.loads(l) for l in Path(p).open(encoding="utf-8")]
user = lambda msgs: next(m["content"] for m in msgs if m["role"] == "user")

train = rd(D / "sft_stage2/train.jsonl")
srcs = {}
for f in ("stage_2_RL_GRPO/grpo_all.jsonl", "stage_2_RL_GRPO/grpo_docs.jsonl", "hard_inss/train.jsonl", "sft_stage2/_replay_eval.jsonl"):
    for r in rd(D / f):
        srcs.setdefault(r["id"], []).append((f, r))
ev = set()
for f in ("stage_2_RL_GRPO/grpo_dev.jsonl", "test_long/test_long.jsonl", "hard_inss/dev.jsonl", "pilot/export/pilot_all.jsonl"):
    ev |= {r["id"].split("#")[0] for r in rd(D / f)}
out, seen, stat = [], set(), {"matched": 0, "text_matched": 0, "unmatched": 0}
for t in train:
    pid = t["id"].removesuffix("#dup")
    if pid in seen:
        continue
    seen.add(pid)
    assert pid.split("#")[0] not in ev, pid
    u = user(t["messages"])
    cands = srcs.get(pid, [])
    hit = next((r for f, r in cands if user(r["prompt"]) == u), None)
    if hit:
        stat["matched"] += 1
    else:
        hit = next((r for f, r in cands if r["text"].strip() in u), None)  # same document, different instruction variant
        stat["text_matched" if hit else "unmatched"] += 1
    if hit:
        # keep the TRAIN row's prompt (what stage 1/2 saw) so generation and training prompts agree
        hit = {**hit, "prompt": [m for m in t["messages"] if m["role"] != "assistant"],
               "meta": {**hit.get("meta", {}), "src": t["meta"]["source"], "lang": t["meta"]["lang"]}}
        out.append(hit)
print(stat, len(out))
with (D / "sft_stage2/selfdistill_src.jsonl").open("w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
