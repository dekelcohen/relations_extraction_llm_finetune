"""Token-length + CoT-vs-records stats for data/sft_stage2/train.jsonl (Qwen3.8-27B tokenizer.json)."""
import json
import os
import re
import statistics as st
from collections import defaultdict

from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.expanduser("~/.cache/huggingface/qwen3.8-27b-tok/tokenizer.json"))
rows = [json.loads(l) for l in open("data/sft_stage2/train.jsonl", encoding="utf-8")]
T = lambda s: len(tok.encode(s).ids)  # noqa: E731
out = defaultdict(list)
flags = []
for r in rows:
    m = r["messages"]
    ass = m[-1]["content"]
    cot = re.search(r"<think>\n?(.*?)\n?</think>", ass, re.S).group(1)
    n_rec = len(json.loads(ass.split("</think>", 1)[1])["records"])
    t_all = sum(T(x["content"]) for x in m)
    t_cot = T(cot)
    key = (r["meta"]["source"], r["meta"]["lang"])
    out[key].append((t_all, t_cot, n_rec))
    out[("ALL", "")].append((t_all, t_cot, n_rec))
    if n_rec >= 4 and t_cot / n_rec < 60:
        flags.append((r["id"], n_rec, t_cot))
    r["_"] = (t_all, t_cot, n_rec)
lines = ["| source/lang | rows | total tok mean/p95/max | CoT tok mean/p95/max | records mean/max |", "|---|---|---|---|---|"]
p95 = lambda v: sorted(v)[int(0.95 * (len(v) - 1))]  # noqa: E731
for k, v in sorted(out.items()):
    a, b, c = zip(*v)
    lines.append(f"| {k[0]}{'/' + k[1] if k[1] else ''} | {len(v)} | {st.mean(a):.0f}/{p95(a)}/{max(a)} | {st.mean(b):.0f}/{p95(b)}/{max(b)} | {st.mean(c):.1f}/{max(c)} |")
bins = defaultdict(list)
for r in rows:
    t_all, t_cot, n = r["_"]
    bins["1-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11+"].append((t_cot, t_cot / n))
lines += ["", "| records | rows | CoT tokens mean | CoT tok / record mean |", "|---|---|---|---|"]
for k in ("1-3", "4-6", "7-10", "11+"):
    v = bins.get(k, [])
    if v:
        lines.append(f"| {k} | {len(v)} | {st.mean(x[0] for x in v):.0f} | {st.mean(x[1] for x in v):.0f} |")
lines += ["", f"Flagged short CoT (>=4 records and <60 CoT tokens/record): {len(flags)} rows", ""]
lines += [f"- `{i}`: {n} records, {c} CoT tokens" for i, n, c in sorted(flags, key=lambda x: x[2] / x[1])[:25]]
open("data/sft_stage2/length_stats.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
