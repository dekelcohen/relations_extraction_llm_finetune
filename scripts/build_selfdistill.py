"""Rejection-sampled self-distillation rows: base-INT4-medium traces that re-solve the stage-2 source docs with slot-F1 == 1.
One row per CHUNK prompt (eval chunking), completion = trace + </think> + the model's JSON (== gold by the filter; the
chat template already prefills "<think>\n"). Docs that never reach F1 == 1 fall back to the Sonnet-CoT row only when its CoT is
not short (and the doc is not chunked), else they are dropped. 2x oversampling follows data/sft_stage2/train.jsonl (#dup ids).
  python scripts/build_selfdistill.py --preds outputs/selfdistill/selfdistill-gen.selfdistill.preds.jsonl [--preds2 ...]"""
import argparse, collections, json, re, statistics, sys
from pathlib import Path

sys.path.insert(0, ".")
from re_synth import chunking
from re_synth.score import aggregate, parse_prediction

rd = lambda p: [json.loads(l) for l in Path(p).open(encoding="utf-8")] if Path(p).exists() else []
ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--preds2", default="")
ap.add_argument("--min_think_chars", type=int, default=1500, help="Sonnet fallback only if its CoT is at least this long")
ap.add_argument("--max_think_chars", type=int, default=0, help="drop chunk traces longer than this (0 = keep all)")
ap.add_argument("--out", default="data/sft_stage2/train_selfdistill.jsonl")
a = ap.parse_args()


def chunk_msgs(r, c):
    if c is None:
        return r["prompt"]
    doc = "<<<\n" + r["text"] + "\n>>>"
    msgs = [dict(m) for m in r["prompt"]]
    msgs[-1]["content"] = msgs[-1]["content"].replace(doc, "<<<\n" + c + "\n>>>")
    return msgs


def split_think(t):
    if "</think>" not in t:
        return None
    th, ans = t.rsplit("</think>", 1)
    return th.replace("<think>", "", 1).strip(), ans.strip()


def good_docs(src_rows, preds_path):
    """doc id -> (chunk prompts, [(think, json)]) for docs with F1 == 1, valid JSON everywhere, every chunk closed its think."""
    from re_synth.score import score_sample
    P = {p["id"]: p for p in rd(preds_path)}
    out, tried = {}, 0
    for r in src_rows:
        p = P.get(r["id"])
        if not p:
            continue
        tried += 1
        raws = p.get("chunks") or [p["raw"]]
        recs, ok = parse_prediction(p["raw"])
        m = score_sample(recs, ok, json.loads(r["gt"]), json.loads(r["decoys"]), r["text"], json.loads(r["slot_types"]))
        f1 = aggregate([m]).get("slot_f1")
        parts = [] if len(r["text"]) <= chunking.CHUNK_MIN else chunking.split(r["text"], chunking.CHUNK_CHARS, chunking.CHUNK_OVERLAP, chunking.CHUNK_MIN)
        cs = parts or [None]
        sp = [split_think(t) for t in raws]
        if f1 != 1.0 or not ok or len(cs) != len(raws) or any(s is None or not parse_prediction(s[1])[1] for s in sp):
            continue
        out[r["id"]] = ([chunk_msgs(r, c) for c in cs], sp)
    return out, tried


src = rd("data/sft_stage2/selfdistill_src.jsonl")
S = {r["id"]: r for r in src}
good, tried = good_docs(src, a.preds)
n1 = len(good)
if a.preds2:
    src2 = [S[i] for i in S if i not in good]
    g2, _ = good_docs(src2, a.preds2)
    good.update(g2)
print(f"docs: {len(src)} first-pass F1=1 {n1}, after retry {len(good)}")

train = rd("data/sft_stage2/train.jsonl")
rows, stat, thinks = [], collections.defaultdict(collections.Counter), collections.defaultdict(list)
seen_fb = set()
for t in train:
    pid = t["id"].removesuffix("#dup")
    key = f"{t['meta']['source']}/{t['meta']['lang']}"
    dup = t["id"].endswith("#dup")
    meta = {**t["meta"], "teacher": "self-distill:Qwen3.8-27B-INT4-medium"}
    if pid in good:
        msgs_l, sp = good[pid]
        if a.max_think_chars and any(len(s[0]) > a.max_think_chars for s in sp):
            stat[key]["dropped_long"] += not dup
            continue
        if not dup:
            stat[key]["selfdistill"] += 1
        for i, (m, (th, ans)) in enumerate(zip(msgs_l, sp)):
            rows.append({"id": f"{t['id']}~c{i}" if len(sp) > 1 else t["id"],
                         "messages": [x for x in m if x["role"] != "assistant"] + [{"role": "assistant", "content": f"{th}\n</think>\n\n{ans}"}],
                         "meta": meta})
            thinks[key].append(len(th))
    else:
        c = t["messages"][-1]["content"]
        th = c.split("</think>")[0].replace("<think>", "", 1).strip() if "</think>" in c else ""
        long_doc = len(S[pid]["text"]) > chunking.CHUNK_MIN
        if len(th) >= a.min_think_chars and not long_doc:
            if not dup:
                stat[key]["sonnet_fallback"] += 1
            rows.append({**t, "messages": t["messages"][:-1] + [{"role": "assistant", "content": c.replace("<think>\n", "", 1)}],
                         "meta": {**meta, "teacher": t["meta"]["teacher"]}})
            thinks[key].append(len(th))
        elif not dup:
            stat[key]["dropped"] += 1

with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
q = lambda v, p: sorted(v)[int(len(v) * p)] if v else 0
lines = ["| source/lang | docs | self-distill kept | Sonnet fallback | dropped | kept rate | think chars p50/p90/mean |", "|---|---|---|---|---|---|---|"]
for k in sorted(stat):
    s = stat[k]; n = sum(s.values()); v = thinks[k]
    lines.append(f"| {k} | {n} | {s['selfdistill']} | {s['sonnet_fallback']} | {s['dropped'] + s['dropped_long']} | {s['selfdistill'] / max(n, 1):.0%} | "
                 f"{q(v, .5)}/{q(v, .9)}/{int(statistics.mean(v)) if v else 0} |")
print("\n".join(lines)); print(f"rows={len(rows)} (chunk rows incl. oversampling)")
Path(a.out).with_name("selfdistill_stats.md").write_text("\n".join(lines) + f"\n\nrows={len(rows)}\n")
