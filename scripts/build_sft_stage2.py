"""Assemble data/sft_stage2/train.jsonl: stage-2 CoT rows (minus eval ids/near-dups) + INSS CoT rows + 25% replay.

  python scripts/build_sft_stage2.py [--inss data/hard_inss/cot/sft_cot_train.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from re_synth.teacher_cot import _ngrams, max_overlap  # noqa: E402

OUT = Path("data/sft_stage2")


def rd(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8")] if Path(p).exists() else []


def parent(i):
    return re.sub(r"(#.*)$", "", i)


def doc_of(row):
    u = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return u.split("## Document", 1)[-1]


# audit-driven emphasis (outputs/audit/27b_vs_strong.md, ranked): H9 H20 H12 H2 H1 H13 H11 H18
TOP = ["H9", "H20", "H12", "H2", "H1", "H13", "H11", "H18"]
INSS_TAGS = {"null_tempting", "status_change", "far_header", "near_miss", "coref", "table_head", "table_prose"}
EVAL_LIKE = 0.3


def dup_factor(r, ov):
    """2 for rows matching the audit's top gap types (never for eval-like text), else 1."""
    hc = set(r["meta"].get("hard_cases", []))
    if ov >= EVAL_LIKE:
        return 1
    if r["meta"]["source"] == "inss":
        return 2 if len(hc & INSS_TAGS) >= 2 and ("far_header" in hc or "status_change" in hc) else 1
    score = sum(8 - i for i, t in enumerate(TOP) if t in hc)  # rank weights 8..1
    return 2 if score >= 12 or (r["meta"]["lang"] == "fa" and score >= 11) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inss", default="data/hard_inss/cot/sft_cot_train.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    ev_ids, ev_texts = set(), []
    for f in ("data/stage_2_RL_GRPO/grpo_dev.jsonl", "data/test_long/test_long.jsonl", "data/hard_inss/dev.jsonl"):
        for r in rd(f):
            ev_ids.add(parent(r["id"]))
            ev_texts.append(r["text"])
    for r in rd("data/pilot/samples.jsonl"):
        ev_ids.add(parent(r["id"]))
        ev_texts.append(r["text"])
    grams = [_ngrams(t) for t in ev_texts]

    def clean(rows, src):
        keep, drop = [], Counter()
        for r in rows:
            if parent(r["id"]) in ev_ids:
                drop["id"] += 1
            elif max_overlap(doc_of(r), grams) >= a.thr:
                drop["near_dup"] += 1
            else:
                keep.append({**r, "meta": {**r["meta"], "source": src}})
        return keep, drop

    s2 = rd("data/stage_2_RL_GRPO/cot_he/sft_cot_train.jsonl") + rd("data/stage_2_RL_GRPO/cot_fa/sft_cot_train.jsonl")
    s2k, s2d = clean(s2, "stage2_synth")
    inss = [{**r, "meta": {**r["meta"], "source": "inss"}} for r in rd(a.inss)]
    rep = rd("data/synth_hefa/sft_cot_train.jsonl")
    rng = random.Random(0)
    rep = rng.sample(rep, k=round(len(rep) * 0.25))
    repk, repd = clean(rep, "replay_synth_hefa")
    rows = []
    n_dup = Counter()
    for r in s2k + inss + repk:
        rows.append(r)
        if r["meta"]["source"] != "replay_synth_hefa":
            if dup_factor(r, max_overlap(doc_of(r), grams)) == 2:
                rows.append({**r, "id": r["id"] + "#dup"})
                n_dup[(r["meta"]["source"], r["meta"]["lang"])] += 1
    random.Random(0).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    info = {"stage2_in": len(s2), "stage2_kept": len(s2k), "stage2_dropped": dict(s2d), "inss": len(inss),
            "replay_sampled": len(rep), "replay_kept": len(repk), "replay_dropped": dict(repd), "total": len(rows), "duplicated": {f"{k[0]}/{k[1]}": v for k, v in n_dup.items()},
            "by_source_lang": {f"{k[0]}/{k[1]}": v for k, v in Counter((r["meta"]["source"], r["meta"]["lang"]) for r in rows).items()}}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
