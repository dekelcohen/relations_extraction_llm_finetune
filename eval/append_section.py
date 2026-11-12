"""Append (or replace) a model-comparison section at the END of an HTML report.

python eval/append_section.py --report outputs/report/pilot_compare.html --id grpo_dev --title "grpo_dev" \
    --rows data/stage_2_RL_GRPO/grpo_dev.jsonl --model qwen3-4b-ft=outputs/eval/sft.grpo_dev.preds.jsonl \
    --model qwen3-4b-grpo=outputs/eval/grpo.grpo_dev.preds.jsonl --note "text under the title"

--rows: doc-level grpo-format rows (gt/decoys/slot_types as JSON strings, meta). Scoring = the report's lenient mode
(re_synth.score.parse_prediction_lenient + score_sample). Groups: all, lang, tier, hard case.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from re_synth.score import aggregate, parse_prediction_lenient, score_sample  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score_model(rows, preds):
    groups = {}
    for r in rows:
        recs, ok, _ = parse_prediction_lenient(preds.get(r["id"]))
        m = score_sample(recs, ok, json.loads(r["gt"]), json.loads(r["decoys"]), r["text"], json.loads(r["slot_types"]))
        meta = r.get("meta", {})
        for k in ["all", f"lang={meta.get('lang')}", f"tier={meta.get('tier')}"] + [f"hc={h}" for h in meta.get("hard_cases", [])]:
            groups.setdefault(k, []).append(m)
    return {k: aggregate(v) for k, v in groups.items()}


def section(a) -> str:
    rows = load(a.rows)
    labels, agg = [], {}
    for spec in a.model:
        label, path = spec.split("=", 1)
        preds = {x["id"]: x["raw"] for x in load(path)}
        sub = [r for r in rows if r["id"] in preds]
        labels.append(label)
        agg[label] = score_model(sub, preds)
    keys = sorted({k for g in agg.values() for k in g}, key=lambda k: (k != "all", not k.startswith("lang"), k))
    head = "".join(f"<th>{html.escape(l)}<br>slot-F1 / exact</th>" for l in labels)
    body = ""
    for k in keys:
        n = max(agg[l].get(k, {}).get("n", 0) for l in labels)
        cells = "".join(
            "<td>" + (f"{agg[l][k]['slot_f1']:.3f} / {agg[l][k]['exact_sample']:.2f}" if k in agg[l] else "") + "</td>"
            for l in labels)
        body += f"<tr><td class=g>{html.escape(k)}</td><td>{n}</td>{cells}</tr>"
    note = f"<p>{html.escape(a.note)}</p>" if a.note else ""
    return (f"<!--sec:{a.id}--><h2 id='{a.id}'>{html.escape(a.title)}</h2>{note}"
            f"<table border=1 cellpadding=3 style='border-collapse:collapse;font-size:13px'><tr><th>group</th><th>n</th>"
            f"{head}</tr>{body}</table><!--/sec:{a.id}-->")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--model", action="append", required=True, help="label=preds.jsonl, in column order")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    with open(a.report, encoding="utf-8") as f:
        h = f.read()
    h = re.sub(rf"<!--sec:{re.escape(a.id)}-->.*?<!--/sec:{re.escape(a.id)}-->", "", h, flags=re.S)
    sec = section(a)
    i = h.rfind("</body>")
    h = h[:i] + sec + h[i:] if i >= 0 else h + sec
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(h)
    print("appended", a.id, "->", a.report)


if __name__ == "__main__":
    main()
