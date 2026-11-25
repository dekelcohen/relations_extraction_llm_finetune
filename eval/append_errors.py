"""Append a per-model error-analysis section (root causes, regressions vs base, examples) for a doc-level eval set.

python eval/append_errors.py --report outputs/report/pilot_compare.html --id err_grpo_dev --title "..." \
    --rows data/stage_2_RL_GRPO/grpo_dev.jsonl --model label=preds.jsonl ... --em qwen3.8-27b-ft-int4 --ref qwen3.8-27b-int4
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
from error_analysis import HC_DESC, analyze, load_audit, render_model_errors  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--em", required=True)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--viewer_url", default="http://127.0.0.1:8111/")
    ap.add_argument("--viewer_dataset", default="", help="viewer dataset name for ?dataset= (e.g. test_long, hard_inss)")
    ap.add_argument("--audit", default="outputs/audit/label_noise_audit.jsonl")
    ap.add_argument("--audit_set", default="")
    a = ap.parse_args()
    rows = load(a.rows)
    samples = []
    for r in rows:
        m = r.get("meta", {})
        samples.append({"id": r["id"], "lang": m.get("lang"), "schema_id": m.get("schema_id"), "hard_cases": m.get("hard_cases", []),
                        "gt": json.loads(r["gt"]), "decoys": json.loads(r["decoys"]), "text": r["text"]})
    errors, labels = {}, []
    for spec in a.model:
        l, p = spec.split("=", 1)
        if not os.path.exists(p):   # not every model was run on every set (e.g. Sonnet skipped on hard_inss)
            print(f"skip {l}: {p} not found")
            continue
        preds = {x["id"]: x["raw"] for x in load(p)}
        errors[l] = analyze([s for s in samples if s["id"] in preds], preds)
        labels.append(l)
    audit = load_audit(a.audit, a.audit_set) if a.audit_set else None
    qs = f"?dataset={a.viewer_dataset}" if a.viewer_dataset else ""
    link = lambda sid: f'<a href="{a.viewer_url}{qs}#{sid}" target="_blank">{sid}</a>'  # noqa: E731
    body = render_model_errors(labels, errors, a.em, a.ref, link, HC_DESC, audit, a.title)
    sec = f"<!--sec:{a.id}--><h2 id='{a.id}'>{a.title}</h2>{body}<!--/sec:{a.id}-->"
    h = open(a.report, encoding="utf-8").read()
    h = re.sub(rf"<!--sec:{re.escape(a.id)}-->.*?<!--/sec:{re.escape(a.id)}-->", "", h, flags=re.S)
    i = h.rfind("</body>")
    h = h[:i] + sec + h[i:] if i >= 0 else h + sec
    open(a.report, "w", encoding="utf-8").write(h)
    print("appended", a.id, {l: len(e) for l, e in errors.items()})


if __name__ == "__main__":
    main()
