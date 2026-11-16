"""Two-model error comparison on a doc-level set (CPU only): per-doc slot-F1 diff, error categories / root causes per
model, thinking length, and the errors only model A makes (where A loses to B) with example ids.
  python scripts/compare_ft_base.py --data data/test_long/test_long.jsonl \
      --a ft=outputs/eval/qwen3.8-27b-ft-int4.test_long.preds.jsonl --b base-medium=outputs/eval/qwen3.8-27b-int4-medium.test_long.preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eval")]
from error_analysis import analyze  # noqa: E402
from predict import score_rows  # noqa: E402


def think_chars(raw: str, chunks: list | None) -> list[int]:
    return [len(t.split("</think>", 1)[0]) if "</think>" in t else len(t) for t in (chunks or [raw])]


def f1(p):
    d = p.get("pred_values", 0) + p.get("n_gold_values", 0)
    return 2 * p.get("tp", 0) / d if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", required=True, help="name=preds (the model whose losses are examined)")
    ap.add_argument("--b", required=True, help="name=preds (reference)")
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    js = lambda v: json.loads(v) if isinstance(v, str) else v  # noqa: E731
    samples = [r | {k: r["meta"].get(k) for k in ("lang", "schema_id")} | {"hard_cases": r["meta"].get("hard_cases", [])}
               | {"gt": js(r["gt"]), "decoys": js(r.get("decoys") or [])} for r in rows]
    M = {}
    for spec in (a.a, a.b):
        name, path = spec.split("=", 1)
        pr = {d["id"]: d for d in map(json.loads, open(path, encoding="utf-8"))}
        raw = {i: d["raw"] for i, d in pr.items()}
        per, rep = score_rows(rows, raw)
        errs = analyze(samples, raw)
        th = {i: think_chars(d["raw"], d.get("chunks")) for i, d in pr.items()}
        M[name] = dict(per=per, f1=rep["all"]["slot_f1"], errs=errs, th=th)
    (na, A), (nb, B) = M.items()
    out = {}
    for n, m in M.items():
        allth = [x for v in m["th"].values() for x in v]
        out[n] = {"slot_f1": m["f1"], "errors": len(m["errs"]), "by_cat": dict(Counter(e["cat"] for e in m["errs"]).most_common()),
                  "by_root": dict(Counter(e["root"] for e in m["errs"]).most_common()),
                  "think_chars_avg_per_chunk": round(sum(allth) / len(allth))}
    d = {r["id"]: f1(A["per"][r["id"]]) - f1(B["per"][r["id"]]) for r in rows}
    loses = sorted((i for i in d if d[i] < -1e-9), key=d.get)
    wins = [i for i in d if d[i] > 1e-9]
    key = lambda e: (e["id"], json.dumps(e["record"], ensure_ascii=False, default=str), e["slot"], str(e["gold"]))  # noqa: E731
    bk = {key(e) for e in B["errs"]}
    only_a = [e for e in A["errs"] if key(e) not in bk]
    avg = lambda ids, m: round(sum(sum(m["th"][i]) / len(m["th"][i]) for i in ids) / len(ids)) if ids else None  # noqa: E731
    out["diff"] = {"docs": len(d), f"{na}_better": len(wins), f"{nb}_better": len(loses), "equal": len(d) - len(wins) - len(loses),
                   "mean_f1_diff": round(sum(d.values()) / len(d), 4),
                   f"{na}_only_errors": len(only_a),
                   f"{na}_only_by_cat": dict(Counter(e["cat"] for e in only_a).most_common(6)),
                   f"{na}_only_by_root": dict(Counter(e["root"] for e in only_a).most_common(6)),
                   f"{na}_only_by_slot_type": dict(Counter(e["type"] for e in only_a).most_common(6)),
                   f"{na}_only_by_lang": dict(Counter(e["lang"] for e in only_a)),
                   "worst_ids": [(i, round(d[i], 3)) for i in loses[:5]],
                   f"{na}_think_chars_on_losses": avg(loses, A), f"{na}_think_chars_on_wins": avg(wins, A),
                   f"{na}_think_chars_on_equal": avg([i for i in d if abs(d[i]) <= 1e-9], A)}
    out["examples"] = [{k: e[k] for k in ("id", "cat", "slot", "gold", "pred")} for e in
                       sorted(only_a, key=lambda e: d[e["id"]])[:5]]
    s = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    print(s)
    if a.json_out:
        open(a.json_out, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    main()
