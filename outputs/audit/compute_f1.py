"""Corrected slot-F1 after applying the audit (run after adjudicate.py). Writes outputs/audit/f1_corrected.json.

Fixes: gold patches (a/b) are applied to the gold and every model is rescored with the unchanged scorer; scorer issues (e)
are credited per audited slot (span: tp+1 fp-1 fn-1; placeholder dash for a null slot: fp-1). Valid-JSON docs only,
as in the headline numbers. Pilot audits are samples, so their deltas are also extrapolated (INT4 x73/25 imperfect docs,
4B x80/40 audited cases); grpo_dev was audited exhaustively.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from re_synth.score import parse_prediction, prf, score_sample  # noqa: E402

A = os.path.join(ROOT, "outputs/audit")
INT4_DIR = os.environ.get("INT4_DIR", "/tmp/int4")
INT4, B4 = "qwen3.8-27b-int4", "qwen3-4b-ft"
SETS = {
    "pilot": ("data/pilot/export/pilot_all.jsonl", {
        INT4: f"{INT4_DIR}/qwen3.8-27b-int4-think.preds.jsonl", B4: "outputs/eval/qwen3-4b-ft-think.preds.jsonl",
        "sonnet-5.5": "data/pilot/preds_strong.jsonl", "llama-3.3-70b": "data/pilot/preds_weak.jsonl",
        "qwen3.8-27b-fp8": "outputs/eval/qwen3.8-27b-fp8-think.preds.jsonl"}),
    "grpo_dev": ("data/stage_2_RL_GRPO/grpo_dev.jsonl", {
        INT4: f"{INT4_DIR}/qwen3.8-27b-int4.grpo_dev.preds.jsonl", B4: "outputs/eval/sft.grpo_dev.preds.jsonl",
        "sonnet-5.5": "data/stage_2_RL_GRPO/preds_strong.jsonl", "llama-3.3-70b": "data/stage_2_RL_GRPO/preds_weak.jsonl",
        "qwen3.8-27b-fp8": "outputs/eval/qwen3.8-27b-fp8.grpo_dev.preds.jsonl"}),
}
SCALE = {("pilot", INT4): 73 / 25, ("pilot", B4): 80 / 40}


def jl(p):
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def patched(rows, patches):
    out = []
    for r in rows:
        ps = [p for p in patches if p["id"] == r["id"]]
        if ps:
            r = dict(r)
            gt = json.loads(r["gt"])
            for p in ps:
                gt["records"][p["rec"]][p["slot"]] = p["new"]
            r["gt"] = json.dumps(gt, ensure_ascii=False)
        out.append(r)
    return out


def score(rows, preds):
    per = {}
    for r in rows:
        recs, ok = parse_prediction(preds.get(r["id"]))
        m = score_sample(recs, ok, json.loads(r["gt"]), json.loads(r["decoys"]), r["text"], json.loads(r["slot_types"]))
        per[r["id"]] = m
    return per


def totals(per, ids):
    tp = sum(per[i].get("tp", 0) for i in ids)
    fp = sum(per[i].get("pred_values", 0) - per[i].get("tp", 0) for i in ids)
    fn = sum(per[i].get("n_gold_values", 0) - per[i].get("tp", 0) for i in ids)
    return [tp, fp, fn]


def fmt(t):
    p, r, f = prf(*t)
    return {"tp": t[0], "fp": t[1], "fn": t[2], "P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}


def slot_delta(o):
    """(dtp, dfp, dfn) when an audited disagreement is counted as correct (e) or dropped (c)."""
    if o["slot"] == "*extra_record*":
        return (0, -sum(1 for k, v in o["pred"].items() if k != "main_entity" and v is not None), 0)
    g, p = o["gold"] is not None, o["pred"] is not None
    if o["category"] == "e":
        return (1, -1, -1) if g and p else ((0, -1, 0) if p else (0, 0, -1))
    return (0, -1 if p else 0, -1 if g else 0)


def main():
    audit = jl("outputs/audit/label_noise_audit.jsonl")
    patches = jl("outputs/audit/gold_patches.jsonl")
    res = {}
    for name, (data, models) in SETS.items():
        rows = jl(data)
        ps = [p for p in patches if p["set"] == name]
        rows_fix = patched(rows, ps)
        for model, path in models.items():
            if not os.path.exists(path if os.path.isabs(path) else os.path.join(ROOT, path)):
                continue
            preds = {r["id"]: r.get("raw") for r in jl(path)}
            per0 = score(rows, preds)
            valid = [i for i, m in per0.items() if m["valid_json"]]
            base = totals(per0, valid)
            per1 = score(rows_fix, preds)
            fixed = totals(per1, valid)
            own = [o for o in audit if o["set"] == name and o["model"] == model and o["id"] in valid]
            e = [sum(x) for x in zip((0, 0, 0), *[slot_delta(o) for o in own if o["category"] == "e"])]
            c = [sum(x) for x in zip((0, 0, 0), *[slot_delta(o) for o in own if o["category"] == "c"])]
            abe = [fixed[k] + e[k] for k in range(3)]
            abec = [abe[k] + c[k] for k in range(3)]
            r = {"n_valid_docs": len(valid), "base": fmt(base), "gold_patched_ab": fmt(fixed), "ab_e_fixed": fmt(abe),
                 "ab_e_fixed_c_dropped": fmt(abec)}
            sc = SCALE.get((name, model))
            if sc:
                own_patch = patched(rows, [p for p in ps if p["found_by"] == model])
                dpatch_own = [a - b for a, b in zip(totals(score(own_patch, preds), valid), base)]
                dpatch_other = [a - b - d for a, b, d in zip(fixed, base, dpatch_own)]
                ex = [base[k] + dpatch_other[k] + sc * (dpatch_own[k] + e[k]) for k in range(3)]
                exc = [ex[k] + sc * c[k] for k in range(3)]
                r["ab_e_fixed_extrapolated"] = fmt([round(v, 1) for v in ex])
                r["ab_e_fixed_c_dropped_extrapolated"] = fmt([round(v, 1) for v in exc])
                r["scale"] = round(sc, 3)
            res[f"{name}/{model}"] = r
            print(f"{name:9s} {model:18s} base F1 {r['base']['F1']}  a/b {r['gold_patched_ab']['F1']}  a/b/e {r['ab_e_fixed']['F1']}"
                  f"  +c {r['ab_e_fixed_c_dropped']['F1']}"
                  + (f"  extrap a/b/e {r['ab_e_fixed_extrapolated']['F1']} +c {r['ab_e_fixed_c_dropped_extrapolated']['F1']}" if sc else ""))
    json.dump(res, open(f"{A}/f1_corrected.json", "w"), indent=1)


if __name__ == "__main__":
    main()
