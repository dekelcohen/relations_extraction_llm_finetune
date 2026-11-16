"""Per-slot diffs (gold vs Qwen3.8-27B-INT4 vs other models) for imperfect docs, reusing re_synth/score.py.

python outputs/audit/build_diffs.py  -> outputs/audit/diffs_{grpo_dev,pilot}.jsonl
Pred files default to /tmp/int4 (read-only inputs); override with INT4_DIR.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from re_synth.score import align, match, parse_prediction, sample_f1, score_sample  # noqa: E402

INT4 = os.environ.get("INT4_DIR", "/tmp/int4")
SETS = {
    "grpo_dev": {
        "data": "data/stage_2_RL_GRPO/grpo_dev.jsonl",
        "int4": f"{INT4}/qwen3.8-27b-int4.grpo_dev.preds.jsonl",
        "others": {"sonnet": "data/stage_2_RL_GRPO/preds_strong.jsonl", "llama70b": "data/stage_2_RL_GRPO/preds_weak.jsonl",
                   "sft4b": "outputs/eval/sft.grpo_dev.preds.jsonl", "fp8_27b": "outputs/eval/qwen3.8-27b-fp8.grpo_dev.preds.jsonl"},
        "sample": None,
    },
    "pilot": {
        "data": "data/pilot/export/pilot_all.jsonl",
        "int4": f"{INT4}/qwen3.8-27b-int4-think.preds.jsonl",
        "others": {"sonnet": "data/pilot/preds_strong.jsonl", "llama70b": "data/pilot/preds_weak.jsonl",
                   "sft4b": "outputs/eval/qwen3-4b-ft-think.preds.jsonl", "fp8_27b": "outputs/eval/qwen3.8-27b-fp8-think.preds.jsonl"},
        "sample": 25,
    },
}


def load(p):
    return {json.loads(l)["id"]: json.loads(l)["raw"] for l in open(os.path.join(ROOT, p), encoding="utf-8")}


def gold_view(recs, gold, st):
    """For each gold record index j: the aligned pred record (or None)."""
    if recs is None:
        return None
    return {j: recs[i] for i, j in align(recs, gold, st)}


def main():
    for name, cfg in SETS.items():
        rows = [json.loads(l) for l in open(os.path.join(ROOT, cfg["data"]), encoding="utf-8")]
        int4 = load(cfg["int4"])
        others = {k: load(v) for k, v in cfg["others"].items() if os.path.exists(os.path.join(ROOT, v))}
        imperfect = []
        for r in rows:
            recs, ok = parse_prediction(int4.get(r["id"]))
            m = score_sample(recs, ok, json.loads(r["gt"]), json.loads(r["decoys"]), r["text"], json.loads(r["slot_types"]))
            if ok and sample_f1(m) < 0.999:
                imperfect.append(r)
        print(name, "valid-json imperfect:", len(imperfect))
        if cfg["sample"]:
            imperfect = random.Random(0).sample(imperfect, cfg["sample"])
        out = open(os.path.join(ROOT, f"outputs/audit/diffs_{name}.jsonl"), "w", encoding="utf-8")
        for r in imperfect:
            gold = json.loads(r["gt"])["records"]
            st = json.loads(r["slot_types"])
            recs, _ = parse_prediction(int4[r["id"]])
            pv = gold_view(recs, gold, st)
            ov = {k: gold_view(parse_prediction(v.get(r["id"]))[0], gold, st) for k, v in others.items()}
            used = set(i for i, _ in align(recs, gold, st))
            for j, g in enumerate(gold):
                p = pv.get(j)
                for k, typ in st.items():
                    gv, v = g.get(k), (p or {}).get(k)
                    if gv is None and v is None:
                        continue
                    if gv is not None and v is not None and match(v, gv, typ):
                        continue
                    d = {"set": name, "id": r["id"], "rec": j, "main_entity": g.get("main_entity"),
                         "pred_main": (p or {}).get("main_entity"), "slot": k, "type": typ, "gold": gv, "pred": v,
                         "record_unaligned": p is None}
                    for on, o in ov.items():
                        d[on] = None if o is None else (o.get(j) or {}).get(k)
                    d["n_agree_pred"] = sum(1 for on, o in ov.items() if o is not None and
                                            match((o.get(j) or {}).get(k), v, typ) if v is not None)
                    out.write(json.dumps(d, ensure_ascii=False) + "\n")
            for i, p in enumerate(recs):
                if i in used:
                    continue
                d = {"set": name, "id": r["id"], "rec": None, "main_entity": None, "pred_main": p.get("main_entity"),
                     "slot": "*extra_record*", "gold": None, "pred": p, "record_unaligned": True}
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        out.close()


if __name__ == "__main__" and "--4b" not in sys.argv:
    main()


def ok(o, j, k, gv, typ):
    """Model o's aligned value for gold record j / slot k agrees with gold (both null counts as agreement)."""
    if o is None:
        return False
    v = (o.get(j) or {}).get(k)
    return (v is None and gv is None) or (v is not None and gv is not None and match(v, gv, typ))


def main_4b(cap_pilot=40):
    """Qwen3-4B SFT errors on slots where BOTH Sonnet 5.5 and Llama-70B match gold."""
    cfgs = {"grpo_dev": ("outputs/eval/sft.grpo_dev.preds.jsonl", None),
            "pilot": ("outputs/eval/qwen3-4b-ft-think.preds.jsonl", cap_pilot)}
    for name, (p4b, cap) in cfgs.items():
        cfg = SETS[name]
        rows = [json.loads(l) for l in open(os.path.join(ROOT, cfg["data"]), encoding="utf-8")]
        m4 = load(p4b)
        son, lla, int4 = load(cfg["others"]["sonnet"]), load(cfg["others"]["llama70b"]), load(cfg["int4"])
        out = []
        for r in rows:
            gold = json.loads(r["gt"])["records"]
            st = json.loads(r["slot_types"])
            recs, ok4 = parse_prediction(m4.get(r["id"]))
            if not ok4:
                continue
            v4 = gold_view(recs, gold, st)
            vs, vl = (gold_view(parse_prediction(d.get(r["id"]))[0], gold, st) for d in (son, lla))
            vi = gold_view(parse_prediction(int4.get(r["id"]))[0], gold, st)
            for j, g in enumerate(gold):
                for k, typ in st.items():
                    gv = g.get(k)
                    if ok(v4, j, k, gv, typ) or not (ok(vs, j, k, gv, typ) and ok(vl, j, k, gv, typ)):
                        continue
                    out.append({"set": name, "id": r["id"], "rec": j, "main_entity": g.get("main_entity"), "slot": k,
                                "type": typ, "gold": gv, "pred": (v4.get(j) or {}).get(k),
                                "record_unaligned": v4.get(j) is None, "sonnet": (vs.get(j) or {}).get(k),
                                "llama70b": (vl.get(j) or {}).get(k), "int4_27b": ((vi or {}).get(j) or {}).get(k)})
        print(name, "4B errors where sonnet+llama match gold:", len(out), "docs", len({d['id'] for d in out}))
        if cap and len(out) > cap:
            out = random.Random(0).sample(out, cap)
        with open(os.path.join(ROOT, f"outputs/audit/diffs4b_{name}.jsonl"), "w", encoding="utf-8") as f:
            for d in sorted(out, key=lambda d: (d["id"], d["rec"], d["slot"])):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__" and "--4b" in sys.argv:
    main_4b()
