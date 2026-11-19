"""INSS round 2: (a) whole-sample he->fa translation, (b) long multi-chunk docs, (c) pool/split/export, (d) CoT inputs.

  python scripts/build_hard_inss2.py translate --n 40      # translate he samples (text + gold + decoys) to fa
  python scripts/build_hard_inss2.py genlong --n 30        # long he docs (2-3 consecutive chunks, 6-12 entities)
  python scripts/build_hard_inss2.py export                # pool -> train.jsonl / dev.jsonl (+ meta.jsonl, cot input)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hard_inss as B  # noqa: E402
from re_synth import prompting  # noqa: E402
from re_synth.llm import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, chat, parse_json_obj  # noqa: E402
from re_synth.schemas import SCHEMAS  # noqa: E402
from re_synth.score import parse_prediction_lenient, sample_f1, score_sample, slot_types_for  # noqa: E402

OUT = B.OUT
POOL = OUT / "pool.jsonl"
LONG_DIFF = {
    "far_header": "FAR HEADERS: a title or section heading (e.g. 'פורום ... של ארגון X' / 'סיכום חודשי – מרכז Y') states the "
                  "organizer/client/employer/location that applies to entities listed 1500+ characters further down; the "
                  "value is NOT repeated next to those entities (it must be inherited from the heading).",
    "table_head": "TABLE HEADERS: a pipe-delimited text table (| a | b |) with 6+ rows where the column header (incl. a "
                  "'previous update'/'proposed' distractor column) appears only at the top and many rows follow; some cells are empty.",
}


def pool_rows():
    return B.read(POOL)


def init_pool():
    """Seed the pool from the round-1 verified samples (split = current dev.jsonl membership)."""
    if POOL.exists():
        return
    dev = {json.loads(l)["id"] for l in open(OUT / "dev.jsonl", encoding="utf-8")}
    rows = []
    for t in ("sonnet", "sonnet2", "sonnet3", "sonnet_fa"):
        for r in B.read(OUT / f"verified_{t}.jsonl"):
            if r["qc"]["status"] == "ok":
                r["id"] = "hard-" + r["id"]
                r["split"] = "dev" if r["id"] in dev else "train"
                r["round"] = 1
                rows.append(r)
    B.write(POOL, rows)


def translate(a):
    init_pool()
    pool = pool_rows()
    have = {r.get("translated_from") for r in pool}
    cands = [r for r in pool if r["lang"] == "he" and r["id"] not in have and r["qc"]["status"] == "ok"
             and len(r["text"]) <= a.maxlen and (r.get("round") == 2) == a.long]
    random.Random(2).shuffle(cands)
    todo = cands[:a.n]
    sch = lambda r: SCHEMAS[r["schema_id"]]  # noqa: E731

    def run(r):
        payload = {"text": r["text"], "records": r["gt"]["records"], "decoys": [d["value"] for d in r["decoys"]]}
        p = ("Translate this Hebrew document into natural Persian (Farsi), together with its extraction gold, CONSISTENTLY.\n"
             "Preserve names of people/organizations/places, numbers, IDs; translate dates into Persian words/format but keep the "
             "calendar (Hebrew-calendar dates stay Hebrew-calendar, Gregorian stay Gregorian; do not convert). Keep markdown/pipe tables "
             "and line structure. Every non-null value in `records` (and every decoy) must appear VERBATIM, character for character, "
             "in the translated text; keep null as null and key names unchanged. Keep the same decoys in the same order (translated).\n"
             "Output ONLY JSON: {\"text\": ..., \"records\": [...same keys...], \"decoys\": [...translated strings...]}\n\n"
             + json.dumps(payload, ensure_ascii=False))
        o = parse_json_obj(chat([{"role": "user", "content": p}], a.model, temperature=0.3, salt="trs",
                                reasoning_effort=a.reasoning_effort)) or {}
        ds = o.get("decoys") or []
        new = {**r, "id": r["id"] + "-fa", "lang": "fa", "translated_from": r["id"], "text": o.get("text"),
               "gt": {"records": o.get("records")}, "round": 2,
               "decoys": [{**d, "value": ds[i]} for i, d in enumerate(r["decoys"]) if i < len(ds) and isinstance(ds[i], str)],
               "gen_model": a.model}
        new["qc"] = {"prog_issues": B.prog_check(new, "", maxlen=14000, min_overlap=0.0)}
        return new

    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, todo))
    out = judge(out, a)
    B.write(POOL, out, "a")
    print("translated", len(out), "ok", sum(o["qc"]["status"] == "ok" for o in out))


def judge(rows, a):
    def run(s):
        iss = s["qc"]["prog_issues"]
        iss = [i for i in iss if not i.startswith("decoys<")]
        s["qc"]["prog_issues"] = iss
        if not iss:
            sd = SCHEMAS[s["schema_id"]].to_prompt_dict()
            raw = chat(prompting.messages(sd, s["text"]), a.judge, temperature=0.0, salt="blind",
                       reasoning_effort=a.reasoning_effort)
            recs, valid, _ = parse_prediction_lenient(raw)
            m = score_sample(recs, valid, s["gt"], s["decoys"], s["text"], slot_types_for(s["schema_id"]))
            s["qc"]["blind_f1"] = round(sample_f1(m), 3)
            s["qc"]["blind_model"] = a.judge
        s["qc"]["status"] = "ok" if not iss and s["qc"].get("blind_f1", 0) >= a.min_f1 else "fail"
        return s
    with ThreadPoolExecutor(a.workers) as ex:
        return list(ex.map(run, rows))


def long_prompt(text, sid, tags):
    s = SCHEMAS[sid]
    return f"""You build HARD, realistic LONG information-extraction test documents. Host passage (Hebrew, real analytical think-tank text):
<<<
{text}
>>>

Schema (one record per distinct main entity of this type):
{json.dumps(s.to_prompt_dict(), ensure_ascii=False, indent=1)}

Task: edit the host passage into a LONG Hebrew document (4000-9000 characters) that keeps most of the original sentences verbatim and
embeds 6-12 invented but plausible records of main type "{s.main_type}" fitting the topic (security/policy: forums/meetings, agreements, appointments), spread across the
whole document (some in prose, some in a list or pipe table). Facts are fictional. Required difficulty features (all must really occur):
""" + "\n".join(f"- {(LONG_DIFF | B.DIFF)[t]}" for t in tags) + f"""

Gold rules: values copied CHARACTER-BY-CHARACTER from the final text (no attached prefixes ב/ל/מ/ה/ו/ש/כ; no normalization); a slot not stated for an
entity (or only hypothetical/cancelled/belonging to another entity) is null; inherited-from-heading values ARE stated for the entity they govern.
"main_entity" is the name exactly as written. "decoys" = tempting wrong same-type values present in the text (verbatim), with slot, role (English) and record index.
Output ONLY one JSON object: {{"text": "...", "records": [{{"main_entity": ..., {', '.join(f'"{x.name}": ...' for x in s.slots)}}}], "decoys": [{{"slot": ..., "value": ..., "role": ..., "record": <idx>}}], "features": {{"<tag>": "<where>"}}}}"""


def genlong(a):
    init_pool()
    chunks = B.read(B.RAW / "chunks.jsonl")
    by = {}
    for c in chunks:
        by.setdefault(c["url"], []).append(c)
    rng = random.Random(6)
    used = {r.get("src_id") for r in pool_rows()}
    jobs = []
    urls = sorted(by)
    rng.shuffle(urls)
    i = 0
    while len(jobs) < a.n and i < 4 * len(urls):
        cs = sorted(by[urls[i % len(urls)]], key=lambda c: c["k"])
        i += 1
        if len(cs) < 2:
            continue
        st = rng.randrange(0, len(cs) - 1)
        grp = cs[st:st + rng.choice([2, 3])]
        sid = "|".join(c["id"] for c in grp)
        if any(c["id"] in used for c in grp[:1]) or sid in {j[0] for j in jobs}:
            continue
        # audit-driven (outputs/audit/27b_vs_strong): H9 nulls, H20 long multi-entity, H12 status change, H2 far values, H1/H11, H13/H18 tables
        tags = ["far_header", "null_tempting", rng.choice(["table_head", "table_prose"]),
                rng.choice(["status_change", "status_change", "near_miss", "coref"])]
        jobs.append((sid, grp, SCHEMA_ORDER[len(jobs) % 3], tags))

    def run(j):
        sid, grp, schema, tags = j
        text = "\n\n".join(c["text"] for c in grp)
        o = parse_json_obj(chat([{"role": "user", "content": long_prompt(text, schema, tags)}], a.model, temperature=0.8,
                                salt="long", reasoning_effort=a.reasoning_effort)) or {}
        s = {"id": "hardL-" + sid.replace("inss-", "").replace("|", "+"), "src_id": grp[0]["id"], "url": grp[0]["url"], "lang": "he",
             "schema_id": schema, "tags": tags, "gen_model": a.model, "text": o.get("text"),
             "gt": {"records": o.get("records")}, "decoys": [d for d in (o.get("decoys") or []) if isinstance(d, dict) and d.get("value")],
             "features": o.get("features"), "round": 2}
        s["qc"] = {"prog_issues": B.prog_check(s, text, maxlen=14000, min_overlap=0.3, min_values=10)}
        return s
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, jobs))
    out = judge(out, a)
    for s in out:
        s["split"] = "dev" if int(hashlib.md5(s["id"].encode()).hexdigest()[:6], 16) % 100 < 20 else "train"
    B.write(OUT / "cand_long.jsonl", out)
    B.write(POOL, out, "a")
    print("long", len(out), "ok", sum(o["qc"]["status"] == "ok" for o in out))


SCHEMA_ORDER = ["meeting", "contract", "person_employment"]


def export(a):
    from re_synth.export import export_eval
    rows = [r for r in pool_rows() if r["qc"]["status"] == "ok"]
    by = {r["id"]: r for r in rows}
    for r in rows:  # translations inherit the split of their Hebrew source
        if r.get("translated_from") in by:
            r["split"] = by[r["translated_from"]]["split"]
        r.setdefault("split", "train")
    samples = []
    for r in rows:
        dec = []
        names = {x.name: x.type for x in SCHEMAS[r["schema_id"]].slots}
        for d in r["decoys"]:
            dec.append({"slot": d.get("slot"), "type": names.get(d.get("slot"), "text"), "value": d["value"],
                        "role": d.get("role", ""), "record": d.get("record", 0), "placement": "near"})
        samples.append({**r, "format": "text", "hard_cases": ["INSS"] + r["tags"], "decoys": dec})
    for sp in ("train", "dev"):
        sub = [s for s in samples if s["split"] == sp]
        n = export_eval(sub, OUT / f"{sp}.jsonl")
        print(sp, n, {l: sum(s["lang"] == l for s in sub) for l in ("he", "fa")})
        if sp == "train":
            B.write(OUT / "cot_input.jsonl", [{k: s[k] for k in ("id", "lang", "format", "schema_id", "hard_cases", "text", "gt", "decoys")} for s in sub])
    B.write(OUT / "meta.jsonl", [{"id": s["id"], "split": s["split"], "src_chunk": s.get("translated_from") or s["src_id"], "url": s["url"],
                                  "lang": s["lang"], "schema_id": s["schema_id"], "tags": s["tags"], "gen_model": s["gen_model"],
                                  "blind_f1": s["qc"].get("blind_f1"), "round": s.get("round", 1), "n_chars": len(s["text"])} for s in samples])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("translate", "genlong", "export"):
        p = sp.add_parser(name)
        p.add_argument("--n", type=int, default=30)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--judge", default="copilot-gpt-6.1-terra")
        p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--min-f1", type=float, default=0.7, dest="min_f1")
        p.add_argument("--maxlen", type=int, default=3600)
        p.add_argument("--long", action="store_true", help="translate round-2 long docs instead of round-1 short ones")
    a = ap.parse_args()
    {"translate": translate, "genlong": genlong, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
