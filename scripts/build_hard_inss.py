"""Hard INSS-hosted extraction samples (he + fa) built on the re_synth pipeline pieces.

Stages (all resumable, outputs under data/raw/inss/ (git-ignored) and data/hard_inss/):
  chunk      articles.jsonl -> chunks.jsonl (re_synth.chunking, CHUNK_CHARS=2500; keeps source URL + offset)
  translate  ~35% of chunks -> Persian (chunks_fa.jsonl, mapping to the Hebrew chunk id)
  gen        chunk -> hard sample candidate (LLM rewrite+inject, gold JSON, decoys), --model copilot-<id>
  verify     programmatic checks + cross-model blind extraction (slot-F1 vs gold via re_synth.score)
  export     verified samples -> data/hard_inss/{train,dev}.jsonl (same row format as grpo_dev / export_eval)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from re_synth import chunking, prompting  # noqa: E402
from re_synth.export import export_eval  # noqa: E402
from re_synth.inject import script_ratio  # noqa: E402
from re_synth.llm import chat, parse_json_obj  # noqa: E402
from re_synth.schemas import SCHEMAS  # noqa: E402
from re_synth.score import norm, parse_prediction_lenient, sample_f1, score_sample, slot_types_for  # noqa: E402

RAW = Path("data/raw/inss")
OUT = Path("data/hard_inss")
LANG = {"he": "Hebrew", "fa": "Persian (Farsi)"}
SCHEMA_IDS = ["meeting", "contract", "person_employment"]
DIFF = {
    "near_miss": "NEAR-MISS DISTRACTORS: for at least two slots include a same-type value that does NOT match the slot "
                 "definition (e.g. a date of a previous/earlier/proposed round vs the actual date, a proposed or cancelled "
                 "event vs the one that really took place, a former role/org vs the current one).",
    "interleaved": "INTERLEAVED ENTITIES: 2-3 records of the same type whose slot values are mentioned alternately, mixed "
                   "within the same sentences/paragraph (not one record per paragraph).",
    "coref": "COREFERENCE: at least one slot value is stated only via a pronoun / description / alias in a later sentence "
             "(e.g. 'the same forum', 'the latter', 'the agreement' ), not next to the entity's name.",
    "hebcal": "CALENDARS: at least two dates written in the Hebrew calendar with geresh/gershayim (e.g. י\"ט בסיוון תשפ\"ו) "
              "mixed with Gregorian dates (for fa: use Jalali dates mixed with Gregorian). One of them is a distractor.",
    "null_tempting": "TEMPTING NULLS: at least one slot of one record must be null (not stated for it) although a "
                     "same-type value that belongs to ANOTHER entity or to a different role sits right nearby.",
    "table_prose": "TABLE+PROSE: one entity is introduced in prose, and some of its slot values appear only in a small "
                   "pipe-delimited text table (| a | b |) later in the text, with a distractor column "
                   "(e.g. 'previous update').",
    "status_change": "STATUS CHANGES: for at least two entities the status/date changed over the text (earlier status/date, later update); "
                     "the gold is the LATEST one, the earlier ones are distractors.",
    "negation": "NEGATION/HYPOTHETICALS: include a postponed/cancelled/denied/'might be held'/'if signed' statement whose "
                "value must NOT be extracted (the slot is null or takes the confirmed value).",
}
TAGS = list(DIFF)


def read(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8")] if Path(p).exists() else []


def write(p, rows, mode="w"):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cid(url, k):
    return f"inss-{hashlib.md5(url.encode()).hexdigest()[:6]}-{k:02d}"


def stage_chunk(a):
    rows = []
    for art in read(RAW / "articles.jsonl"):
        parts, cur = [], ""
        for para in re.split(r"(?<=[.!?:])\s+", art["text"]):  # sentence packing up to chunking.CHUNK_CHARS
            if cur and len(cur) + len(para) > chunking.CHUNK_CHARS:
                parts.append(cur.strip())
                cur = ""
            cur += para + " "
        parts.append(cur.strip())
        for k, c in enumerate(parts):
            if 1200 <= len(c) <= 2700:
                rows.append({"id": cid(art["url"], k), "url": art["url"], "title": art["title"], "k": k, "text": c})
    write(RAW / "chunks.jsonl", rows)
    print(len(rows), "chunks")


def _llm(prompt, model, salt, effort="low"):
    return chat([{"role": "user", "content": prompt}], model, temperature=0.7, salt=salt, reasoning_effort=effort)


def stage_translate(a):
    rng = random.Random(1)
    chunks = [c for c in read(RAW / "chunks.jsonl")]
    rng.shuffle(chunks)
    pick = chunks[:a.n]
    done = {r["id"] for r in read(RAW / "chunks_fa.jsonl")}

    def tr(c):
        p = ("Translate the following Hebrew analytical text into natural, fluent Persian (Farsi). Preserve names of "
             "people, organizations, places, all dates and numbers exactly (transliterate names; keep Hebrew-calendar "
             "dates as they are written in Persian words, do not convert calendars). Output only the translation.\n\n"
             + c["text"])
        return {**c, "id": c["id"] + "-fa", "src_id": c["id"], "text": _llm(p, a.model, "tr").strip()}
    todo = [c for c in pick if c["id"] + "-fa" not in done]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(tr, todo))
    write(RAW / "chunks_fa.jsonl", out, "a")
    print("translated", len(out))


def gen_prompt(chunk, lang, schema_id, tags):
    s = SCHEMAS[schema_id]
    sd = s.to_prompt_dict()
    return f"""You build HARD, realistic information-extraction test documents. Source passage ({LANG[lang]}, a real analytical think-tank text):
<<<
{chunk['text']}
>>>

Schema (extract one record per distinct main entity of this type):
{json.dumps(sd, ensure_ascii=False, indent=1)}

Task: lightly edit the passage into a document in {LANG[lang]} that keeps at least 70% of the original sentences verbatim, and weave in
2-3 invented but plausible records of main type "{s.main_type}" that fit the passage's topic (security/policy: meetings/forums/summits, agreements/deals, appointments).
Use fictional facts consistent with the topic; do not state invented facts about real named people as if real quotes. You may add 3-8 sentences (and a
small pipe table if required) and may modify sentences. The final document must be 1200-3200 characters.
Required difficulty features (all must really occur):
""" + "\n".join(f"- {DIFF[t]}" for t in tags) + f"""

Rules for the gold annotation:
- Gold values are copied CHARACTER-BY-CHARACTER from the final text (no grammatical prefixes like ב/ל/מ/ה/ו/ש/כ attached; same calendar/spelling, no normalization).
- A slot not stated for an entity (or only stated as hypothetical/cancelled/other entity's) is null. Follow each slot definition strictly.
- "main_entity" is the entity's name/description exactly as written in the text.
- "decoys" lists the tempting wrong same-type values present in the text (verbatim) with the slot they tempt and a short role (English).
Output ONLY one JSON object: {{"text": "<final document>", "records": [{{"main_entity": ..., {', '.join(f'"{x.name}": ...' for x in s.slots)}}}], "decoys": [{{"slot": ..., "value": ..., "role": ..., "record": <index>}}], "features": {{"<tag>": "<one-line explanation of where it occurs>"}}}}"""


def stage_gen(a):
    chunks = read(RAW / "chunks.jsonl") + read(RAW / "chunks_fa.jsonl")
    for c in chunks:
        c["lang"] = "fa" if c["id"].endswith("-fa") else "he"
    rng = random.Random(a.seed)
    rng.shuffle(chunks)
    outp = OUT / f"cand_{a.tag}.jsonl"
    done = {r["id"] for r in read(outp)}
    if a.lang:  # only chunks no earlier run has used
        used = {r["id"] for f in OUT.glob("cand_*.jsonl") for r in read(f)}
        chunks = [c for c in chunks if c["lang"] == a.lang and c["id"] not in used]
    jobs = []
    for i, c in enumerate(chunks[a.offset:a.offset + a.n]):
        r = random.Random(f"{c['id']}-{a.seed}")
        tags = r.sample(TAGS, 3)
        sid = SCHEMA_IDS[(i + a.offset) % len(SCHEMA_IDS)]
        jobs.append((f"{c['id']}", c, sid, tags))

    def run(j):
        sid_, c, sid, tags = j
        raw = chat([{"role": "user", "content": gen_prompt(c, c["lang"], sid, tags)}], a.model, temperature=0.8,
                   salt=a.tag, reasoning_effort="low")
        o = parse_json_obj(raw) or {}
        return {"id": sid_, "src_id": c.get("src_id", c["id"].removesuffix("-fa")), "url": c["url"], "lang": c["lang"],
                "schema_id": sid, "tags": tags, "gen_model": a.model, "text": o.get("text"),
                "gt": {"records": o.get("records")}, "decoys": o.get("decoys", []), "features": o.get("features")}
    todo = [j for j in jobs if j[0] not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, todo))
    write(outp, out, "a")
    print("generated", len(out))


def prog_check(s, source_text, maxlen=3600, min_overlap=0.45, min_values=4):
    """Programmatic verifier: gold verbatim in text, shapes, nulls, language, length, host overlap."""
    iss = []
    t = s.get("text")
    recs = (s.get("gt") or {}).get("records")
    if not t or not isinstance(recs, list) or not recs:
        return ["no_json"]
    sch = SCHEMAS[s["schema_id"]]
    names = ["main_entity"] + [x.name for x in sch.slots]
    nt = norm(t)
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            iss.append("bad_record")
            continue
        if set(r) != set(names):
            iss.append(f"r{i}_keys")
        for k in names:
            v = r.get(k)
            if v is None:
                continue
            if not isinstance(v, str) or (v not in t and norm(v) not in nt):
                iss.append(f"r{i}.{k}_not_in_text")
    if len({norm(r.get("main_entity")) for r in recs if isinstance(r, dict)}) < len(recs):
        iss.append("dup_main")
    if not any(isinstance(r, dict) and any(r.get(x.name) is None for x in sch.slots) for r in recs):
        iss.append("no_null")
    if sum(1 for r in recs for x in sch.slots if isinstance(r, dict) and r.get(x.name)) < min_values:
        iss.append("too_few_values")
    if not 900 <= len(t) <= maxlen:
        iss.append(f"length:{len(t)}")
    if script_ratio(t, s["lang"]) < 0.6:
        iss.append("script_ratio")
    ds = [d for d in s.get("decoys") or [] if isinstance(d, dict) and d.get("value")]
    ok_d = [d for d in ds if d["value"] in t or norm(d["value"]) in nt]
    if len(ok_d) < 2:
        iss.append("decoys<2")
    for d in ok_d:  # a decoy equal to a gold value of its own record is not a distractor
        rr = d.get("record")
        if isinstance(rr, int) and 0 <= rr < len(recs) and isinstance(recs[rr], dict) \
                and norm(recs[rr].get(d.get("slot"))) == norm(d["value"]):
            iss.append("decoy_is_gold")
    sm = difflib.SequenceMatcher(None, source_text[:2600], t[:2600], autojunk=False)
    ov = sum(b.size for b in sm.get_matching_blocks()) / max(1, min(len(source_text[:2600]), len(t[:2600])))
    if ov < min_overlap:
        iss.append(f"host_overlap:{ov:.2f}")
    s["host_overlap"] = round(ov, 2)
    return sorted(set(iss))


def stage_verify(a):
    srcs = {c["id"]: c["text"] for c in read(RAW / "chunks.jsonl") + read(RAW / "chunks_fa.jsonl")}
    inp, outp = OUT / f"cand_{a.tag}.jsonl", OUT / f"verified_{a.tag}.jsonl"
    cands = read(inp)
    done = {r["id"] for r in read(outp)}
    todo = [c for c in cands if c["id"] not in done]

    def run(s):
        s["decoys"] = [d for d in (s.get("decoys") or []) if isinstance(d, dict) and d.get("value")]
        iss = prog_check(s, srcs.get(s["id"], ""))
        s["qc"] = {"prog_issues": iss}
        if not iss:
            sd = SCHEMAS[s["schema_id"]].to_prompt_dict()
            raw = chat(prompting.messages(sd, s["text"]), a.judge, temperature=0.0, salt="blind", reasoning_effort="low")
            recs, valid, _ = parse_prediction_lenient(raw)
            m = score_sample(recs, valid, s["gt"], s["decoys"], s["text"], slot_types_for(s["schema_id"]))
            s["qc"]["blind_f1"] = round(sample_f1(m), 3)
            s["qc"]["blind_model"] = a.judge
            s["qc"]["blind_pred"] = recs
        s["qc"]["status"] = ("ok" if not iss and s["qc"].get("blind_f1", 0) >= a.min_f1 else "fail")
        return s
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(run, todo))
    write(outp, out, "a")
    allv = read(outp)
    n_prog = sum(not r["qc"]["prog_issues"] for r in allv)
    n_ok = sum(r["qc"]["status"] == "ok" for r in allv)
    print(f"{a.tag}: n={len(allv)} prog_pass={n_prog} final_ok={n_ok}")


def stage_export(a):
    rows = []
    for t in a.tags:
        rows += [r for r in read(OUT / f"verified_{t}.jsonl") if r["qc"]["status"] == "ok"]
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    samples = []
    for r in uniq:
        samples.append({**r, "id": "hard-" + r["id"], "format": "text", "hard_cases": ["INSS"] + r["tags"],
                        "decoys": [{"slot": d.get("slot"), "type": SCHEMAS[r["schema_id"]].slot(d["slot"]).type
                                    if d.get("slot") in {x.name for x in SCHEMAS[r["schema_id"]].slots} else "text",
                                    "value": d["value"], "role": d.get("role", ""), "record": d.get("record", 0),
                                    "placement": "near"} for d in r["decoys"]]})
    dev_ids = set()  # stratified by language: every ~(100/dev_pct)-th sample by hash order
    for lg in ("he", "fa"):
        sub = sorted((s for s in samples if s["lang"] == lg), key=lambda s: hashlib.md5(s["id"].encode()).hexdigest())
        dev_ids |= {s["id"] for s in sub[:max(1, round(len(sub) * a.dev_pct / 100))]}
    for name, sub in (("train", [s for s in samples if s["id"] not in dev_ids]), ("dev", [s for s in samples if s["id"] in dev_ids])):
        n = export_eval(sub, OUT / f"{name}.jsonl")
        print(name, n)
    write(OUT / "meta.jsonl", [{"id": s["id"], "src_chunk": s["src_id"], "url": s["url"], "lang": s["lang"],
                                "schema_id": s["schema_id"], "tags": s["tags"], "gen_model": s["gen_model"],
                                "features": s["features"], "blind_f1": s["qc"].get("blind_f1")} for s in samples])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("chunk")
    t = sp.add_parser("translate"); t.add_argument("--n", type=int, default=40); t.add_argument("--model", default="copilot-claude-sonnet-5.5")
    g = sp.add_parser("gen"); g.add_argument("--model", required=True); g.add_argument("--tag", required=True)
    g.add_argument("--n", type=int, default=10); g.add_argument("--offset", type=int, default=0)
    g.add_argument("--seed", type=int, default=0); g.add_argument("--lang", choices=["he", "fa"]); g.add_argument("--workers", type=int, default=3)
    v = sp.add_parser("verify"); v.add_argument("--tag", required=True); v.add_argument("--judge", required=True)
    v.add_argument("--min-f1", type=float, default=0.7, dest="min_f1"); v.add_argument("--workers", type=int, default=3)
    e = sp.add_parser("export"); e.add_argument("--tags", nargs="+", required=True); e.add_argument("--dev-pct", type=int, default=20, dest="dev_pct")
    a = ap.parse_args()
    {"chunk": stage_chunk, "translate": stage_translate, "gen": stage_gen, "verify": stage_verify, "export": stage_export}[a.cmd](a)


if __name__ == "__main__":
    main()
