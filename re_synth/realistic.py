"""H20 "realistic long document": 4-7 sections mixing LLM-written prose (entities spread through the narrative,
gold grounded per section by inject.verify_and_ground) with deterministic structured sections (table / list / cards,
gold = plan). ~20% of the entities are *cross-section*: introduced in prose with some slots, repeated in a later
table/list with the remaining slots (or with an updated status) -> ONE merged gold record."""
from __future__ import annotations

import html
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import layout
from .hardcases import DECOY_ROLES, LONG_KINDS, _gen_decoy, _plan_section
from .inject import LANG_NAME, MARK, SYSTEM, build_prompt, dump, parse_title_body, script_ratio, strip_markers, \
    verify_and_ground
from .llm import chat
from .schemas import SCHEMAS, gen_main, gen_value
from .score import norm
from .source import load_seeds

log = logging.getLogger(__name__)
GENRES = ["annual report", "quarterly operations report", "tender / procurement committee minutes",
          "investigation report", "audit report", "municipal council protocol", "news round-up feature",
          "internal memo with appendices", "court / regulatory decision"]
FORMAT_W = {"html": 0.5, "text": 0.25, "email": 0.25}
PROSE_CASES = ("H1", "H2", "H4", "H11", "H12")
MAX_TRIES = 3  # LLM attempts per prose section / frame


def _record(schema, rng, lang, name, null_p) -> dict:
    slots = {s.name: None if s.name != schema.id_slot and rng.random() < null_p else gen_value(schema, s, rng, lang)
             for s in schema.slots}
    return {"main_entity": name, "slots": slots}


def _val(rec: dict, k: str):
    return (rec["slots"].get(k) or {}).get("value")


def plan_realistic(rng: random.Random, lang: str, fmt: str, schema_id: str | None = None) -> dict:
    schema = SCHEMAS[schema_id or rng.choice([k for k, v in SCHEMAS.items() if v.id_slot])]
    n_sec = rng.randint(4, 7)
    n_struct = rng.choice([1, 2])
    n_prose = n_sec - n_struct
    n_total = max(rng.randint(10, 20), 2 * n_prose + 3 * n_struct)
    sizes = [2] * n_prose
    for _ in range(rng.randint(n_prose, 2 * n_prose)):  # 2-4 entities per prose section within the budget
        i = rng.randrange(n_prose)
        if sizes[i] < 4 and sum(sizes) < n_total - 3 * n_struct:
            sizes[i] += 1
    struct_n = [3] * n_struct
    for _ in range(n_total - sum(sizes) - sum(struct_n)):
        struct_n[rng.randrange(n_struct)] += 1
    # section order: prose first; email keeps the structured sections last (appendix)
    kinds = ["prose"] * (n_prose - 1) + ["struct"] * n_struct
    if fmt != "email":
        rng.shuffle(kinds)
    kinds = ["prose"] + kinds
    names: list[str] = []
    while len(names) < n_total:
        m = gen_main(schema, rng, lang)
        if m not in names:
            names.append(m)
    null_p = rng.choice([0.0, 0.1, 0.2])
    records = [_record(schema, rng, lang, m, null_p) for m in names]
    ids = [_val(r, schema.id_slot) for r in records]
    while len(set(ids)) < len(ids):  # identifiers are unique (they tell same-name items apart)
        for i, v in enumerate(ids):
            if ids.index(v) != i:
                records[i]["slots"][schema.id_slot] = gen_value(schema, schema.slot(schema.id_slot), rng, lang)
        ids = [_val(r, schema.id_slot) for r in records]
    # entity -> section
    pool, secs, cur = list(range(n_total)), [], 0
    pi = si = 0
    for kd in kinds:
        n = sizes[pi] if kd == "prose" else struct_n[si]
        pi, si = pi + (kd == "prose"), si + (kd == "struct")
        secs.append({"kind": kd, "ents": pool[cur:cur + n]})
        cur += n
    # cross-section entities: a prose entity re-appears in a LATER structured section
    prose_ents = [(i, e) for i, s in enumerate(secs) if s["kind"] == "prose" for e in s["ents"]]
    n_cross = min(max(1, round(0.2 * n_total)), len(prose_ents))
    cross, taken = [], set()
    struct_pos = [i for i, s in enumerate(secs) if s["kind"] == "struct"]
    nonid = [s.name for s in schema.slots if s.name != schema.id_slot]
    stn = next((s.name for s in schema.slots if s.type == "status"), None)
    has_status = stn is not None
    for _ in range(n_cross * 4):
        if len(cross) >= n_cross:
            break
        q = rng.choice(struct_pos)
        cand = [(i, e) for i, e in prose_ents if i < q and e not in taken]
        if not cand:
            continue
        ps, e = rng.choice(cand)
        taken.add(e)
        r = records[e]
        for k in nonid:  # a cross entity has all slots (no H9 nulls)
            if r["slots"][k] is None:
                r["slots"][k] = gen_value(schema, schema.slot(k), rng, lang)
        stated = rng.sample(nonid, k=rng.randint(1, len(nonid) - 1))
        c = {"ent": e, "prose": ps, "struct": q, "stated": stated, "variant": "complete", "short": None, "old_status": None}
        if has_status and rng.random() < 0.3:  # status update: prose states an older status, the table the current one
            old = r["slots"][stn]
            r["slots"][stn] = gen_value(schema, schema.slot(stn), rng, lang)
            while r["slots"][stn]["value"] == old["value"]:
                r["slots"][stn] = gen_value(schema, schema.slot(stn), rng, lang)
            c.update(variant="update", old_status=old["value"])
            if stn not in stated:
                stated.append(stn)
        sur = r["main_entity"].split()[-1]
        if schema.id == "person_employment" and len(r["main_entity"].split()) >= 2 and rng.random() < 0.5 and \
                not any(sur in x.split() for j, x in enumerate(names) if j != e):
            c["short"] = sur  # surname only (full name occurred earlier in the prose); never ambiguous
        cross.append(c)
    by_ent = {c["ent"]: c for c in cross}
    cases: set = {"H20"}
    decoys: list[dict] = []
    all_vals = {v["value"] for r in records for v in r["slots"].values() if v}
    # structured sections
    for i, s in enumerate(secs):
        if s["kind"] != "struct":
            continue
        rows, mine_c = [], [c for c in cross if c["struct"] == i]
        for e in s["ents"] + [c["ent"] for c in mine_c]:
            if e in by_ent and by_ent[e]["struct"] == i:
                c, r = by_ent[e], records[e]
                keep = {schema.id_slot} | (set(nonid) - set(c["stated"])) | ({stn} if c["variant"] == "update" else set())
                rows.append({"main_entity": c["short"] or r["main_entity"],
                             "slots": {k: (v if k in keep else None) for k, v in r["slots"].items()}})
            else:
                rows.append(records[e])
        kind = rng.choice([k for k in LONG_KINDS if not (mine_c and k == "list_org")])
        sec, c_, d_ = _plan_section(schema, rng, lang, fmt, rows, 0, len(rows), kind, set(), 0, safe=bool(mine_c))
        ents = s["ents"] + [c["ent"] for c in mine_c]
        cases |= c_
        s.update(sec=sec, rows=rows, decoys=d_, update=any(c["variant"] == "update" for c in mine_c))
        s["ents"] = ents
        decoys += [{k: v for k, v in d.items() if k != "col"} | {"record": ents[d["record"]]} for d in d_]
    # prose sections: 1-2 text hard cases on the first non-cross entity
    for i, s in enumerate(secs):
        if s["kind"] != "prose":
            continue
        order = sorted(s["ents"], key=lambda e: e in by_ent)
        s["ents"] = order
        recs = [records[e] for e in order]
        sc: list[str] = []
        free = [e for e in order if e not in by_ent]
        if free:
            sc = rng.sample(PROSE_CASES, k=rng.choice([1, 2]))
            if "H4" in sc and len(free) < 2:
                sc.remove("H4")
            if "H12" in sc and not has_status:
                sc.remove("H12")
        r0 = records[free[0]] if free else None
        if "H4" in sc:
            r1 = records[free[1]]
            cand = [k for k in nonid if schema.slot(k).type in ("date", "location") and r0["slots"].get(k)]
            if cand:
                r1["slots"][rng.choice(cand)] = None
        if "H12" in sc and r0["slots"].get(stn) is None:
            r0["slots"][stn] = gen_value(schema, schema.slot(stn), rng, lang)
        ldec = []
        if r0 and any(c in sc for c in ("H1", "H2", "H12")):
            cand_slots = [x for x in schema.slots if x.name != schema.id_slot or "H1" in sc]
            for x in rng.sample(cand_slots, k=min(rng.randint(1, 2), len(cand_slots))):
                if x.type == "status" and "H12" not in sc:
                    continue
                ldec.append({"slot": x.name, "type": x.type, "value": _gen_decoy(schema, x, rng, lang, all_vals),
                             "role": rng.choice(DECOY_ROLES[x.type]), "record": 0,
                             "placement": "between" if "H2" in sc and r0["slots"].get(x.name) else "near"})
            if "H12" in sc and not any(d["slot"] == stn for d in ldec):
                ldec.append({"slot": stn, "type": "status", "role": DECOY_ROLES["status"][0], "record": 0,
                             "value": _gen_decoy(schema, schema.slot(stn), rng, lang, {_val(r0, stn)}),
                             "placement": "near"})
        view = []
        for e in order:
            r = records[e]
            if e in by_ent:
                c = by_ent[e]
                keep = {schema.id_slot} | set(c["stated"])
                sl = {k: (v if k in keep else None) for k, v in r["slots"].items()}
                if c["variant"] == "update":
                    sl[stn] = {"value": c["old_status"]}
                view.append({"main_entity": r["main_entity"], "slots": sl})
            else:
                view.append({"main_entity": r["main_entity"], "slots": dict(r["slots"])})
        s.update(view=view, cases=sorted(sc), decoys=ldec,
                 later=any(e in by_ent for e in order))
        cases.update(sc)
        decoys += [{**d, "record": order[d["record"]]} for d in ldec]
    if any(v is None for r in records for v in r["slots"].values()):
        cases.add("H9")
    return {"lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": sorted(cases), "records": records,
            "decoys": decoys, "realistic": {"sections": secs, "cross": cross}}


FRAME_SYSTEM = SYSTEM.replace("<title>/<body>", "requested tags")


def frame_prompt(plan: dict, seed: dict, genre: str) -> str:
    schema = SCHEMAS[plan["schema_id"]]
    lang = LANG_NAME[plan["lang"]]
    secs = plan["realistic"]["sections"]
    lines = []
    for i, s in enumerate(secs, 1):
        if s["kind"] == "prose":
            lines.append(f"{i}. narrative section (its heading is written elsewhere)")
        else:
            what = {"table": "a table", "list": "a bulleted list", "list_org": "a list under an organization sub-heading",
                    "cards": "one small card per item"}[s["sec"]["kind"]]
            upd = " It contains the LATEST updated status of some items, superseding what was said earlier." \
                if s.get("update") else ""
            lines.append(f"{i}. structured section: {what} of {layout.PLURAL[schema.id]['en']} "
                         f"({', '.join(x.name for x in schema.slots)}).{upd}")
    structured = [i for i, s in enumerate(secs, 1) if s["kind"] == "struct"]
    tags = "".join(f"<s{i}>HEADING</s{i}>\n<l{i}>ONE-SENTENCE LEAD-IN</l{i}>\n" for i in structured)
    return (f"We are writing a long {genre} in {lang} "
            f"about {layout.PLURAL[schema.id]['en']}. Sections in order:\n" + "\n".join(lines) +
            "\n\nWrite ONLY the framing text, in " + lang + ", genre-appropriate: a document title, for each structured "
            "section a short heading and a one-sentence lead-in that introduces the table/list, and optionally a short "
            "generic footnote. Never mention any specific item, name, ID, organization, place or date.\n\n"
            f"Return exactly:\n<title>TITLE</title>\n{tags}<foot>OPTIONAL FOOTNOTE or empty</foot>")


def parse_frame(raw: str, plan: dict) -> dict | None:
    if not isinstance(raw, str):
        return None
    g = lambda t: (m.group(1).strip() if (m := re.search(rf"<{t}>(.*?)</{t}>", raw, re.S)) else "")
    out = {"title": g("title"), "foot": g("foot")}
    for i, s in enumerate(plan["realistic"]["sections"], 1):
        if s["kind"] == "struct":
            out[f"s{i}"], out[f"l{i}"] = g(f"s{i}"), g(f"l{i}")
            if not out[f"s{i}"]:
                return None
    return out if out["title"] else None


def _heading(style: str, k: int, h: str) -> str:
    return {"num": f"{k}. {h}", "md": f"## {h}", "par": f"§{k} {h}"}[style]


def generate_realistic(plan: dict, seed: dict, rng: random.Random, model: str = "gen", sid: str = "") -> dict:
    schema = SCHEMAS[plan["schema_id"]]
    lang, fmt = plan["lang"], plan["format"]
    secs, cross = plan["realistic"]["sections"], plan["realistic"]["cross"]
    genre = plan.get("genre") or rng.choice(GENRES)
    sample = {"id": sid, "lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": plan["hard_cases"],
              "seed": {k: seed.get(k) for k in ("source", "dataset", "title", "url")}, "genre": genre,
              "gen": {"model": model}, "plan": plan}
    issues: list[str] = []
    seed_short = {**seed, "text": seed["text"][:1500]}

    def llm(prompt, salt, system=SYSTEM):
        return chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}], model,
                    temperature=0.8, max_tokens=12000, salt=salt, reasoning_effort="low")

    frame = None
    for t in range(MAX_TRIES):
        frame = parse_frame(llm(frame_prompt(plan, seed_short, genre), f"{sid}-frame-{t}", FRAME_SYSTEM), plan)
        if frame:
            break
    if not frame:
        sample["qc"] = {"status": "fail", "issues": ["frame_missing"]}
        return sample

    gold = [{"main_entity": r["main_entity"], **{k: (v["value"] if v else None) for k, v in r["slots"].items()}}
            for r in plan["records"]]
    by_ent = {c["ent"]: c for c in cross}
    nonid = [s.name for s in schema.slots if s.name != schema.id_slot]
    decoys: list[dict] = []
    parts: list[tuple[str, str, list[str] | str]] = []  # (kind, heading, paragraphs | block)
    stn = next((x.name for x in schema.slots if x.type == "status"), None)
    for i, s in enumerate(secs, 1):
        if s["kind"] == "prose":
            sub = {"lang": lang, "schema_id": schema.id, "records": s["view"], "decoys": s["decoys"],
                   "hard_cases": s["cases"], "format": fmt}
            for t in range(MAX_TRIES):
                prompt = build_prompt(sub, seed_short, genre, False, {"k": i, "n": len(secs), "later": s["later"]})
                title, body = parse_title_body(llm(prompt, f"{sid}-sec{i}-{t}"))
                gt, iss, kept = verify_and_ground(sub, title, body) if body else (None, ["no_body"], [])
                if not iss:
                    break
            if iss:
                issues += [f"sec{i}:{x}" for x in iss]
                continue
            for e, g in zip(s["ents"], gt["records"]):
                gold[e]["main_entity"] = g["main_entity"]
                for k in nonid + [schema.id_slot]:
                    if e in by_ent and (k not in by_ent[e]["stated"] and k != schema.id_slot
                                        or k == stn and by_ent[e]["variant"] == "update"):
                        continue  # filled from the structured section
                    gold[e][k] = g.get(k)
                if e in by_ent and by_ent[e]["variant"] == "update" and g.get(stn):
                    decoys.append({"slot": stn, "type": "status", "value": g[stn], "record": e, "placement": "near",
                                   "role": "an earlier status, superseded by the later status table"})
            decoys += [{**d, "record": s["ents"][d["record"]]} for d in kept]
            parts.append(("prose", strip_markers(title)[0].strip(), [p.strip() for p in strip_markers(body)[0].split("\n\n")
                                                                       if p.strip()]))
        else:
            as_text = fmt != "html"
            mini = {"records": s["rows"], "decoys": s["decoys"],
                    "long": {"sections": [{**s["sec"], "start": 0, "end": len(s["rows"])}]}}
            block = layout.long_doc(schema, mini, lang, rng, as_text)
            if not s["sec"]["org"]:
                block = block.split("\n", 1)[1]
            parts.append(("struct", frame[f"s{i}"], [frame[f"l{i}"], block] if frame[f"l{i}"] else [block]))
    if issues:
        sample["qc"] = {"status": "fail", "issues": sorted(set(issues))}
        return sample
    # structured decoys are known from the plan; prose decoys were grounded above
    decoys += [d for d in plan["decoys"] if d["placement"] in ("table", "card")]
    style = rng.choice(["num", "md", "par"])
    title = frame["title"]
    if fmt == "html":
        paras = []
        for kind, h, body in parts:
            paras.append(f"<h2>{html.escape(h, quote=False)}</h2>")
            for b in body:
                paras.append(b if b.lstrip().startswith("<") else f"<p>{html.escape(b, quote=False)}</p>")
        if frame["foot"]:
            paras.append(f"<p><small>{html.escape(frame['foot'], quote=False)}</small></p>")
        text, d2 = layout.to_html(title, paras, lang, rng, None)
    else:
        chunks = []
        for k, (kind, h, body) in enumerate(parts, 1):
            chunks.append(_heading(style, k, h) + "\n\n" + "\n\n".join(body))
        if frame["foot"]:
            chunks.append("* " + frame["foot"])
        body_txt = "\n\n".join(chunks)
        if fmt == "email":
            text, d2 = layout.to_email(title, body_txt, lang, rng, None)
        else:
            text, d2 = title + "\n\n" + body_txt, []
    decoys += d2
    gt = {"records": gold}
    plain = html.unescape(re.sub(r"<[^>]+>", " ", text) if fmt == "html" else text)
    for r in gold:
        for k, v in r.items():
            if v and v not in plain and norm(v) not in norm(plain):
                issues.append(f"gt_not_in_text:{k}")
    sr = script_ratio(plain, lang)
    if sr < 0.55:
        issues.append(f"script_ratio:{sr:.2f}")
    n_chars = len(re.sub(r"\s+", " ", plain))
    if not 4500 <= n_chars <= 16000:
        issues.append(f"length:{n_chars}")
    if "[[" in text:
        issues.append("leftover_markers")
    sample.update({"text": text, "gt": gt, "decoys": decoys,
                   "qc": {"status": "fail" if issues else "ok", "issues": sorted(set(issues))}})
    return sample


def build_realistic(seeds_path: Path, out_path: Path, langs: list[str], n_per_lang: int, model: str = "gen",
                    seed: int = 3, workers: int = 6, schema_id: str | None = None) -> dict:
    """Resumable parallel build of H20 documents; ids = <lang>-t<seed>-<i+1:05d>."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open(encoding="utf-8")}
    jobs = []
    for lang in langs:
        seeds = load_seeds(seeds_path, lang)
        for i in range(n_per_lang if seeds else 0):
            sid = f"{lang}-t{seed}-{i + 1:05d}"
            if sid in done:
                continue
            rng = random.Random(sid)
            fmt = rng.choices(list(FORMAT_W), list(FORMAT_W.values()))[0]
            jobs.append((sid, plan_realistic(rng, lang, fmt, schema_id), seeds[i % len(seeds)], rng))
    log.info("build_realistic: %d jobs (%d done) -> %s", len(jobs), len(done), out_path)
    stats = {"ok": 0, "fail": 0, "error": 0}
    with ThreadPoolExecutor(workers) as ex, out_path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(generate_realistic, plan, sd, rng, model, sid): sid for sid, plan, sd, rng in jobs}
        for fu in as_completed(futs):
            try:
                s = fu.result()
            except Exception as e:
                log.warning("sample %s error: %s", futs[fu], e)
                stats["error"] += 1
                continue
            stats[s["qc"]["status"]] += 1
            if s["qc"]["status"] != "ok":
                log.info("sample %s QC fail: %s", s["id"], s["qc"]["issues"])
            f.write(dump(s) + "\n")
            f.flush()
    log.info("build_realistic stats: %s", stats)
    return stats
