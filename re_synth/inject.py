"""LLM scrub+inject: seed excerpt + plan -> document with marked values -> verified sample with GT."""
from __future__ import annotations

import difflib
import json
import logging
import random
import re

from . import layout
from .hardcases import directives
from .llm import chat
from .schemas import SCHEMAS
from .score import norm

log = logging.getLogger(__name__)
LANG_NAME = {"he": "Hebrew", "fa": "Persian (Farsi)", "en": "English"}
GENRES = {"text": ["news report", "official notice", "internal memo", "forum post", "encyclopedic paragraph",
                   "chat conversation summary", "police/incident report", "local blog post"],
          "html": ["news web page", "company announcement page", "classifieds listing page", "municipal notice page"],
          "email": ["business email", "customer-service email", "operations update email"]}
MARK = re.compile(r"\[\[(r\d+\.[a-z_]+|x\d+)\]\](.*?)\[\[/\]\]", re.S)

SYSTEM = ("You generate realistic synthetic documents for training information-extraction models. "
          "You follow the marking rules exactly and output only the requested <title>/<body> tags.")


def build_prompt(plan: dict, seed: dict, genre: str, block: bool) -> str:
    schema = SCHEMAS[plan["schema_id"]]
    lang = LANG_NAME[plan["lang"]]
    lines = []
    if not block:
        for i, r in enumerate(plan["records"]):
            lines.append(f"Record r{i}: [[r{i}.main]] = {r['main_entity']}")
            for k, v in r["slots"].items():
                if "H3" in plan["hard_cases"] and k == schema.id_slot:
                    continue
                d = schema.slot(k).definition
                lines.append(f"  [[r{i}.{k}]] = {v['value'] if v else 'NOT STATED (do not mention any ' + k + ')'}"
                             f"    # {d}")
        for j, x in enumerate(plan["decoys"]):
            if x["placement"] in ("quoted", "table", "footer", "header"):
                continue
            lines.append(f"Distractor [[x{j}]] = {x['value']}  -> role: {x['role']} (related to r{x['record']}, "
                         f"placement: {x['placement']}; it is NOT the {x['slot']} of r{x['record']})")
    rules = [
        f"Write a {genre} in {lang} (180-350 words) that uses the seed's topic/style as background.",
        f"Scrub: do not keep any {schema.main_type} entities, IDs, dates, places or organizations from the seed that "
        "could be confused with the facts below (replace or drop them).",
        "Wrap each fact value exactly once (its main statement) with its marker, e.g. [[r0.location]]VALUE[[/]]; the "
        "first mention of each main entity as [[r0.main]]NAME[[/]]; each distractor as [[x0]]VALUE[[/]].",
        "Copy values character-by-character (same digits, same spelling). Grammatical prefixes/suffixes (Hebrew "
        "ב/ל/מ/ה/ו/ש/כ, Persian را) go OUTSIDE the markers.",
        f"Do not mention any other {schema.main_type} besides the given records.",
    ] + directives(plan)
    if block:
        rules = rules[:2] + [
            f"Include a line containing only [[BLOCK]] where a list/table of {layout.PLURAL[schema.id]['en']} will be "
            f"inserted. Do NOT mention any specific {schema.main_type}, ID or organization name elsewhere; do not use any "
            "other [[...]] markers.",
            "Write the text before and after the block so that it reads naturally (an introduction and a closing).",
        ]
    if "H3" in plan["hard_cases"]:
        r0 = plan["records"][0]
        rules.append(f"The title is the email SUBJECT and must contain [[r0.main]]{r0['main_entity']}[[/]] and "
                     f"[[r0.{schema.id_slot}]]{r0['slots'][schema.id_slot]['value']}[[/]]. The body must NOT repeat "
                     f"the {schema.id_slot}; in the body refer to the entity generically (e.g. 'the {schema.main_type}' "
                     "in the target language) and wrap that first generic mention as [[r0.main]]...[[/]] only if it is "
                     "the full name.")
    out_fmt = '"<title>TITLE</title>\n<body>\nPARAGRAPHS (separated by blank lines)\n</body>"'
    return (f"SEED ARTICLE ({seed.get('title', '')}):\n<<<\n{seed['text']}\n>>>\n\n"
            + ("FACTS:\n" + "\n".join(lines) + "\n\n" if lines else "")
            + "RULES:\n" + "\n".join(f"- {r}" for r in rules) + f"\n\nReturn exactly this format (no JSON, no code fences): {out_fmt}")


def parse_title_body(raw) -> tuple[str, str]:
    raw = raw if isinstance(raw, str) else ""
    t = re.search(r"<title>(.*?)</title>", raw, re.S)
    b = re.search(r"<body>(.*?)(?:</body>|$)", raw, re.S)
    return (t.group(1).strip() if t else ""), (b.group(1).strip() if b else "")


def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def strip_markers(s: str) -> tuple[str, list[tuple[str, str]]]:
    found = [(m.group(1), m.group(2)) for m in MARK.finditer(s)]
    return MARK.sub(lambda m: m.group(2), s), found


def script_ratio(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "he":
        ok = sum("\u0590" <= c <= "\u05ff" for c in letters)
    elif lang == "fa":
        ok = sum("\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufeff" for c in letters)
    else:
        ok = sum(c.isascii() for c in letters)
    return ok / len(letters)


def verify_and_ground(plan: dict, title: str, body: str) -> tuple[dict | None, list[str], list[dict]]:
    """Check marked values against plan; returns (gt records, issues, kept decoys). GT uses document surfaces."""
    schema = SCHEMAS[plan["schema_id"]]
    issues = []
    _, found_t = strip_markers(title)
    clean_body, found_b = strip_markers(body)
    found = found_t + found_b
    marks: dict[str, str] = {}
    for tag, val in found:
        marks.setdefault(tag, val.strip())
    full_clean = strip_markers(title)[0] + "\n" + clean_body
    if "[[" in full_clean or "]]" in full_clean:
        issues.append("leftover_markers")
    recs = []
    for i, r in enumerate(plan["records"]):
        main = marks.get(f"r{i}.main")
        if main is None or _fuzzy(main, r["main_entity"]) < 0.6:
            if r["main_entity"] in full_clean:
                main = r["main_entity"]
            else:
                issues.append(f"r{i}.main_missing")
                main = r["main_entity"]
        out = {"main_entity": main}
        for k, v in r["slots"].items():
            if v is None:
                out[k] = None
                continue
            got = marks.get(f"r{i}.{k}")
            if got is not None and (norm(got) == norm(v["value"]) or _fuzzy(got, v["value"]) >= 0.75):
                out[k] = got
            elif v["value"] in full_clean:
                out[k] = v["value"]
            else:
                issues.append(f"r{i}.{k}_missing")
                out[k] = v["value"]
        recs.append(out)
    kept = []
    for j, x in enumerate(plan["decoys"]):
        if x["placement"] in ("quoted", "table", "footer", "header"):
            continue
        got = marks.get(f"x{j}")
        if got is not None and _fuzzy(got, x["value"]) >= 0.75:
            kept.append({**x, "value": got})
        elif x["value"] in full_clean:
            kept.append(x)
    if "H3" in plan["hard_cases"]:
        idv = recs[0][schema.id_slot]
        if idv and norm(idv) in norm(clean_body):
            issues.append("H3_id_in_body")
        if idv and norm(idv) not in norm(strip_markers(title)[0]):
            issues.append("H3_id_not_in_subject")
    return {"records": recs}, issues, kept


def generate_sample(plan: dict, seed: dict, rng: random.Random, model: str = "gen", sid: str = "") -> dict:
    """Plan -> LLM -> verify -> layout. Returns a sample dict with qc.status in {ok, fail}."""
    schema = SCHEMAS[plan["schema_id"]]
    lang, fmt = plan["lang"], plan["format"]
    block = "H6" in plan["hard_cases"] or "H7" in plan["hard_cases"]
    genre = rng.choice(GENRES[fmt])
    prompt = build_prompt(plan, seed, genre, block)
    raw = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], model,
               temperature=0.8, max_tokens=12000, salt=sid, reasoning_effort="low")
    title, body = parse_title_body(raw)
    sample = {"id": sid, "lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": plan["hard_cases"],
              "seed": {k: seed.get(k) for k in ("source", "dataset", "title", "url")}, "genre": genre,
              "gen": {"model": model}, "plan": plan}
    if not body:
        sample["qc"] = {"status": "fail", "issues": ["no_json"]}
        return sample
    decoys: list[dict] = []
    if block:
        if "[[BLOCK]]" not in body:
            body = body.rstrip() + "\n\n[[BLOCK]]"
        body = MARK.sub(lambda m: m.group(2), body)
        gt = {"records": [{"main_entity": r["main_entity"],
                           **{k: (v["value"] if v else None) for k, v in r["slots"].items()}} for r in plan["records"]]}
        issues = []
    else:
        gt, issues, decoys = verify_and_ground(plan, title, body)
    title_c, body_c = strip_markers(title)[0], strip_markers(body)[0]
    paras = [p.strip() for p in body_c.split("\n\n") if p.strip()]

    block_text = block_html = None
    if "H6" in plan["hard_cases"]:
        sub, filler, items = layout.h6_block(schema, plan["records"], lang, seed["text"], rng)
        block_text = "\n".join([sub] + filler + items)
        block_html = (f"<p><b>{sub}</b></p>" + "".join(f"<p>{f}</p>" for f in filler)
                      + "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    if "H7" in plan["hard_cases"]:
        block_html, tdecoys = layout.h7_table(schema, plan["records"], lang, rng, "H7c" in plan["hard_cases"])
        decoys += tdecoys

    if fmt == "html":
        text, d2 = layout.to_html(title_c, paras, lang, rng, block_html)
        decoys += d2
    elif fmt == "email":
        body_txt = "\n\n".join(block_text if p == "[[BLOCK]]" else p for p in paras)
        quoted = None
        if "H5" in plan["hard_cases"]:
            qd = next((x for x in plan["decoys"] if x["placement"] == "quoted"), None)
            if qd:
                sents = layout._sentences(seed["text"])
                quoted = (qd["value"], rng.choice(sents) if sents else "...")
        text, d2 = layout.to_email(title_c, body_txt, lang, rng, quoted)
        decoys += d2
    else:
        text = title_c + "\n\n" + "\n\n".join(block_text if p == "[[BLOCK]]" else p for p in paras)

    # final checks on the rendered document
    import html as _h
    plain = _h.unescape(text)
    for r in gt["records"]:
        for k, v in r.items():
            if v and v not in plain and norm(v) not in norm(plain):
                issues.append(f"gt_not_in_text:{k}")
    sr = script_ratio(re.sub(r"<[^>]+>", " ", plain), lang)
    if sr < (0.8 if lang == "en" else 0.55):
        issues.append(f"script_ratio:{sr:.2f}")
    n_words = len(re.sub(r"<[^>]+>", " ", plain).split())
    if not 60 <= n_words <= 1500:
        issues.append(f"length:{n_words}")
    if "[[" in text:
        issues.append("leftover_markers")
    sample.update({"text": text, "gt": gt, "decoys": decoys, "qc": {"status": "fail" if issues else "ok",
                                                                    "issues": sorted(set(issues))}})
    return sample


def dump(sample: dict) -> str:
    return json.dumps(sample, ensure_ascii=False)
