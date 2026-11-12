"""Plan a sample: GT records, decoys (distractors) and hard-case directives (H1-H12, see docs/plans)."""
from __future__ import annotations

import random

from . import lexicon as L
from .schemas import SCHEMAS, Schema, gen_main, gen_value

HARD_CASES = {
    "H1": "near distractor of the same type (date/location/org/id) close to the main entity",
    "H2": "true slot value 2-3 sentences away, distractor in between",
    "H3": "email: ID only in the subject header, far from the body",
    "H4": "two entities; one lacks a slot -> must stay null (no bleeding)",
    "H5": "email quoted thread with dates of previous messages",
    "H6": "subtitle with org refers to a list several lines below",
    "H7": "HTML table with a distractor column / caption org",
    "H9": "slot not stated -> null",
    "H10": "locale formats (Jalali, Persian digits, Hebrew calendar, prefixes)",
    "H11": "coreference: slot stated via pronoun / alias",
    "H12": "status changed: earlier status is a distractor",
    # large-table family (deterministic block, gold reliable): header far from the values
    "H13": "large table (6-10 rows x 9-14 cols): several same-type distractor + noise columns, header only at the top",
    "H14": "table section rows ('Owner: X') - the org applies to all following rows until the next section",
    "H15": "two-level header: identical leaf labels under 'current' vs 'previous' group headers",
    "H16": "ditto marks (〃 / כנ״ל / همان) - the cell repeats the value from the row above",
    "H17": "transposed table: entities are columns, attributes (incl. distractor attributes) are rows",
    "H18": "plain-text (pipe) table inside text / email instead of HTML",
    "H19": "long document: 10-30 distinct entities in 2-4 sections (tables / lists / org-headed lists / cards)",
}
TABLE_CASES = ("H13", "H14", "H15", "H16", "H17", "H18")
BLOCK_CASES = ("H6", "H7", "H13", "H19")
FOCUS_LONG_P = 0.2  # focus mode: share of long multi-section documents (H19)
LONG_KINDS = ("table", "list", "list_org", "cards")

DECOY_ROLES = {
    "date": ["the date it was manufactured / founded", "the date of a previous, unrelated message",
             "a person's birth date", "the date of an older inspection", "the publication date of a related article"],
    "location": ["where it was manufactured", "where the owner's employee lives/works", "where it was previously located",
                 "the headquarters of an unrelated company"],
    "org": ["the manufacturer", "the insurance company", "a previous owner", "an unrelated company mentioned in passing"],
    "id": ["a phone number of a contact person", "an expired / previous reference number", "an unrelated reference number"],
    "person": ["a previous holder", "a manager who approved a request"],
    "amount": ["a previous offer that was rejected", "a fine paid last year"],
    "status": ["an earlier status that has since changed"],
    "text": ["a previous job title"],
}

NULLABLE_P = 0.2


def _gen_decoy(schema: Schema, slot, rng, lang, avoid: set, own: set | None = None) -> str:
    """Value of the slot's type not in `avoid`; small vocabularies fall back to differing only from `own`."""
    from .score import norm
    for av in (avoid, own or set()):
        navoid = {norm(a) for a in av}
        for _ in range(40):
            v = gen_value(schema, slot, rng, lang)["value"]
            if v not in av and norm(v) not in navoid:
                return v
    return v


# "focus" mode (build --focus): oversample the hard cases behind the fine-tuned model's pilot errors
# (eval/error_analysis.py root causes): long-range/coref (H2,H3,H5,H11), list/table header org (H6,H7,H7c),
# distractors/status (H1,H12) and multi-entity bleeding (H4).
FOCUS_POOL_W = {"H1": 2, "H2": 3, "H4": 4, "H9": 1, "H11": 3, "H12": 4}


def _focus_cases(rng: random.Random, fmt: str, schema: Schema) -> list[str]:
    pool, cases = dict(FOCUS_POOL_W), []
    for _ in range(rng.choice([2, 3, 3])):
        c = rng.choices(list(pool), list(pool.values()))[0]
        cases.append(c)
        pool.pop(c)
    if fmt == "email":
        if schema.id_slot and rng.random() < 0.7:
            cases.append("H3")
        if rng.random() < 0.85:
            cases.append("H5")
    if fmt == "html" and rng.random() < 0.85:
        cases.append("H7")
    if fmt != "email" and "H7" not in cases and rng.random() < 0.5:
        cases.append("H6")
    return cases


# focus mode: share of samples that get a large table (H13 family), per format
FOCUS_TABLE_P = {"html": 0.75, "text": 0.4, "email": 0.4}
DIST_TYPES = ("location", "status", "date", "org", "person", "id", "amount")


def _table_cases(rng: random.Random, fmt: str) -> list[str]:
    cases = ["H13"] + (["H18"] if fmt != "html" else [])
    if rng.random() < 0.2:
        return cases + ["H17"]
    return cases + rng.sample(["H14", "H15", "H16"], k=rng.choice([1, 1, 2, 2, 3]))


def _plan_table(schema: Schema, cases: list[str], records: list[dict], rng: random.Random, lang: str) -> tuple[dict, list[dict]]:
    """Mutate records for H14 (sections) / H16 (ditto runs); choose distractor columns. Returns (table cfg, decoys)."""
    cfg: dict = {"sections": [], "ditto_slot": None, "prev_slots": [], "dist_cols": []}
    cont = schema.container_slot
    if "H14" in cases:
        k = min(rng.choice([2, 3]), len(records) // 2)
        cuts = sorted(rng.sample(range(2, len(records) - 1), k - 1)) if k > 1 else []
        starts = [0] + cuts
        orgs = []
        while len(orgs) < k:
            o = gen_value(schema, schema.slot(cont), rng, lang)
            if o["value"] not in [x["value"] for x in orgs]:
                orgs.append(o)
        for gi, st in enumerate(starts):
            end = starts[gi + 1] if gi + 1 < len(starts) else len(records)
            for r in records[st:end]:
                r["slots"][cont] = dict(orgs[gi])
            cfg["sections"].append([st, orgs[gi]["value"]])
    if "H16" in cases:
        cand = [s.name for s in schema.slots if s.type in ("location", "org") and not (s.name == cont and "H14" in cases)]
        if cand:
            ds = cfg["ditto_slot"] = rng.choice(cand)
            for i in range(1, len(records)):
                if records[i - 1]["slots"][ds] and (rng.random() < 0.55 or i == 1):
                    records[i]["slots"][ds] = dict(records[i - 1]["slots"][ds])
    all_vals = {v["value"] for r in records for v in r["slots"].values() if v}
    avoid_prev = {cont} if "H14" in cases else set()
    typed = [s for s in schema.slots if s.type in DIST_TYPES and s.name != schema.id_slot and s.name not in avoid_prev]
    if "H15" in cases and typed:
        cfg["prev_slots"] = [s.name for s in rng.sample(typed, k=min(len(typed), rng.choice([2, 2, 3])))]
    n_dist = rng.choice([1, 2]) if cfg["prev_slots"] else rng.choice([2, 3, 3])
    pool = [s for s in schema.slots if s.type in DIST_TYPES and s.name not in cfg["prev_slots"]]
    cfg["dist_cols"] = [s.name for s in rng.sample(pool, k=min(n_dist, len(pool)))]
    decoys = []
    for kind, names in (("prev", cfg["prev_slots"]), ("dist", cfg["dist_cols"])):
        for name in names:
            s = schema.slot(name)
            for i, r in enumerate(records):
                own = {r["slots"][name]["value"]} if r["slots"].get(name) else set()
                avoid = own if s.type == "status" else all_vals | own  # few statuses: only differ from this row
                decoys.append({"slot": name, "type": s.type, "value": _gen_decoy(schema, s, rng, lang, avoid, own),
                               "role": f"table {kind} column for {name}", "record": i, "placement": "table",
                               "col": f"{kind}:{name}"})
    return cfg, decoys


def _plan_long(schema: Schema, rng: random.Random, lang: str, fmt: str) -> dict:
    """H19: 10-30 distinct entities split into 2-4 sections, each rendered as table / list / org-headed list / cards."""
    n = rng.randint(10, 30)
    names, tries = [], 0
    while len(names) < n and tries < 3000:
        tries += 1
        m = gen_main(schema, rng, lang)
        if m not in names:
            names.append(m)
    n = len(names)
    null_p = rng.choice([0.0, 0.1, 0.2])
    records = []
    for m in names:
        slots = {}
        for sl in schema.slots:
            slots[sl.name] = None if sl.name != schema.id_slot and rng.random() < null_p else gen_value(schema, sl, rng, lang)
        records.append({"main_entity": m, "slots": slots})
    k = min(rng.randint(2, 4), n // 3)
    cuts = sorted(rng.sample(range(3, n - 2), k - 1)) if k > 1 else []
    bounds = list(zip([0] + cuts, cuts + [n]))
    kinds = [rng.choice(LONG_KINDS) for _ in bounds]
    if len(set(kinds)) == 1 and len(kinds) > 1:
        kinds[-1] = rng.choice([x for x in LONG_KINDS if x != kinds[0]])
    cases, decoys, sections = {"H19"}, [], []
    used_orgs: set = set()
    for (st, en), kind in zip(bounds, kinds):
        sub = records[st:en]
        sec = {"kind": kind, "start": st, "end": en, "org": None, "table": None, "sub_cases": [], "dist": []}
        if kind == "list_org":  # container org stated only in the section heading
            o = gen_value(schema, schema.slot(schema.container_slot), rng, lang)
            while o["value"] in used_orgs:
                o = gen_value(schema, schema.slot(schema.container_slot), rng, lang)
            used_orgs.add(o["value"])
            sec["org"] = o["value"]
            for r in sub:
                r["slots"][schema.container_slot] = dict(o)
            cases.add("H6")
        if kind == "table":
            sc = ["H13"] + rng.sample(["H14", "H15", "H16"], k=rng.choice([0, 1, 2])) + (["H18"] if fmt != "html" else [])
            if len(sub) < 4 and "H14" in sc:
                sc.remove("H14")
            cfg, td = _plan_table(schema, sc, sub, rng, lang)
            sec["table"], sec["sub_cases"] = cfg, sc
            cases.update(sc)
            for d in td:
                decoys.append({**d, "record": d["record"] + st, "col": f"s{len(sections)}:{d['col']}"})
        if kind == "cards":  # 1-2 labelled distractor lines per card
            pool = [x for x in schema.slots if x.type in DIST_TYPES]
            sec["dist"] = [x.name for x in rng.sample(pool, k=min(len(pool), rng.choice([1, 2])))]
            vals = {v["value"] for r in records for v in r["slots"].values() if v}
            for i, r in enumerate(sub):
                for name in sec["dist"]:
                    sl = schema.slot(name)
                    own = {r["slots"][name]["value"]} if r["slots"].get(name) else set()
                    decoys.append({"slot": name, "type": sl.type, "record": st + i, "placement": "card",
                                   "role": f"card distractor line for {name}", "col": f"s{len(sections)}:dist:{name}",
                                   "value": _gen_decoy(schema, sl, rng, lang, own if sl.type == "status" else vals | own, own)})
        sections.append(sec)
    if any(v is None for r in records for v in r["slots"].values()):
        cases.add("H9")
    return {"lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": sorted(cases),
            "records": records, "decoys": decoys, "long": {"sections": sections}}


def plan_sample(rng: random.Random, lang: str, fmt: str, schema_id: str | None = None,
                force_cases: list[str] | None = None, focus: bool = False) -> dict:
    long_pick = focus and not force_cases and rng.random() < FOCUS_LONG_P
    if long_pick:
        schema = SCHEMAS[schema_id or rng.choice([k for k, v in SCHEMAS.items() if v.id_slot])]
        return _plan_long(schema, rng, lang, fmt)
    table_pick = focus and not force_cases and rng.random() < FOCUS_TABLE_P[fmt]
    if focus and not schema_id and (table_pick or fmt != "email" and rng.random() < 0.7):  # H6/H7/H13 need an id slot
        schema_id = rng.choice([k for k, v in SCHEMAS.items() if v.id_slot])
    schema = SCHEMAS[schema_id or rng.choice(list(SCHEMAS))]
    cases: list[str] = list(force_cases or [])
    if table_pick:
        cases += _table_cases(rng, fmt)
        if rng.random() < 0.5:
            cases.append("H9")
    elif focus and not force_cases:
        cases += _focus_cases(rng, fmt, schema)
    elif not force_cases:
        pool = ["H1", "H2", "H4", "H9", "H11", "H12"]
        cases += rng.sample(pool, k=rng.choice([1, 2, 2, 3]))
        if fmt == "email":
            if schema.id_slot and rng.random() < 0.6:
                cases.append("H3")
            if rng.random() < 0.6:
                cases.append("H5")
        if fmt == "html" and rng.random() < 0.6:
            cases.append("H7")
        if fmt != "email" and "H7" not in cases and rng.random() < 0.25:
            cases.append("H6")
    if "H3" in cases and not schema.id_slot:
        cases.remove("H3")
    if ("H6" in cases or "H7" in cases) and not schema.id_slot:
        cases = [c for c in cases if c not in ("H6", "H7")]
    if "H12" in cases and not any(s.type == "status" for s in schema.slots):
        cases.remove("H12")
    cases = sorted(set(cases))

    def make_record(null_p: float) -> dict:
        slots = {}
        for s in schema.slots:
            if s.name != schema.id_slot and rng.random() < null_p:
                slots[s.name] = None
            else:
                slots[s.name] = gen_value(schema, s, rng, lang)
        return {"main_entity": gen_main(schema, rng, lang), "slots": slots}

    table = "H13" in cases
    if table:
        cases = [c for c in cases if c in TABLE_CASES or c == "H9"]
    n_rec = 1
    if table:
        n_rec = rng.randint(3, 5) if "H17" in cases else rng.randint(6, 10)
    elif "H4" in cases:
        n_rec = 2
    if "H6" in cases or "H7" in cases:
        n_rec = rng.randint(2, 4)
    records = [make_record(NULLABLE_P if "H9" in cases else 0.0) for _ in range(n_rec)]
    # distinct main entities
    from collections import Counter
    seen: Counter = Counter()
    max_dup = 2 if table and schema.id_slot else 1  # table rows may repeat an item name; the ID column tells them apart
    for r in records:
        for _ in range(50):
            if seen[r["main_entity"]] < max_dup:
                break
            r["main_entity"] = gen_main(schema, rng, lang)
        seen[r["main_entity"]] += 1
    if table:
        cfg, tdecoys = _plan_table(schema, cases, records, rng, lang)
        if "H9" not in cases and any(v is None for r in records for v in r["slots"].values()):
            cases.append("H9")
        return {"lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": sorted(set(cases)),
                "records": records, "decoys": tdecoys, "table": cfg}

    if "H4" in cases:  # second record lacks a date/location slot -> must not inherit record 0's value
        cand = [s.name for s in schema.slots if s.type in ("date", "location") and records[0]["slots"][s.name]]
        if cand:
            records[1]["slots"][rng.choice(cand)] = None
    if "H6" in cases:  # list items: main + id only, container org shared
        org = gen_value(schema, schema.slot(schema.container_slot), rng, lang)
        for r in records:
            r["slots"] = {s.name: (r["slots"][s.name] if s.name == schema.id_slot else
                                   org if s.name == schema.container_slot else None) for s in schema.slots}
    if "H7" in cases and rng.random() < 0.5:  # shared org in caption
        org = gen_value(schema, schema.slot(schema.container_slot), rng, lang)
        for r in records:
            r["slots"][schema.container_slot] = org
        cases.append("H7c")
    if "H6" in cases or "H7" in cases:  # records live in a deterministic block; LLM only writes the context
        cases = [c for c in cases if c not in ("H1", "H2", "H4", "H11", "H12", "H9")]
    if "H3" in cases:
        records = records[:1]
        cases = [c for c in cases if c != "H4"]
    if "H9" not in cases and any(v is None for r in records for v in r["slots"].values()):
        cases.append("H9")

    all_vals = {v["value"] for r in records for v in r["slots"].values() if v}
    decoys = []
    block = "H6" in cases or "H7" in cases
    if not block and (any(c in cases for c in ("H1", "H2", "H5", "H12")) or rng.random() < 0.5):
        n = rng.randint(1, 3)
        r0 = records[0]
        cand_slots = [s for s in schema.slots if s.name != schema.id_slot or "H1" in cases]
        for s in rng.sample(cand_slots, k=min(n, len(cand_slots))):
            if s.type == "status" and "H12" not in cases:
                continue
            decoys.append({"slot": s.name, "type": s.type, "value": _gen_decoy(schema, s, rng, lang, all_vals),
                           "role": rng.choice(DECOY_ROLES[s.type]), "record": 0,
                           "placement": "between" if "H2" in cases and r0["slots"].get(s.name) else "near"})
    if "H12" in cases:
        st = next(s for s in schema.slots if s.type == "status")
        if records[0]["slots"].get(st.name) is None:
            records[0]["slots"][st.name] = gen_value(schema, st, rng, lang)
        if not any(d["slot"] == st.name for d in decoys):
            decoys.append({"slot": st.name, "type": "status", "role": DECOY_ROLES["status"][0], "record": 0,
                           "value": _gen_decoy(schema, st, rng, lang, {records[0]["slots"][st.name]["value"]}),
                           "placement": "near"})
    if "H5" in cases and not any(d["type"] == "date" for d in decoys):
        ds = next((s for s in schema.slots if s.type == "date"), None)
        if ds:
            decoys.append({"slot": ds.name, "type": "date", "role": "the date of a previous message in the thread",
                           "value": _gen_decoy(schema, ds, rng, lang, all_vals), "record": 0, "placement": "quoted"})
    if any(v and v.get("style") in ("hebcal", "jalali_num_fa", "jalali_long_fa", "jalali_long", "jalali_num")
           for r in records for v in r["slots"].values()):
        cases.append("H10")
    return {"lang": lang, "format": fmt, "schema_id": schema.id, "hard_cases": sorted(set(cases)),
            "records": records, "decoys": decoys}


def directives(plan: dict) -> list[str]:
    """Natural-language instructions for the injector LLM, per hard case."""
    c, out = plan["hard_cases"], []
    if "H2" in c:
        out.append("Mention the main entity first; state the 'between'-placed true values 2-3 sentences later, and put "
                   "the matching distractor value in a sentence BETWEEN the entity and its true value.")
    if "H1" in c:
        out.append("Place each 'near' distractor in the same or adjacent sentence as the main entity, clearly with its "
                   "described role (so a careful reader knows it is not the requested slot).")
    if "H4" in c:
        out.append("Mention record r1 right after r0 (adjacent sentences). Record r1 has NO value for its null slots: "
                   "do not state any such value for r1, and make sure r0's value is not described as applying to r1.")
    if "H11" in c:
        out.append("After the first mention, refer to the main entity by a pronoun or short alias when stating at least "
                   "one slot value.")
    if "H12" in c:
        out.append("Describe the status distractor as an EARLIER status, then later state the true status as the "
                   "current/latest status.")
    if "H9" in c:
        out.append("For slots marked null, do not state any value for that entity (no guessable hint either).")
    return out
