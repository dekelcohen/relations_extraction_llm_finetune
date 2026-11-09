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
}

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


def _gen_decoy(schema: Schema, slot, rng, lang, avoid: set) -> str:
    for _ in range(10):
        v = gen_value(schema, slot, rng, lang)["value"]
        if v not in avoid:
            return v
    return v


def plan_sample(rng: random.Random, lang: str, fmt: str, schema_id: str | None = None,
                force_cases: list[str] | None = None) -> dict:
    schema = SCHEMAS[schema_id or rng.choice(list(SCHEMAS))]
    cases: list[str] = list(force_cases or [])
    if not force_cases:
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

    n_rec = 1
    if "H4" in cases:
        n_rec = 2
    if "H6" in cases or "H7" in cases:
        n_rec = rng.randint(2, 4)
    records = [make_record(NULLABLE_P if "H9" in cases else 0.0) for _ in range(n_rec)]
    # distinct main entities
    seen = set()
    for r in records:
        for _ in range(20):
            if r["main_entity"] not in seen:
                break
            r["main_entity"] = gen_main(schema, rng, lang)
        seen.add(r["main_entity"])

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
