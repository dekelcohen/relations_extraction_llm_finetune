"""Slot-level error taxonomy for extraction predictions (uses the scorer's own record alignment + match()).

Every gold/pred slot disagreement is assigned exactly one category (first rule that fires, in this order):

  no_answer          output not parseable even after tolerant repair (e.g. thinking ran out of tokens)
  missed_record      a gold record has no aligned prediction (all its gold values are lost)
  extra_record       predicted record not aligned to any gold record (decoy entity / split / invented)
  decoy_value        predicted value equals a planted distractor (near date/org/location, old status, quoted email date)
  cross_record       predicted value belongs to ANOTHER gold record (bleeding / swapping between entities)
  missed_value       gold has a value, prediction is null
  spurious_value     gold is null (slot not stated), prediction fills it anyway (value is in the text)
  hallucinated       predicted value does not occur in the document (invented, translated, re-formatted)
  date_mismatch      date slot: both sides parse but resolve to different dates (calendar/format conversion)
  span_boundary      same value with a longer/shorter span than allowed (token overlap, below match leniency)
  wrong_value        any other in-text value of the right slot that is simply the wrong one
"""
from __future__ import annotations

from collections import Counter, defaultdict

from re_synth.score import align, compact, in_text, match, norm, parse_date, parse_prediction_lenient, slot_types_for

CATS = ["no_answer", "missed_record", "extra_record", "decoy_value", "cross_record", "missed_value", "spurious_value",
        "hallucinated", "date_mismatch", "span_boundary", "wrong_value"]
CAT_DESC = {c: line.split(None, 1)[1].strip() for c in CATS for line in __doc__.splitlines()
            if line.strip().startswith(c + " ")}


def _overlap(a, b) -> bool:
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return bool(ta & tb)


def classify_sample(s: dict, raw: str) -> list[dict]:
    st = slot_types_for(s["schema_id"], inline=s.get("schema"))
    gold = s["gt"].get("records", [])
    decoys = s.get("decoys", [])
    tn, tc = norm(s["text"]), compact(s["text"])
    recs, valid, repair = parse_prediction_lenient(raw)
    errs = []

    def add(cat, slot=None, gold_v=None, pred_v=None, rec=None):
        errs.append({"id": s["id"], "lang": s["lang"], "schema": s["schema_id"], "hc": s.get("hard_cases", []),
                     "cat": cat, "slot": slot, "type": st.get(slot) if slot else None, "gold": gold_v, "pred": pred_v,
                     "record": rec, "repair": repair})

    if not valid:
        add("no_answer", gold_v=f"{sum(1 for g in gold for k in st if g.get(k) is not None)} gold values")
        return errs
    preds = recs or []
    pairs = align(preds, gold, st)
    used_p, used_g = {i for i, _ in pairs}, {j for _, j in pairs}

    def is_decoy(v, typ):
        return any(d.get("type") == typ and match(v, d.get("value"), typ) for d in decoys)

    for j, g in enumerate(gold):
        if j not in used_g:
            add("missed_record", gold_v=g.get("main_entity"), rec=j)
    for i, p in enumerate(preds):
        if i not in used_p:
            add("extra_record", pred_v=p.get("main_entity"), rec=i)
    for i, j in pairs:
        p, g = preds[i], gold[j]
        for k, typ in st.items():
            gv, pv = g.get(k), p.get(k)
            if gv is None and pv is None:
                continue
            if gv is not None and pv is not None and match(pv, gv, typ):
                continue
            if pv is None:
                add("missed_value", k, gv, pv, j)
            elif is_decoy(pv, typ):
                add("decoy_value", k, gv, pv, j)
            elif any(match(pv, o.get(k), typ) for jj, o in enumerate(gold) if jj != j and o.get(k) is not None):
                add("cross_record", k, gv, pv, j)
            elif gv is None:
                add("spurious_value" if in_text(pv, tn, tc) else "hallucinated", k, gv, pv, j)
            elif not in_text(pv, tn, tc):
                add("hallucinated", k, gv, pv, j)
            elif typ == "date" and parse_date(str(pv)) and parse_date(str(gv)):
                add("date_mismatch", k, gv, pv, j)
            elif _overlap(pv, gv):
                add("span_boundary", k, gv, pv, j)
            else:
                add("wrong_value", k, gv, pv, j)
    return errs


def analyze(samples: list[dict], preds: dict[str, str]) -> list[dict]:
    out = []
    for s in samples:
        if s["id"] in preds:
            out.extend(classify_sample(s, preds[s["id"]]))
    for e in out:
        e["root"] = root_cause(e)
    return out


# Root-cause groups (first matching rule), derived from manual review of the fine-tuned Qwen3-4B errors.
ROOTS = {
    "no_answer": "No final answer: thinking hit the token budget (looping), nothing to parse.",
    "list_header_propagation": "List/table header states the org once ('the following X belong to Y', table caption); "
                               "the model must copy it into every row (H6/H7c) but leaves it null or takes the "
                               "distractor column/caption instead.",
    "copy_error": "Value not copied verbatim: misspelling/transliteration of a name or place, foreign-script tokens "
                  "leaking in, or a value invented outright.",
    "span_convention": "Right value, wrong span: includes descriptive context ('returned to the warehouse fully', "
                       "'the conference center in Lod') or trims it; mostly an annotation-convention mismatch.",
    "multi_entity_bleeding": "Two+ entities in one document (H4): values swapped between records or bled into the "
                             "record that should stay null; or records split/merged/missed.",
    "distractor_or_role": "Took a nearby same-type distractor (other org/date/location, earlier status, quoted email "
                          "date) or swapped roles (supplier vs client).",
    "long_range_or_coref": "Value stated far from the entity, via pronoun/alias, or in the email header/quoted "
                           "thread (H2/H3/H5/H11) and the model left it null.",
    "null_slot_filled": "Slot not stated for this entity (H9) but the model filled it from elsewhere in the text.",
    "other_miss": "Other missed value.",
}


def root_cause(e: dict) -> str:
    c, hc = e["cat"], set(e["hc"])
    if c == "no_answer":
        return "no_answer"
    if c == "hallucinated":
        return "copy_error"
    if hc & {"H6", "H7c"} and e["type"] == "org" and c in ("missed_value", "decoy_value", "wrong_value"):
        return "list_header_propagation"
    if c == "span_boundary":
        return "span_convention"
    if c in ("cross_record", "missed_record", "extra_record"):
        return "multi_entity_bleeding"
    if c in ("decoy_value", "wrong_value", "date_mismatch"):
        return "distractor_or_role"
    if c == "spurious_value":
        return "null_slot_filled"
    if c == "missed_value":
        return "long_range_or_coref" if hc & {"H2", "H3", "H5", "H11"} else "other_miss"
    return "other_miss"


def summarize(errs: list[dict]) -> dict:
    by = lambda key: {c: dict(Counter(e[key] for e in errs if e["cat"] == c)) for c in CATS}  # noqa: E731
    hc = defaultdict(Counter)
    for e in errs:
        for h in e["hc"]:
            hc[e["cat"]][h] += 1
    return {"total": len(errs), "by_cat": dict(Counter(e["cat"] for e in errs)), "by_lang": by("lang"),
            "by_slot_type": by("type"), "by_hc": {c: dict(v) for c, v in hc.items()},
            "samples_with_errors": len({e["id"] for e in errs})}
