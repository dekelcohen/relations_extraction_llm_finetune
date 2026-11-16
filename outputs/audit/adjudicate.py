"""Turn the per-slot diffs (build_diffs.py) + manual verdicts into the label-noise audit outputs.

python outputs/audit/build_diffs.py && python outputs/audit/build_diffs.py --4b && python outputs/audit/adjudicate.py
Writes outputs/audit/{label_noise_audit.jsonl, gold_patches.jsonl, remove_ids.json, f1_corrected.json} and the
corrected copies data/pilot/export/pilot_all_corrected.jsonl, data/pilot/samples_corrected.jsonl.
Categories: a gold_wrong, b gold_incomplete, c ambiguous, d model_error (subtype), e scorer_issue.
"""
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from re_synth.score import compact, in_text, match, norm, parse_date  # noqa: E402

A = os.path.join(ROOT, "outputs/audit")
CAT = {"a": "gold_wrong", "b": "gold_incomplete", "c": "ambiguous", "d": "model_error", "e": "scorer_issue"}
INT4, B4 = "qwen3.8-27b-int4", "qwen3-4b-ft"
SPAN = ("e", None, "span_boundary", "Same value; pred keeps a qualifier/descriptor around the gold head "
        "(gold is a sub-span). match() rejects it (substring ratio <0.6, token subset <0.33).",
        "scorer: accept pred that contains the whole gold value as a head/suffix (or store gold alternates)")
SPANLOC = "re_synth/score.py match() substring/token-subset leniency rules"

# (id, rec, slot) -> (cat, subtype, detail, reason, proposed_fix[, new_gold_value])
# ---- Qwen3.8-27B-INT4 vs gold (manual, by reading the document + slot definition)
V27 = {
    # grpo_dev
    ("he-2-00065", 7, "supplier_org"): ("d", "missed", "ditto_unresolved", 'Supplier cell is a ditto mark (") chained from the first row; gold resolves it correctly.', "none (gold correct)"),
    ("he-2-00129", 2, "destination"): ("d", "missed", "ditto_unresolved", 'Destination cell is a ditto mark (") = previous row "אילת"; all models left it null.', "none (gold correct)"),
    ("he-2-00136", 0, "status"): SPAN,
    ("he-2-00239", 1, "origin"): ("d", "missed", "ditto_unresolved", "Origin cell is 〃 (ditto) = row above; Sonnet/Llama resolve it.", "none (gold correct)"),
    ("he-2-00239", 2, "origin"): ("d", "missed", "ditto_unresolved", "Origin cell is 〃 (ditto) = row above; Sonnet/Llama resolve it.", "none (gold correct)"),
    ("he-2-00239", 19, "status"): ("d", "wrong_value", "copy_typo", 'Text says "ממתין לאיסוף"; model wrote "ממתיין".', "none (gold correct)"),
    ("fa-2-00072", 1, "location"): ("d", "wrong_value", "ditto_literal", 'Location cell is "همان" (=same as row above, اصفهان); no previous-location column, so the ditto reading is the only one. All models copied the literal.', "none (gold correct)"),
    ("fa-2-00159", 19, "client_org"): ("d", "wrong_value", "ditto_literal", 'Client cell "همان" = row above; model copied the literal word.', "none (gold correct)"),
    ("fa-2-00169", 25, "employer_org"): ("d", "wrong_value", "copy_typo", 'Text "نمایندگی روشن"; model wrote "نماینده روشن".', "none (gold correct)"),
    ("fa-2-00233", 15, "status"): ("d", "wrong_value", "hallucinated", 'Row says "منقضی"; model output "ممنوع" (not in the row).', "none (gold correct)"),
    # pilot
    ("en-0-00018", 0, "status"): SPAN,
    ("en-0-00042", 0, "status"): ("c", None, "status_synonym", 'Text: "successfully finalized" (status sentence) and later "paperwork officially signed". Both defensible; all 5 models say finalized.', "accept both (gold alternates) or drop slot"),
    ("he-0-00025", 0, "status"): SPAN,
    ("fa-0-00050", 0, "date"): ("d", "missed", "", "Delivery attempt date 17 مرداد 1402 is stated; Sonnet/Llama/4B extract it.", "none (gold correct)"),
    ("he-0-00014", 1, "status"): SPAN,
    ("en-0-00055", 0, "location"): SPAN,
    ("en-0-00023", 0, "status"): ("d", "wrong_value", "", 'Status is "at sea"; model took the navigation clause "actively steering toward Malmo".', "none (gold correct)"),
    ("fa-0-00077", 0, "date"): ("c", None, "event_vs_definition", 'Date is the refit completion/reopening date; slot = arrival/departure date of the current voyage. Null is defensible.', "drop slot or set null"),
    ("en-0-00086", 1, "status"): SPAN,
    ("en-0-00006", 0, "status"): SPAN,
    ("en-0-00062", 0, "status"): SPAN,
    ("en-0-00062", 0, "operator_org"): ("c", None, "role_unclear", 'Vessel "managed under our charter" (sender = Fleet Operations Logistics); Summit Freight only "supervises cargo handling". Operator is not clearly stated.', "drop slot (set null)"),
    ("en-0-00070", 1, "status"): SPAN,
    ("he-0-00061", 0, "status"): ("b", None, "status_missed_in_gold", '"המכונית מוצגת כעת לבדיקה עבור קונים פוטנציאליים" is the current status; Sonnet/4B/FP8 agree.', "set status", "מוצגת כעת לבדיקה עבור קונים פוטנציאליים"),
    ("fa-0-00046", 1, "supplier_org"): ("c", None, "role_unstated", 'Contract "between نمایندگی آسمان and شرکت نمایندگی نوین": roles (supplier/client) not stated; null is defensible.', "drop slot (set null)"),
    ("fa-0-00046", 1, "client_org"): ("c", None, "role_unstated", "Same sentence; roles not stated.", "drop slot (set null)"),
    ("he-0-00045", 0, "location"): SPAN,
    ("he-0-00090", 0, "date"): ("c", None, "inspection_vs_delivery_date", "1 April 2025 is the date of the inspection that found it was returned to sender, not a delivery date.", "drop slot (set null)"),
    ("he-0-00041", None, "*extra_record*"): ("c", None, "vague_second_entity", 'Extra record for "the active checking account" mentioned in passing (status פעיל); not an identified entity; main_entity filled with the type name.', "none; keep gold"),
    ("en-0-00014", 0, "location"): ("a", None, "previous_location", 'Car "had been stationed outside the Bristol industrial park ... before it vanished" (stolen): previous location; slot = current location.', "set null", None),
    ("he-0-00040", 0, "client_org"): ("d", "wrong_value", "copy_typo", 'Text "מוסכי כרמל"; model wrote "מוסקי כרמל".', "none (gold correct)"),
    ("fa-0-00082", 0, "location"): SPAN,
    ("fa-0-00080", 0, "holder"): ("d", "missed", "", "Holder (Ms. Maryam Tehrani manages the fund) is stated.", "none (gold correct)"),
    ("he-0-00083", 0, "location"): ("a", None, "previous_location", 'Projector was used in Holon but "הוחזר למחסן" (returned to the warehouse) now; current location is the warehouse.', "set to warehouse", "מחסן"),
    ("he-0-00083", 0, "status"): SPAN,
    ("fa-0-00061", 0, "location"): ("c", None, "status_as_location", '"در تعمیرگاه" is phrased as the status; all strong models also fill location=تعمیرگاه. Both defensible.', "accept either"),
    ("en-0-00092", 0, "status"): SPAN,
}
for r in (7, 8, 9, 12, 14, 17):
    V27[("fa-2-00019", r, "operator_org")] = ("d", "wrong_value", "ditto_literal", 'Operator cell "همان" (=row above); no previous-operator column. Model copied the literal; Sonnet/Llama resolve it.', "none (gold correct)")
for r in (13, 14, 15):
    V27[("fa-2-00159", r, "client_org")] = ("d", "missed", "ditto_unresolved", 'Client cell "همان" = row above (شرکت نوین); model left null.', "none (gold correct)")
for r in range(16, 26):
    V27[("fa-2-00233", r, "client_org")] = ("c", None, "ditto_vs_prev_column", '"همان" in the client column sits next to a "previous client" column: "same as the row above" (gold) vs "same as previous client" (all 4 models) are both readable.', "remove doc or regenerate without ditto next to a prev-column")
for r in range(3):
    V27[("fa-0-00052", r, "destination")] = ("b", None, "destination_in_text", 'Text states the listed shipments are headed to the central warehouse ("انبار مرکزی"); gold left destination null.', "set destination", "انبار مرکزی")
for r in range(3, 6):
    V27[("fa-2-00236", r, "status")] = ("d", "decoy", "previous_column", 'Current status cell is "-"; model took the "previous info" status column.', "none (gold correct)")

# ---- Qwen3-4B SFT vs gold, slots where BOTH Sonnet and Llama match gold (pilot sample of 40, manual)
V4 = {
    ("en-0-00008", 1, "client_org"): ("d", "missed", "header_propagation", "Client given once for the list; 4B left null.", "none (gold correct)"),
    ("en-0-00008", 3, "client_org"): ("d", "missed", "header_propagation", "Client given once for the list; 4B left null.", "none (gold correct)"),
    ("en-0-00009", 1, "location"): SPAN[:3] + ("Gold is the full address, 4B gives only the city (sub-span).", SPAN[4]),
    ("en-0-00024", 0, "date"): ("a", None, "inspection_vs_event_date", '26 Jan 2027 is the date of "structural evaluations"; slot = arrival/departure date of the current voyage. 4B (and INT4) null is right.', "set null", None),
    ("en-0-00026", 0, "carrier_org"): ("d", "missed", "", "Carrier stated.", "none (gold correct)"),
    ("en-0-00026", 0, "destination"): ("d", "wrong_value", "hallucinated", '"hospital set props" is not a destination.', "none (gold correct)"),
    ("en-0-00026", 0, "status"): ("d", "decoy", "cross_record", "Took the other shipment's status (held at customs).", "none (gold correct)"),
    ("en-0-00026", 1, "destination"): ("d", "missed", "", "", "none (gold correct)"),
    ("en-0-00026", 1, "status"): ("d", "missed", "", "", "none (gold correct)"),
    ("en-0-00047", 0, "date"): ("a", None, "cancelled_travel_date", 'USER FINDING CONFIRMED: "after further review of the upcoming travel calendar for 17 Jul 2025 ... the seminar ... is officially cancelled". 17 Jul 2025 is the travel-calendar date, never stated as the workshop date. 4B null is right; Sonnet/Llama/INT4 copy gold.', "set null", None),
    ("en-0-00065", 0, "location"): ("d", "missed", "", "Event site (the Boise depot, reserved for the session) is stated.", "none (gold correct)"),
    ("en-0-00071", 0, "status"): ("d", "wrong_value", "", '"active" instead of "at sea".', "none (gold correct)"),
    ("en-0-00086", 1, "status"): SPAN,
    ("en-0-00090", 0, "id_number"): ("d", "missed", "", "", "none (gold correct)"),
    ("fa-0-00012", 0, "status"): ("d", "decoy", "earlier_status", "Took a follow-up phrase instead of the latest status (postponed).", "none (gold correct)"),
    ("fa-0-00016", 0, "assigned_to"): ("d", "wrong_value", "role_not_name", "Gave the job title instead of the person.", "none (gold correct)"),
    ("fa-0-00027", 2, "origin"): ("d", "decoy", "", "Took another location (کرج).", "none (gold correct)"),
    ("fa-0-00028", 0, "supplier_org"): ("d", "decoy", "role_swap", "Took the other party.", "none (gold correct)"),
    ("fa-0-00038", 0, "status"): ("d", "normalization_format", "span_partial", '"متوقف گردیده" drops "در گمرک" and adds a verb.', "none (gold correct)"),
    ("fa-0-00044", 0, "role"): ("d", "decoy", "null_slot_filled", '"مشغول به فعالیت" is a status phrase, not a role.', "none (gold correct)"),
    ("fa-0-00060", 0, "client_org"): ("d", "wrong_value", "copy_typo", "زاگرس -> زگرس.", "none (gold correct)"),
    ("fa-0-00074", 0, "location"): ("d", "wrong_value", "copy_typo", "صنعتی -> صنایی.", "none (gold correct)"),
    ("fa-0-00082", 0, "location"): ("d", "missed", "", "", "none (gold correct)"),
    ("fa-0-00090", 0, "supplier_org"): ("d", "missed", "", "", "none (gold correct)"),
    ("he-0-00030", 0, "role"): ("d", "wrong_value", "copy_typo", "מנהלת -> מנהל (gender form not copied verbatim).", "none (gold correct)"),
    ("he-0-00043", 0, "status"): ("d", "wrong_value", "hallucinated", 'Garbled mixed-script "ה Chad to occur" where gold is null.', "none (gold correct)"),
    ("he-0-00048", 0, "branch_location"): ("d", "decoy", "previous_location", "Rehovot is the previous office.", "none (gold correct)"),
    ("he-0-00062", 0, "location"): ("d", "wrong_value", "copy_typo", "מודיעין -> הודין.", "none (gold correct)"),
    ("he-0-00062", 1, "status"): ("d", "decoy", "earlier_status", "Took an earlier status.", "none (gold correct)"),
    ("he-0-00075", 0, "origin"): ("d", "wrong_value", "copy_typo", 'Transliterated "moidhin" instead of copying מודיעין.', "none (gold correct)"),
    ("he-0-00077", 0, "port"): ("d", "missed", "", "", "none (gold correct)"),
    ("he-0-00088", 1, "employer_org"): ("d", "decoy", "adjacent_column", "Employer cell empty; took the insurer column.", "none (gold correct)"),
    ("he-0-00088", 2, "employer_org"): ("d", "decoy", "adjacent_column", "Employer cell empty; took the insurer column.", "none (gold correct)"),
    ("he-0-00091", 0, "bank_org"): ("d", "decoy", "header_propagation", 'Caption "accounts belong to בנק אופק"; took the insurer column.', "none (gold correct)"),
    ("he-0-00091", 1, "bank_org"): ("d", "decoy", "header_propagation", 'Caption "accounts belong to בנק אופק"; took the insurer column.', "none (gold correct)"),
    ("he-0-00094", 2, "employer_org"): ("d", "decoy", "cross_record", "Org of another person.", "none (gold correct)"),
}
for k in (("fa-0-00020", 1, "date"), ("fa-0-00020", 1, "organizer_org"), ("fa-0-00020", 1, "status"), ("fa-0-00071", 1, "date")):
    V4[k] = ("d", "missed", "record_omitted", "Second event record not extracted.", "none (gold correct)")


def jl(p):
    return [json.loads(l) for l in open(os.path.join(ROOT, p), encoding="utf-8")]


def auto_4b(x, row):
    """Rule-based verdict for the 421 grpo_dev 4B errors (both Sonnet and Llama agree with gold => gold presumed right)."""
    g, p, typ = x["gold"], x["pred"], x["type"]
    text = row["text"]
    decoys = json.loads(row["decoys"])
    others = [r.get(x["slot"]) for j, r in enumerate(json.loads(row["gt"])["records"]) if j != x["rec"]]
    if x["record_unaligned"]:
        return ("d", "missed", "record_omitted", "Whole gold record not extracted (not truncation: all chunks closed </think>).", "none (gold correct)")
    if p is not None and norm(p) == "":
        if g is None:
            return ("e", None, "placeholder_as_value", f'Pred is the placeholder "{p}" for an empty cell; semantically null but scored as a value.', "scorer: map '-', '—', '–' to null in _records()")
        return ("d", "missed", "placeholder", "Placeholder dash instead of the value.", "none (gold correct)")
    if p is None:
        return ("d", "missed", "", "", "none (gold correct)")
    if str(p).strip() in ('"', "〃", "همان", "״", "שם", "ditto"):
        return ("d", "wrong_value", "ditto_literal", "Copied the ditto mark literally instead of resolving it from the row above.", "none (gold correct)")
    if any(d.get("type") == typ and match(p, d.get("value"), typ) for d in decoys):
        return ("d", "decoy", "planted_distractor", "Pred equals a planted distractor.", "none (gold correct)")
    if any(o is not None and match(p, o, typ) for o in others):
        return ("d", "decoy", "cross_record", "Pred belongs to another record.", "none (gold correct)")
    if g is None:
        return ("d", "decoy", "null_slot_filled", "Slot not stated for this entity; pred filled from elsewhere.", "none (gold correct)")
    if not in_text(p, norm(text), compact(text)):
        return ("d", "wrong_value", "copy_typo", "Pred not in the document (misspelled/transliterated/invented).", "none (gold correct)")
    if typ == "date" and parse_date(str(p)) and parse_date(str(g)):
        return ("d", "wrong_value", "other_date", "Different in-text date.", "none (gold correct)")
    ng, np_ = norm(g), norm(p)
    if ng and np_ and (ng in np_ or np_ in ng):
        return SPAN[:3] + ("Gold and pred are nested spans of the same value.", SPAN[4])
    return ("d", "wrong_value", "", "Other in-text value.", "none (gold correct)")


def main():
    out, patches, remove = [], [], set()
    gold_nvals = {}
    for name, data in (("grpo_dev", "data/stage_2_RL_GRPO/grpo_dev.jsonl"), ("pilot", "data/pilot/export/pilot_all.jsonl")):
        rows = {r["id"]: r for r in jl(data)}
        # ---- 27B
        diffs = jl(f"outputs/audit/diffs_{name}.jsonl")
        seen_rec = set()
        for x in diffs:
            if x["record_unaligned"] and x["slot"] != "*extra_record*":  # whole missed record -> one line
                key = (x["id"], x["rec"])
                if key in seen_rec:
                    continue
                seen_rec.add(key)
                g = json.loads(rows[x["id"]]["gt"])["records"][x["rec"]]
                n = sum(1 for k in json.loads(rows[x["id"]]["slot_types"]) if g.get(k) is not None)
                v = ("d", "missed", "truncation", f"Whole record missed: its chunk's thinking hit max_tokens (12288) without </think>; {n} gold values lost.", "none (gold correct); inference: raise max_tokens / shorter chunks")
                out.append(dict(set=name, model=INT4, id=x["id"], rec=x["rec"], slot="*record*", gold=g.get("main_entity"),
                                pred=None, sonnet=None, n_values=n, category=v[0], category_name=CAT[v[0]], subtype=v[1],
                                detail=v[2], reason=v[3], proposed_fix=v[4], method="manual"))
                continue
            v = V27[(x["id"], x["rec"], x["slot"])]
            out.append(dict(set=name, model=INT4, id=x["id"], rec=x["rec"], slot=x["slot"], gold=x["gold"], pred=x["pred"],
                            sonnet=x.get("sonnet"), llama70b=x.get("llama70b"), n_values=1, category=v[0], category_name=CAT[v[0]],
                            subtype=v[1], detail=v[2], reason=v[3], proposed_fix=v[4], method="manual",
                            **({"scorer_code": SPANLOC} if v[0] == "e" else {})))
            if v[0] in "ab" and len(v) > 5:
                patches.append(dict(set=name, id=x["id"], rec=x["rec"], slot=x["slot"], old=x["gold"], new=v[5], category=v[0], reason=v[3], found_by=INT4))
        # ---- 4B
        for x in jl(f"outputs/audit/diffs4b_{name}.jsonl"):
            key = (x["id"], x["rec"], x["slot"])
            if name == "pilot":
                v, method = V4[key], "manual"
            else:
                v, method = auto_4b(x, rows[x["id"]]), "rule"
            out.append(dict(set=name, model=B4, id=x["id"], rec=x["rec"], slot=x["slot"], gold=x["gold"], pred=x["pred"],
                            sonnet=x.get("sonnet"), llama70b=x.get("llama70b"), int4_27b=x.get("int4_27b"), n_values=1,
                            category=v[0], category_name=CAT[v[0]], subtype=v[1], detail=v[2], reason=v[3], proposed_fix=v[4],
                            method=method, **({"scorer_code": SPANLOC if v[2] == "span_boundary" else "re_synth/score.py _records() null-token list"} if v[0] == "e" else {})))
            if v[0] in "ab" and len(v) > 5 and not any(p["id"] == x["id"] and p["rec"] == x["rec"] and p["slot"] == x["slot"] for p in patches):
                patches.append(dict(set=name, id=x["id"], rec=x["rec"], slot=x["slot"], old=x["gold"], new=v[5], category=v[0], reason=v[3], found_by=B4))
    # date-pattern scan (cancelled/postponed/travel/as-of dates) over all 298 pilot gold dates: 26 candidate windows read
    out.append(dict(set="pilot", model="all(gold-only scan)", id="he-0-00038", rec=0, slot="date", gold="26/07/2023", pred="26/07/2023",
                    sonnet="26/07/2023", n_values=1, category="c", category_name=CAT["c"], subtype=None, detail="as_of_status_date",
                    reason='"נכון לתאריך 26/07/2023 הגעתה נדחית" = date the delay was reported (as-of), not the arrival/departure date; all 4 models copy it, so no error is scored.',
                    proposed_fix="drop slot (set null)", method="manual"))
    for o in out:
        if o["category"] == "c" and o["detail"] == "ditto_vs_prev_column":
            remove.add((o["set"], o["id"], "whole doc: ambiguous ditto next to a previous-value column (8 client_org slots)"))
        elif o["category"] == "c" and o["model"] != B4 and o["detail"] != "vague_second_entity":
            remove.add((o["set"], o["id"], f"slot {o['slot']} (rec {o['rec']}): {o['detail']} -> drop/null the slot or accept alternates"))
    with open(f"{A}/label_noise_audit.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(f"{A}/gold_patches.jsonl", "w", encoding="utf-8") as f:
        for p in patches:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump([{"set": s, "id": i, "why": w} for s, i, w in sorted(remove)], open(f"{A}/remove_ids.json", "w"), ensure_ascii=False, indent=1)
    write_corrected(patches)
    print(len(out), "audit lines;", len(patches), "patches;", len(remove), "removal/drop recs")
    print(Counter((o["set"], o["model"], o["category"]) for o in out))


def write_corrected(patches):
    """Corrected copies of the pilot eval files (export rows have gt as a JSON string, samples.jsonl as a dict)."""
    by = {}
    for p in patches:
        by.setdefault((p["set"], p["id"]), []).append(p)
    for name, src, dst, as_str in (("pilot", "data/pilot/export/pilot_all.jsonl", "data/pilot/export/pilot_all_corrected.jsonl", True),
                                   ("pilot", "data/pilot/samples.jsonl", "data/pilot/samples_corrected.jsonl", False),
                                   ("grpo_dev", "data/stage_2_RL_GRPO/grpo_dev.jsonl", "data/stage_2_RL_GRPO/grpo_dev_corrected.jsonl", True)):
        if not any(s == name for s, _ in by):
            continue
        with open(os.path.join(ROOT, dst), "w", encoding="utf-8") as f:
            for r in jl(src):
                ps = by.get((name, r["id"]))
                if ps:
                    gt = json.loads(r["gt"]) if as_str else r["gt"]
                    for p in ps:
                        assert gt["records"][p["rec"]].get(p["slot"]) == p["old"], p
                        gt["records"][p["rec"]][p["slot"]] = p["new"]
                    r["gt"] = json.dumps(gt, ensure_ascii=False) if as_str else gt
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
