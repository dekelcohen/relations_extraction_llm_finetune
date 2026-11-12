import json
import random
from datetime import date

from re_synth import lexicon as L
from re_synth.hardcases import plan_sample
from re_synth.inject import strip_markers, verify_and_ground
from re_synth.score import match, parse_date, parse_prediction, reward_from_metrics, score_sample


def test_jalali_roundtrip():
    assert L.g2j(2024, 3, 20) == (1403, 1, 1)
    assert L.g2j(2026, 2, 12) == (1404, 11, 23)
    rng = random.Random(0)
    for _ in range(500):
        d = L.random_date(rng, date(2000, 1, 1), date(2030, 12, 31))
        assert L.j2g(*L.g2j(d.year, d.month, d.day)) == (d.year, d.month, d.day)


def test_gematria():
    assert L.gematria(15) == 'ט"ו'
    assert L.gematria(23) == 'כ"ג'
    assert L.gematria(786) == 'תשפ"ו'


def test_parse_date_locales():
    assert parse_date("12/2/26") == "2026-02-12"
    assert parse_date("12 בפברואר 2026") == "2026-02-12"
    assert parse_date("۱۴۰۴/۱۱/۲۳") == "2026-02-12"
    assert parse_date("23 بهمن 1404") == "2026-02-12"
    assert parse_date("February 12, 2026") == "2026-02-12"
    assert parse_date("2026-02-12") == "2026-02-12"


def test_format_date_parses_back():
    rng = random.Random(1)
    for lang in ("he", "fa", "en"):
        for _ in range(200):
            d = L.random_date(rng)
            s, style = L.format_date(d, lang, rng)
            if style != "hebcal":
                assert parse_date(s) == d.isoformat(), (lang, style, s)


def test_match_values():
    assert match("12-345-67", "1234567", "id")
    assert match("ביהוד", "יהוד", "location")
    assert match("שירותי אורן בע\"מ", "שירותי אורן בע״מ", "org")
    assert not match("חיפה", "יהוד", "location")
    assert match(["Paris", "Lyon"], ["Lyon", "Paris"], "location")


def _gt():
    return {"records": [{"main_entity": "Mazda 3", "plate": "12-345-67", "location": "Yehud", "date": "12/2/26",
                         "status": None, "owner_org": "Hertz"}]}


TYPES = {"plate": "id", "location": "location", "date": "date", "status": "status", "owner_org": "org"}
TEXT = "Black Mazda 3 12-345-67 is parked in Yehud. Made on 1/1/1955. Hertz will pick it on 12/2/26."


def test_score_perfect_and_decoy():
    decoys = [{"type": "date", "value": "1/1/1955"}]
    recs, ok = parse_prediction("```json\n" + json.dumps(_gt()) + "\n```")
    m = score_sample(recs, ok, _gt(), decoys, TEXT, TYPES)
    assert m["tp"] == 4 and m.get("fp", 0) == 0 and m["null_ok"] == 1
    assert reward_from_metrics(m) == 1.0
    bad = _gt()
    bad["records"][0]["date"] = "1/1/1955"
    m2 = score_sample(bad["records"], True, _gt(), decoys, TEXT, TYPES)
    assert m2["decoy_hits"] == 1 and reward_from_metrics(m2) < 0.8


def test_reward_invalid_json_never_raises():
    for out in ["", "not json", "{", None, [{"role": "assistant", "content": "{\"records\": 5}"}], "[]"]:
        recs, ok = parse_prediction(out)
        assert reward_from_metrics(score_sample(recs, ok, _gt(), [], TEXT, TYPES)) <= 0.2


def test_lenient_parse_repairs_formatting_only():
    from re_synth.score import parse_prediction_lenient
    raw = '<think>x</think>\n{"records": [{"main_entity": "a", "org": "אורן בע"מ", "date": "כ"א בטבת"}]}'
    assert not parse_prediction(raw)[1]
    recs, ok, tag = parse_prediction_lenient(raw)
    assert ok and tag == "inner_quotes" and recs[0]["org"] == 'אורן בע"מ' and recs[0]["date"] == 'כ"א בטבת'
    assert parse_prediction_lenient('{"records": [{"main_entity": "a"}]}}')[1:] == (True, "trailing_data")
    assert parse_prediction_lenient("<think>never ends")[1:] == (False, "unclosed_think")
    assert parse_prediction_lenient(json.dumps(_gt()))[2] == ""


def test_marker_verify():
    rng = random.Random(3)
    plan = plan_sample(rng, "en", "text", "vehicle", force_cases=["H1"])
    r = plan["records"][0]
    body = f"The [[r0.main]]{r['main_entity']}[[/]] is here."
    for k, v in r["slots"].items():
        if v:
            body += f" Its {k} is [[r0.{k}]]{v['value']}[[/]]."
    for j, x in enumerate(plan["decoys"]):
        body += f" Unrelated: [[x{j}]]{x['value']}[[/]]."
    gt, issues, kept = verify_and_ground(plan, "Title", body)
    assert not issues, issues
    assert len(kept) == len([x for x in plan["decoys"] if x["placement"] in ("near", "between")])
    assert "[[" not in strip_markers(body)[0]


def test_plans_all_formats():
    for i in range(300):
        rng = random.Random(i)
        lang, fmt = rng.choice(["he", "fa", "en"]), rng.choice(["text", "html", "email"])
        p = plan_sample(rng, lang, fmt)
        assert p["records"] and all(r["main_entity"] for r in p["records"])


def test_match_span_leniency():
    from re_synth.score import match
    assert match("נמל נהריה", "נהריה", "location")
    assert match("נמל בלוד", "הנמל בלוד", "location")
    assert match("חברת לוגיסטיקה צור בע\"מ", "לוגיסטיקה צור בע\"מ", "org")
    assert not match("מרכז הכנסים בעיר לוד", "לוד", "location")
    assert not match("תל אביב", "חיפה", "location")


def test_rewards_accept_json_string_columns():
    import json, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from rewards import extraction_reward, format_reward
    gt = {"records": [{"main_entity": "Mazda", "location": "Yehud", "date": "12/2/26"}]}
    st = {"main_entity": "text", "location": "location", "date": "date"}
    kw = dict(gt=[json.dumps(gt)] * 2, decoys=["[]"] * 2, text=["Mazda parked in Yehud, pickup 12/2/26"] * 2,
              slot_types=[json.dumps(st)] * 2)
    assert extraction_reward(completions=[json.dumps(gt), "oops"], **kw) == [1.0, 0.0]
    assert format_reward(completions=[json.dumps(gt), "oops"]) == [1.0, 0.0]


def test_chunking_roundtrip_long_doc():
    """split -> per-chunk gold -> merge recovers the full gold of a deterministic H19 long document."""
    from re_synth import chunking, hardcases, layout
    from re_synth.schemas import SCHEMAS
    from re_synth.score import sample_f1, slot_types_for
    old, hardcases.FOCUS_LONG_P = hardcases.FOCUS_LONG_P, 1.0
    try:
        for lang, fmt in (("he", "text"), ("fa", "html")):
            rng = random.Random(f"chunk-{lang}")
            p = plan_sample(rng, lang, fmt, focus=True)
            assert "H19" in p["hard_cases"]
            s = SCHEMAS[p["schema_id"]]
            text = "Title\n\n" + layout.long_doc(s, p, lang, rng, fmt != "html") * 1
            gt = {"records": [{"main_entity": r["main_entity"],
                               **{k: (v or {}).get("value") for k, v in r["slots"].items()}} for r in p["records"]]}
            parts = chunking.split(text, max_chars=1500, overlap=300, min_chars=1000)
            assert len(parts) >= 2 and all(len(c) <= 1600 for c in parts)
            st = slot_types_for(p["schema_id"])
            golds = [json.dumps(chunking.chunk_gold(gt, [], c)[0], ensure_ascii=False) for c in parts]
            merged = json.loads(chunking.merge_raws(golds, st))["records"]
            m = score_sample(merged, True, gt, [], text, st)
            assert sample_f1(m) == 1.0, (lang, m)
    finally:
        hardcases.FOCUS_LONG_P = old


def test_chunk_merge_rules():
    from re_synth.chunking import merge_records
    st = {"plate": "id", "location": "location"}
    a = {"main_entity": "Ford", "plate": "11-222-33", "location": None}
    a2 = {"main_entity": "Ford", "plate": "11-222-33", "location": "Haifa"}   # same entity seen in the overlap
    b = {"main_entity": "Ford", "plate": "99-888-77", "location": "Eilat"}    # same name, different id
    out = merge_records([[a, b], [a2]], st)
    assert len(out) == 2 and {r["location"] for r in out} == {"Haifa", "Eilat"}
    # two same-name records in ONE chunk are never merged
    assert len(merge_records([[a, dict(a)]], st)) == 2


def test_chunk_overlap_duplicates_removed():
    """Records repeated in chunk overlaps (with surface noise) are extracted twice but merged into one."""
    from re_synth import chunking, hardcases, layout
    from re_synth.schemas import SCHEMAS
    from re_synth.score import norm, score_sample, slot_types_for
    old, hardcases.FOCUS_LONG_P = hardcases.FOCUS_LONG_P, 1.0
    try:
        rng = random.Random("dups-he")
        p = plan_sample(rng, "he", "text", focus=True)
        s = SCHEMAS[p["schema_id"]]
        text = "Title\n\n" + layout.long_doc(s, p, "he", rng, True)
        gt = {"records": [{"main_entity": r["main_entity"],
                           **{k: (v or {}).get("value") for k, v in r["slots"].items()}} for r in p["records"]]}
        st = slot_types_for(p["schema_id"])
        parts = chunking.split(text, max_chars=1200, overlap=400, min_chars=1000)
        per_chunk, seen = [], set()
        for ci, c in enumerate(parts):
            recs = chunking.chunk_gold(gt, [], c)[0]["records"]
            out = []
            for r in recs:
                r = dict(r)
                if r["main_entity"] in seen:            # overlap copy: noisy surface form, a slot left out
                    r["main_entity"] = ("ה" + r["main_entity"]) if ci % 2 else f'"{r["main_entity"]}"'
                    k = next((k for k in st if r.get(k) is not None and st[k] != "id"), None)
                    if k:
                        r[k] = None
                out.append(dict(reversed(list(r.items()))))
            seen |= {r["main_entity"] for r in recs}
            per_chunk.append(out)
        n_raw = sum(len(x) for x in per_chunk)
        assert n_raw > len(gt["records"]), "test doc must have records repeated in overlaps"
        merged = chunking.merge_records(per_chunk, st)
        keys = [norm(r["main_entity"]).strip('"') for r in merged]
        assert len(merged) == len(gt["records"]), (n_raw, len(merged), len(gt["records"]))
        m = score_sample(merged, True, gt, [], text, st)
        assert m["rec_fp"] == 0 and m["rec_fn"] == 0 and m["tp"] == m["n_gold_values"], m
        assert len(set(keys)) == len(keys) or len(gt["records"]) != len({norm(r["main_entity"]) for r in gt["records"]})
    finally:
        hardcases.FOCUS_LONG_P = old
