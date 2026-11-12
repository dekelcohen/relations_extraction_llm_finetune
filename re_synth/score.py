"""Scoring: normalization (he/fa/en), value matching (ids, dates incl. Jalali), record alignment, metrics, GRPO reward.

Pure python, no deps; must never raise on model output.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import date

from .lexicon import EN_MONTHS, FA_G_MONTHS, FA_J_MONTHS, HE_MONTHS, j2g

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHARS = str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه", "\u0640": None, "\u200c": " ", "\u200f": None, "\u200e": None,
                        "״": '"', "׳": "'"})
_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_PUNCT = re.compile(r"[\"'`“”‘’«»,.;:!?()\[\]{}<>،؛؟|*#•·–—]+")
HE_PREFIX = "בלמהוכש"


def norm(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).translate(_DIGITS).translate(_CHARS)
    s = _NIQQUD.sub("", s).lower()
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s) -> str:
    return re.sub(r"[\s\-_/.\\]", "", norm(s)).replace("imo", "")


# --------------------------------------------------------------------------- dates
_MONTHS: dict[str, tuple[int, str]] = {}
for _i, _m in enumerate(EN_MONTHS):
    _MONTHS[_m.lower()] = (_i + 1, "g")
    _MONTHS[_m.lower()[:3]] = (_i + 1, "g")
for _i, _m in enumerate(HE_MONTHS):
    _MONTHS[_m] = (_i + 1, "g")
_MONTHS["מרס"] = (3, "g")
for _i, _m in enumerate(FA_G_MONTHS):
    _MONTHS[norm(_m)] = (_i + 1, "g")
for _i, _m in enumerate(FA_J_MONTHS):
    _MONTHS[norm(_m)] = (_i + 1, "j")


def _mk(y, m, d, cal="g") -> str | None:
    try:
        y, m, d = int(y), int(m), int(d)
        if y < 100:
            y += 2000
        if cal == "j" or 1300 <= y <= 1500:
            y, m, d = j2g(y, m, d)
        return date(y, m, d).isoformat()
    except Exception:
        return None


def parse_date(s) -> str | None:
    """Best-effort -> ISO (Gregorian). Jalali (year 1300-1500 or Jalali month names) converted. None if unknown."""
    if not s:
        return None
    t = unicodedata.normalize("NFKC", str(s)).translate(_DIGITS).translate(_CHARS).strip()
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
    if m:
        return _mk(*m.groups())
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t)
    if m:
        d, mo, y = m.groups()
        if int(mo) > 12 and int(d) <= 12:
            d, mo = mo, d
        return _mk(y, mo, d)
    words = re.findall(r"[^\W\d_]+|\d+", t.lower())
    nums = [w for w in words if w.isdigit()]
    month = None
    for w in words:
        for cand in (w, w[1:] if w[:1] in HE_PREFIX else None, norm(w)):
            if cand and cand in _MONTHS:
                month = _MONTHS[cand]
                break
        if month:
            break
    if month and len(nums) >= 2:
        y = next((n for n in nums if len(n) == 4), None)
        d = next((n for n in nums if len(n) <= 2), None)
        if y and d:
            return _mk(y, month[0], d, month[1])
    return None


# --------------------------------------------------------------------------- value matching
def _strip_he_prefix(tok: str) -> str:
    return tok[1:] if len(tok) > 2 and tok[0] in HE_PREFIX else tok


def _tok_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    return (len(a) == len(b) + 1 and a[0] in HE_PREFIX and a[1:] == b and len(b) > 1) or \
        (len(b) == len(a) + 1 and b[0] in HE_PREFIX and b[1:] == a and len(a) > 1)


def match(pred, gold, typ: str = "text") -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    if isinstance(pred, list) or isinstance(gold, list):
        pl = [x for x in (pred if isinstance(pred, list) else [pred]) if x is not None]
        gl = [x for x in (gold if isinstance(gold, list) else [gold]) if x is not None]
        return bool(pl) and all(any(match(p, g, typ) for p in pl) for g in gl) and \
            all(any(match(p, g, typ) for g in gl) for p in pl)
    pred, gold = str(pred), str(gold)
    if typ == "id":
        cp, cg = compact(pred), compact(gold)
        return bool(cp) and (cp == cg or (cg in cp and len(cg) / len(cp) >= 0.7))
    if typ == "date":
        dp, dg = parse_date(pred), parse_date(gold)
        if dp and dg:
            return dp == dg
    np_, ng = norm(pred), norm(gold)
    if not np_ or not ng:
        return False
    if np_ == ng:
        return True
    tp, tg = np_.split(), ng.split()
    if len(tp) == len(tg) and all(_tok_eq(a, b) for a, b in zip(tp, tg)):
        return True
    short, long_ = sorted([np_, ng], key=len)
    if short in long_ and len(short) / len(long_) >= 0.6:
        return True
    # span-boundary leniency: shorter token set contained in longer one (e.g. "נמל נהריה" ~ "נהריה")
    ts, tl = sorted([tp, tg], key=len)
    return len(ts) / len(tl) >= 0.33 and all(any(_tok_eq(a, b) for b in tl) for a in ts)


def in_text(val, text_norm: str, text_compact: str) -> bool:
    if val is None:
        return True
    v = norm(val)
    if not v:
        return True
    if v in text_norm or compact(val) in text_compact:
        return True
    toks = [_strip_he_prefix(t) for t in v.split()]
    return all(t in text_norm for t in toks)


# --------------------------------------------------------------------------- prediction parsing
def parse_prediction(out) -> tuple[list[dict] | None, bool]:
    """Returns (records, valid_json). Accepts str or chat-message list (TRL conversational completions)."""
    if isinstance(out, list):  # [{"role": "assistant", "content": ...}]
        out = out[-1].get("content", "") if out and isinstance(out[-1], dict) else ""
    if not isinstance(out, str):
        return None, False
    s = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i:j + 1])
            except Exception:
                obj = None
    if isinstance(obj, list):
        obj = {"records": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("records"), list):
        return None, False
    return _records(obj), True


def _records(obj: dict) -> list[dict]:
    recs = []
    for r in obj["records"]:
        if isinstance(r, dict):
            recs.append({str(k): (None if v in ("", "null", "None", [], "N/A") else v) for k, v in r.items()})
    return recs


def _escape_inner_quotes(s: str) -> str:
    """Escape `"` that can't be structural: inside a string, a quote only closes it if the next non-space char is
    one of , : } ] (or end). Fixes Hebrew acronyms written with a plain quote (בע"מ, ש"ח, תשפ"ה)."""
    out, in_str, i = [], False, 0
    while i < len(s):
        c = s[i]
        if in_str and c == "\\":
            out.append(s[i:i + 2])
            i += 2
            continue
        if c == '"':
            if not in_str:
                in_str = True
            else:
                rest = s[i + 1:].lstrip()
                if not rest or rest[0] in ",:}]":
                    in_str = False
                else:
                    out.append('\\"')
                    i += 1
                    continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_prediction_lenient(out) -> tuple[list[dict] | None, bool, str]:
    """Tolerant parse -> (records, valid, repair). repair: "" (strict parse ok), "inner_quotes", "trailing_data",
    "trailing_commas", "unclosed_think" (no answer after <think>: not repairable) or "unparseable"."""
    recs, valid = parse_prediction(out)
    if valid:
        return recs, True, ""
    if isinstance(out, list):
        out = out[-1].get("content", "") if out and isinstance(out[-1], dict) else ""
    if not isinstance(out, str):
        return None, False, "unparseable"
    if "<think>" in out and "</think>" not in out:
        return None, False, "unclosed_think"
    s = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i = s.find("{")
    if i < 0:
        return None, False, "unparseable"
    s = s[i:]
    dec = json.JSONDecoder()
    for tag, cand in (("trailing_data", s), ("inner_quotes", _escape_inner_quotes(s)),
                      ("trailing_commas", re.sub(r",\s*([}\]])", r"\1", _escape_inner_quotes(s)))):
        try:
            obj = dec.raw_decode(cand)[0]  # raw_decode ignores trailing junk (e.g. an extra "}")
        except Exception:
            continue
        if isinstance(obj, list):
            obj = {"records": obj}
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            return _records(obj), True, tag
    return None, False, "unparseable"


# --------------------------------------------------------------------------- scoring
def _rec_sim(p: dict, g: dict, slot_types: dict) -> float:
    id_slots = [k for k, t in slot_types.items() if t == "id" and g.get(k)]
    for k in id_slots:
        if p.get(k) and match(p.get(k), g[k], "id"):
            return 2.0
    tp, tg = set(norm(p.get("main_entity")).split()), set(norm(g.get("main_entity")).split())
    tok = len(tp & tg) / max(1, len(tp | tg))
    keys = [k for k in slot_types if g.get(k) is not None]
    agree = sum(match(p.get(k), g.get(k), slot_types[k]) for k in keys) / max(1, len(keys))
    return 0.5 * tok + 0.5 * agree


def align(preds: list[dict], gold: list[dict], slot_types: dict) -> list[tuple[int, int]]:
    """Greedy pred<->gold record alignment by _rec_sim (pairs below 0.3 similarity stay unaligned)."""
    pairs, used_p, used_g = [], set(), set()
    cands = sorted(((_rec_sim(p, g, slot_types), i, j) for i, p in enumerate(preds) for j, g in enumerate(gold)),
                   reverse=True)
    for s, i, j in cands:
        if s < 0.3 or i in used_p or j in used_g:
            continue
        pairs.append((i, j))
        used_p.add(i)
        used_g.add(j)
    return pairs


def score_sample(pred_records: list[dict] | None, valid: bool, gt: dict, decoys: list[dict], text: str,
                 slot_types: dict) -> dict:
    gold = gt.get("records", [])
    m = defaultdict(float)
    m["valid_json"] = float(valid)
    m["n_gold_values"] = sum(1 for g in gold for k in slot_types if g.get(k) is not None)
    preds = pred_records or []
    tn, tc = norm(text), compact(text)
    pairs = align(preds, gold, slot_types)
    used_p, used_g = {i for i, _ in pairs}, {j for _, j in pairs}
    m["rec_tp"] = len(pairs)
    m["rec_fp"] = len(preds) - len(pairs)
    m["rec_fn"] = len(gold) - len(pairs)

    def is_decoy(val, typ):
        return any(d.get("type") == typ and match(val, d.get("value"), typ) for d in decoys)

    for i, j in pairs:
        p, g = preds[i], gold[j]
        m["main_ok"] += float(match(p.get("main_entity"), g.get("main_entity"), "text"))
        for k, typ in slot_types.items():
            gv, pv = g.get(k), p.get(k)
            if gv is not None and pv is not None:
                if match(pv, gv, typ):
                    m["tp"] += 1
                else:
                    m["fp"] += 1
                    m["fn"] += 1
                    m["decoy_hits"] += float(is_decoy(pv, typ))
            elif gv is not None:
                m["fn"] += 1
            elif pv is not None:
                m["fp"] += 1
                m["null_total"] += 1
                m["decoy_hits"] += float(is_decoy(pv, typ))
            else:
                m["null_total"] += 1
                m["null_ok"] += 1
            if pv is not None:
                m["pred_values"] += 1
                m["halluc"] += float(not in_text(pv, tn, tc))
    for i, p in enumerate(preds):
        if i in used_p:
            continue
        for k, typ in slot_types.items():
            if p.get(k) is not None:
                m["fp"] += 1
                m["pred_values"] += 1
                m["halluc"] += float(not in_text(p.get(k), tn, tc))
    for j, g in enumerate(gold):
        if j not in used_g:
            m["fn"] += sum(1 for k in slot_types if g.get(k) is not None)
    m["n_decoys"] = len(decoys)
    return dict(m)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def sample_f1(m: dict) -> float:
    return prf(m.get("tp", 0), m.get("fp", 0), m.get("fn", 0))[2]


def reward_from_metrics(m: dict) -> float:
    if not m.get("valid_json"):
        return 0.0
    if m.get("rec_tp", 0) == 0 and m.get("rec_fn", 0) > 0:  # missed every gold record: no partial credit
        return 0.0
    slot_f1 = sample_f1(m)
    rec_f1 = prf(m.get("rec_tp", 0), m.get("rec_fp", 0), m.get("rec_fn", 0))[2]
    null_acc = m["null_ok"] / m["null_total"] if m.get("null_total") else 1.0
    halluc = m["halluc"] / m["pred_values"] if m.get("pred_values") else 0.0
    decoy = min(1.0, m.get("decoy_hits", 0) / max(1, m.get("n_decoys", 0))) if m.get("n_decoys") else 0.0
    r = 0.6 * slot_f1 + 0.15 * rec_f1 + 0.15 * null_acc + 0.1 * (1 - halluc) - 0.2 * decoy
    return max(0.0, min(1.0, r))


def aggregate(rows: list[dict]) -> dict:
    """rows: list of metric dicts -> summary."""
    s = defaultdict(float)
    for m in rows:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                s[k] += v
    p, r, f = prf(s["tp"], s["fp"], s["fn"])
    rp, rr, rf = prf(s["rec_tp"], s["rec_fp"], s["rec_fn"])
    n = max(1, len(rows))
    return {"n": len(rows), "slot_p": round(p, 4), "slot_r": round(r, 4), "slot_f1": round(f, 4),
            "rec_f1": round(rf, 4), "null_acc": round(s["null_ok"] / s["null_total"], 4) if s["null_total"] else None,
            "decoy_hit_rate": round(s["decoy_hits"] / s["n_decoys"], 4) if s["n_decoys"] else None,
            "halluc_rate": round(s["halluc"] / s["pred_values"], 4) if s["pred_values"] else None,
            "json_valid": round(s["valid_json"] / n, 4),
            "exact_sample": round(sum(1 for m in rows if m.get("valid_json") and sample_f1(m) >= 0.999) / n, 4)}


def slot_types_for(schema_id: str, rename: dict | None = None, inline: dict | None = None) -> dict:
    from .schemas import schema_of
    rename = rename or {}
    sch = schema_of({"schema_id": schema_id, "schema": inline} if inline else {"schema_id": schema_id})
    return {rename.get(s.name, s.name): s.type for s in sch.slots}
