"""Overlapping chunking of long documents + merging/deduplicating the records extracted from the chunks.

Long docs (many entities, big tables) overflow small models' context/attention, so inference runs per chunk:
    chunks = split(text)                       # [] -> document is short, run it whole
    raws = [model(prompt(c)) for c in chunks]
    raw = merge_raws(raws, slot_types)         # one {"records": [...]} JSON, duplicates from overlaps removed

Chunks only cut between units (paragraph, list item, card, table row) and repeat the context a unit depends on:
the document preamble (title / email headers), the active section heading, and inside a table its header rows
plus the current group row (e.g. "■ bank: X" / <th colspan>). Transposed tables (records are columns) stay whole.
For training, chunk_gold() derives per-chunk gold from the document gold.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .score import _tok_eq, compact, in_text, match, norm, parse_prediction, parse_prediction_lenient

CHUNK_CHARS = 2500    # ~1.1k (he) / ~1.5k (fa) Qwen3 tokens of document per chunk
CHUNK_OVERLAP = 500   # trailing units of the previous chunk repeated at the start of the next
CHUNK_MIN = 4000      # documents up to this length are not chunked
PREAMBLE_MAX = 600


@dataclass
class Unit:
    text: str
    heading: str = ""        # active section heading (repeated in later chunks)
    table_head: str = ""     # header of the table this unit is a body row of
    group: str = ""          # active table group/section row
    is_ctx: bool = False     # the unit itself is a heading / group row
    preamble: bool = False


@dataclass
class _State:
    heading: str = ""
    table_head: str = ""
    group: str = ""
    units: list = field(default_factory=list)

    def add(self, text, **kw):
        kw.setdefault("heading", self.heading)
        kw.setdefault("table_head", self.table_head)
        kw.setdefault("group", self.group)
        self.units.append(Unit(text, **kw))


def is_html(text: str) -> bool:
    return bool(re.search(r"<(table|p|h[1-6]|ul|li|article|html)\b", text[:3000], re.I))


# --------------------------------------------------------------------------- html
_HTML_CUT = re.compile(r"(?=<(?:p|h[1-6]|table|/table|ul|ol|/ul|/ol|li|tr|div|section|blockquote|pre|footer)\b)", re.I)


def _html_units(text: str) -> list[Unit]:
    pieces = [p for p in _HTML_CUT.split(text) if p]
    st, i, seen_h1 = _State(), 0, False
    pre = []
    # preamble: everything up to and including the first <h1> (or the first piece if there is none)
    while i < len(pieces) and not seen_h1:
        if re.match(r"<h1\b", pieces[i], re.I) or not any(re.match(r"<h1\b", p, re.I) for p in pieces[i:]):
            seen_h1 = True
        pre.append(pieces[i])
        i += 1
    st.add("".join(pre), preamble=True)
    while i < len(pieces):
        p = pieces[i]
        if re.match(r"<table\b", p, re.I):
            j = i
            while j < len(pieces) and not re.match(r"</table\b", pieces[j], re.I):
                j += 1
            _html_table(st, pieces[i:j])
            if j < len(pieces):           # "</table>..." (may carry trailing text)
                st.add(pieces[j])
            i = j + 1
            continue
        m = re.match(r"<h([1-6])\b", p, re.I)
        if m and int(m.group(1)) <= 2:
            st.heading = re.sub(r"\s+$", "", p)
            st.add(p, is_ctx=True)
        elif m and i + 1 < len(pieces):   # card title (h3+) glued to its body
            st.add(p + pieces[i + 1])
            i += 1
        else:
            st.add(p)
        i += 1
    return st.units


def _html_table(st: _State, pieces: list[str]) -> None:
    body = "".join(pieces)
    rows = [p for p in pieces if re.match(r"<tr\b", p, re.I)]
    transposed = sum(bool(re.match(r"<tr>\s*<th[^>]*>[^<]*</th>\s*<td", r, re.I)) for r in rows) >= max(2, len(rows) // 2)
    if transposed or len(rows) < 3:
        st.add(body)
        return
    # header = everything before the first body row (after </thead>, or the first <tr> when there is no thead)
    hi = next((k for k, p in enumerate(pieces) if re.search(r"</thead>", p, re.I)), None)
    if hi is None:
        hi = next(k for k, p in enumerate(pieces) if re.match(r"<tr\b", p, re.I))
    head = "".join(pieces[:hi + 1])
    st.add(head, is_ctx=True)
    st.table_head, st.group = head, ""
    for p in pieces[hi + 1:]:
        if re.match(r"<tr>\s*<th[^>]*colspan", p, re.I):
            st.group = p
            st.add(p, is_ctx=True)
        else:
            st.add(p)
    st.table_head, st.group = "", ""


# --------------------------------------------------------------------------- plain text / email
_PIPE = re.compile(r"\S\s*\|\s*\S|\|\s*\S")
_SEP = re.compile(r"^\s*\|?\s*[-=:|+\s]{5,}\s*$")


def _text_units(text: str) -> list[Unit]:
    lines = text.splitlines(keepends=True)
    st, i = _State(), 0
    # preamble: email header block (until the first blank line) or the first non-empty line (title)
    if re.match(r"^(From|To|מאת|از|فرستنده|Subject|נושא)\s*:", lines[0] if lines else "", re.I):
        while i < len(lines) and lines[i].strip():
            i += 1
    else:
        while i < len(lines) and not lines[i].strip():
            i += 1
        i += 1
    pre = "".join(lines[:i])
    if len(pre) > PREAMBLE_MAX:
        pre = pre[:PREAMBLE_MAX].rsplit("\n", 1)[0] + "\n"
    st.add("".join(lines[:i]), preamble=True)
    st.units[0].text = "".join(lines[:i])
    st.units[0].heading = pre  # (re-used below as the preamble text that is repeated)
    while i < len(lines):
        ln = lines[i]
        if _PIPE.search(ln):
            j = i
            while j < len(lines) and (_PIPE.search(lines[j]) or _SEP.match(lines[j]) or
                                      (lines[j].strip() and len(lines[j]) < 90 and j + 1 < len(lines)
                                       and _PIPE.search(lines[j + 1]))):
                j += 1
            _text_table(st, lines[i:j])
            i = j
            continue
        if re.match(r"\s*#{1,3}\s", ln) or (ln.strip().endswith(":") and len(ln) < 90 and i + 1 < len(lines)
                                             and lines[i + 1].lstrip().startswith(("•", "-", "*", "–"))):
            st.heading = ln if ln.startswith("\n") or not ln.strip() else ln
            st.add(ln, is_ctx=True)
            i += 1
            continue
        # card: a line followed by indented "key: value" lines
        j = i + 1
        while j < len(lines) and lines[j].strip() and lines[j][:1] in (" ", "\t"):
            j += 1
        st.add("".join(lines[i:j]))
        i = j
    return st.units


def _text_table(st: _State, lines: list[str]) -> None:
    sep = next((k for k, ln in enumerate(lines) if _SEP.match(ln) and not re.search(r"\w", ln)), None)
    if sep is None or len(lines) - sep < 3:   # no header separator -> transposed / small: keep whole
        st.add("".join(lines))
        return
    head = "".join(lines[:sep + 1])
    st.add(head, is_ctx=True)
    st.table_head, st.group = head, ""
    for ln in lines[sep + 1:]:
        if not _PIPE.search(ln) and ln.strip():
            st.group = ln
            st.add(ln, is_ctx=True)
        else:
            st.add(ln)
    st.table_head, st.group = "", ""


# --------------------------------------------------------------------------- split
_DITTO = re.compile(r'(?:<td[^>]*>|\|)\s*(?:〃|כנ״ל|כנ"ל|همان|ditto|"|״)\s*(?:</td>|\||$)', re.M)


def units(text: str) -> list[Unit]:
    return _html_units(text) if is_html(text) else _text_units(text)


def split(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP, min_chars: int = CHUNK_MIN
          ) -> list[str]:
    """Overlapping chunks of `text`; [] when the document is short enough to be processed whole."""
    if not text or len(text) <= min_chars:
        return []
    us = units(text)
    html = is_html(text)
    preamble = us[0].heading if not html else us[0].text
    if html and len(preamble) > PREAMBLE_MAX:   # keep <title>/<h1>, drop nav/boilerplate
        t = re.findall(r"<h1\b.*?</h1>|<title>.*?</title>", preamble, re.I | re.S)
        preamble = "\n".join(t[-1:]) or preamble[-PREAMBLE_MAX:]

    def prefix(first: int, idx: list[int]) -> str:
        u = us[first]
        parts = [] if first == 0 else [preamble]
        if u.heading and not (us[first].is_ctx and us[first].text == u.heading) and not (not html and first == 0):
            if not any(us[k].text == u.heading for k in idx):
                parts.append(u.heading)
        if u.table_head and us[first].text != u.table_head:
            parts.append(u.table_head)
            if u.group and us[first].text != u.group:
                parts.append(u.group)
        out = "".join(p if p.endswith("\n") or html else p + "\n" for p in parts[:1]) + ("" if html or not parts[1:]
                                                                                    else "\n")
        out += "".join(p if p.endswith("\n") or html else p + "\n" for p in parts[1:])
        return out + ("\n" if html and out else "")

    chunks, cur = [], []
    size = lambda idx: sum(len(us[k].text) for k in idx) + len(prefix(idx[0], idx))  # noqa: E731
    for k in range(len(us)):
        if cur and size(cur + [k]) > max_chars:
            chunks.append(cur)
            tail, n = [], 0
            for t in reversed(cur):          # overlap: trailing units of the finished chunk (not context rows)
                n += len(us[t].text)
                if n > overlap or len(tail) + 1 >= len(cur):
                    break
                tail.insert(0, t)
            while tail and us[tail[0]].is_ctx:
                tail.pop(0)
            # a row whose cell says "ditto" needs the row above it in the same chunk
            while (tail[0] if tail else k) > 0 and _DITTO.search(us[tail[0] if tail else k].text) \
                    and (tail[0] if tail else k) - 1 in cur and not us[(tail[0] if tail else k) - 1].is_ctx:
                tail.insert(0, (tail[0] if tail else k) - 1)
            cur = tail if tail and size(tail + [k]) <= max_chars else []
        cur.append(k)
    if cur:
        chunks.append(cur)
    if len(chunks) < 2:
        return []
    return [(prefix(c[0], c) + "".join(us[k].text for k in c)).strip() for c in chunks]


# --------------------------------------------------------------------------- merge
def _key(v) -> str:
    return compact(json.dumps(v, ensure_ascii=False)) if isinstance(v, list) else norm(v) if v is not None else ""


def _same_name(a, b) -> bool:
    ta, tb = norm(a).strip().split(), norm(b).strip().split()
    return bool(ta) and len(ta) == len(tb) and all(_tok_eq(x, y) for x, y in zip(ta, tb))


def _same_entity(a: dict, b: dict, slot_types: dict) -> bool:
    """Same id value -> same entity (even if one chunk mangled the name); different id -> different entities;
    otherwise the names must be equal (modulo Hebrew prefixes/punctuation) and the slots mostly agree."""
    agree = conflict = 0
    for k, t in slot_types.items():
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        if t == "id":
            return match(va, vb, t)
        if match(va, vb, t):
            agree += 1
        else:
            conflict += 1
    return _same_name(a.get("main_entity"), b.get("main_entity")) and (conflict <= agree or conflict <= 1)


def merge_records(per_chunk: list[list[dict] | None], slot_types: dict, text: str | None = None) -> list[dict]:
    """Union of chunk records with overlap duplicates removed: a record is merged into an earlier one from ANOTHER
    chunk when _same_entity (records of one chunk are never merged - same-name entities stay distinct).
    Slot value = most frequent non-null; main_entity prefers a name that occurs in `text`."""
    groups: list[tuple[set, list[dict]]] = []
    for ci, recs in enumerate(per_chunk):
        for r in recs or []:
            if not isinstance(r, dict):
                continue
            g = next((g for g in groups if ci not in g[0] and _same_entity(_combine(g[1], slot_types), r,
                                                                            slot_types)), None)
            if g is None:
                groups.append(({ci}, [r]))
            else:
                g[0].add(ci)
                g[1].append(r)
    tn = norm(text) if text else None
    return [_combine(rs, slot_types, tn) for _, rs in groups]


def _vote(vals: list, prefer=None):
    cnt: dict[str, list] = {}
    for v in vals:
        cnt.setdefault(_key(v), []).append(v)
    return max(cnt.values(), key=lambda xs: (bool(prefer and prefer(xs[0])), len(xs), len(str(xs[0]))))[0] \
        if cnt else None


def _combine(rs: list[dict], slot_types: dict, text_norm: str | None = None) -> dict:
    out = {"main_entity": _vote([r.get("main_entity") for r in rs if r.get("main_entity")],
                                (lambda v: norm(v) in text_norm) if text_norm else None)}
    for k in dict.fromkeys(k for r in rs for k in r if k != "main_entity"):
        out[k] = _vote([r.get(k) for r in rs if r.get(k) is not None])
    return out


def merge_raws(raws: list[str], slot_types: dict, text: str | None = None) -> str:
    """Merge raw model outputs (one per chunk) into a single {"records": [...]} JSON string.
    Unparseable chunk outputs are skipped (an all-invalid document yields the first raw unchanged)."""
    per, ok = [], False
    for raw in raws:
        recs, valid = parse_prediction(raw)
        if not valid:
            recs, valid, _ = parse_prediction_lenient(raw)
        per.append(recs if valid else None)
        ok |= valid
    if not ok:
        return raws[0] if raws else ""
    return json.dumps({"records": merge_records(per, slot_types, text)}, ensure_ascii=False)


# --------------------------------------------------------------------------- training targets
def chunk_gold(gt: dict, decoys: list[dict], chunk: str) -> tuple[dict, list[dict], bool]:
    """Per-chunk gold: records whose main entity occurs in the chunk with the slot values that occur in it.
    Returns (gt, decoys, clean); clean=False when some record is only partially inside the chunk (or only its
    name is) - such chunks have an ambiguous target and should not be used for training."""
    tn, tc = norm(chunk), compact(chunk)
    names = {norm(r.get("main_entity")) for r in gt["records"] if r.get("main_entity")}
    cand = []
    for r in gt["records"]:
        main = r.get("main_entity")
        if not main:
            continue
        nm = norm(main)
        # occurrences inside a longer gold name (e.g. "Skoda Octavia" in "Skoda Octavia red") do not count
        t = tn
        for other in sorted((n for n in names if nm in n and n != nm), key=len, reverse=True):
            t = t.replace(other, " ")
        if nm not in t and (compact(main) not in tc or t != tn):
            continue
        vals = {k: v for k, v in r.items() if k != "main_entity" and v is not None}
        inside = {k for k, v in vals.items() if _present(v, tn, tc)}
        cand.append((r, nm, vals, inside, len(inside) / len(vals) if vals else 1.0))
    recs, clean = [], True
    for r, nm, vals, inside, ratio in cand:
        # same-name records (told apart by id): skip the twin whose values are not in this chunk
        if ratio < 0.5 and any(o[1] == nm and o[0] is not r and o[4] >= 0.5 for o in cand):
            continue
        if len(inside) < len(vals):
            clean = False
        recs.append({"main_entity": r["main_entity"],
                     **{k: (r[k] if k in inside else None) for k in r if k != "main_entity"}})
    dec = [d for d in decoys if _present(d.get("value"), tn, tc)]
    return {"records": recs}, dec, clean


def _present(v, tn: str, tc: str) -> bool:
    if isinstance(v, list):
        return all(_present(x, tn, tc) for x in v)
    return v is not None and (norm(v) in tn or compact(v) in tc or in_text(v, tn, tc) and len(norm(v)) > 3)
