"""Deterministic layout: plain text / HTML / email wrappers and H6 (subtitle->list) / H7 (table) blocks."""
from __future__ import annotations

import html
import random

from . import lexicon as L
from .schemas import Schema, gen_value

LABELS = {  # slot/column labels per language
    "main": {"he": "פריט", "fa": "مورد", "en": "Item"},
    "plate": {"he": "מספר רישוי", "fa": "شماره پلاک", "en": "Plate"},
    "tracking_id": {"he": "מספר מעקב", "fa": "کد رهگیری", "en": "Tracking no."},
    "id_number": {"he": "מספר זהות", "fa": "کد ملی", "en": "ID"},
    "serial": {"he": "מספר סידורי", "fa": "شماره سریال", "en": "Serial"},
    "contract_id": {"he": "מספר חוזה", "fa": "شماره قرارداد", "en": "Contract no."},
    "account_number": {"he": "מספר חשבון", "fa": "شماره حساب", "en": "Account no."},
    "imo": {"he": "מספר IMO", "fa": "شماره IMO", "en": "IMO"},
    "location": {"he": "מיקום", "fa": "مکان", "en": "Location"},
    "origin": {"he": "מוצא", "fa": "مبدأ", "en": "Origin"},
    "destination": {"he": "יעד", "fa": "مقصد", "en": "Destination"},
    "port": {"he": "נמל", "fa": "بندر", "en": "Port"},
    "branch_location": {"he": "סניף", "fa": "شعبه", "en": "Branch"},
    "date": {"he": "תאריך", "fa": "تاریخ", "en": "Date"},
    "start_date": {"he": "תאריך תחילת עבודה", "fa": "تاریخ شروع کار", "en": "Start date"},
    "status": {"he": "סטטוס", "fa": "وضعیت", "en": "Status"},
    "owner_org": {"he": "בעלים", "fa": "مالک", "en": "Owner"},
    "carrier_org": {"he": "מוביל", "fa": "شرکت حمل", "en": "Carrier"},
    "employer_org": {"he": "מעסיק", "fa": "کارفرما", "en": "Employer"},
    "organizer_org": {"he": "מארגן", "fa": "برگزارکننده", "en": "Organizer"},
    "supplier_org": {"he": "ספק", "fa": "تأمین‌کننده", "en": "Supplier"},
    "client_org": {"he": "לקוח", "fa": "کارفرما", "en": "Client"},
    "operator_org": {"he": "מפעיל", "fa": "بهره‌بردار", "en": "Operator"},
    "bank_org": {"he": "בנק", "fa": "بانک", "en": "Bank"},
    "role": {"he": "תפקיד", "fa": "سمت", "en": "Role"},
    "assigned_to": {"he": "מחזיק", "fa": "تحویل‌گیرنده", "en": "Assigned to"},
    "holder": {"he": "בעל החשבון", "fa": "صاحب حساب", "en": "Holder"},
    "value": {"he": "סכום", "fa": "مبلغ", "en": "Value"},
}
DISTRACTOR_COLS = {  # (label, value-type): plausible but NOT the requested slot
    "date": [{"he": "תאריך ייצור", "fa": "تاریخ تولید", "en": "Manufactured"},
             {"he": "עדכון קודם", "fa": "به‌روزرسانی قبلی", "en": "Previous update"}],
    "location": [{"he": "מיקום קודם", "fa": "مکان قبلی", "en": "Previous location"},
                 {"he": "מקום ייצור", "fa": "محل تولید", "en": "Made in"}],
    "org": [{"he": "יצרן", "fa": "سازنده", "en": "Manufacturer"}, {"he": "מבטח", "fa": "بیمه‌گر", "en": "Insurer"}],
}
PLURAL = {
    "vehicle": {"he": "הרכבים", "fa": "خودروهای", "en": "vehicles"},
    "shipment": {"he": "המשלוחים", "fa": "محموله‌های", "en": "shipments"},
    "person_employment": {"he": "העובדים", "fa": "کارکنان", "en": "employees"},
    "asset": {"he": "הפריטים", "fa": "اقلام", "en": "items"},
    "contract": {"he": "החוזים", "fa": "قراردادهای", "en": "contracts"},
    "account": {"he": "החשבונות", "fa": "حساب‌های", "en": "accounts"},
    "vessel": {"he": "כלי השיט", "fa": "کشتی‌های", "en": "vessels"},
    "meeting": {"he": "האירועים", "fa": "رویدادهای", "en": "events"},
}
SUBTITLE = {"he": "{p} הבאים שייכים ל{org}:", "fa": "{p} زیر متعلق به {org} هستند:",
            "en": "The following {p} belong to {org}:"}


def label(name: str, lang: str) -> str:
    return LABELS.get(name, {}).get(lang, name)


def _sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?؟。])\s+|\n+", text) if 25 < len(s.strip()) < 220]


def h6_block(schema: Schema, records: list[dict], lang: str, seed_text: str, rng: random.Random) -> tuple[str, list[str]]:
    """Subtitle with container org, 2-3 irrelevant lines, then '<main> <id>' lines. Returns (block, lines)."""
    org = records[0]["slots"][schema.container_slot]["value"]
    filler = rng.sample(_sentences(seed_text) or ["..."], k=min(rng.randint(2, 3), max(1, len(_sentences(seed_text)))))
    items = [f"{r['main_entity']} {r['slots'][schema.id_slot]['value']}" for r in records]
    sub = SUBTITLE[lang].format(p=PLURAL[schema.id][lang], org=org)
    return sub, filler, items


def h7_table(schema: Schema, records: list[dict], lang: str, rng: random.Random, caption_org: bool) -> tuple[str, list[dict]]:
    """HTML table: GT columns + one distractor column. Returns (html, decoys)."""
    cols = [s for s in schema.slots if not (caption_org and s.name == schema.container_slot)]
    dtype = rng.choice([t for t in DISTRACTOR_COLS if any(s.type == t for s in schema.slots)] or ["date"])
    dlabel = rng.choice(DISTRACTOR_COLS[dtype])[lang]
    dslot = next((s for s in schema.slots if s.type == dtype), None)
    pos = rng.randint(1, len(cols))
    decoys, rows = [], []
    for i, r in enumerate(records):
        dval = gen_value(schema, dslot, rng, lang)["value"] if dslot else L.format_date(L.random_date(rng), lang, rng)[0]
        decoys.append({"slot": dslot.name if dslot else "date", "type": dtype, "value": dval,
                       "role": f"table column '{dlabel}'", "record": i, "placement": "table"})
        cells = [r["main_entity"]] + [(r["slots"][s.name] or {}).get("value", "") for s in cols]
        cells.insert(pos + 1, dval)
        rows.append(cells)
    heads = [label("main", lang)] + [label(s.name, lang) for s in cols]
    heads.insert(pos + 1, dlabel)
    e = lambda x: html.escape(x or "", quote=False)
    cap = ""
    if caption_org:
        org = records[0]["slots"][schema.container_slot]["value"]
        cap = f"<caption>{e(SUBTITLE[lang].format(p=PLURAL[schema.id][lang], org=org))}</caption>"
    t = (f"<table border=\"1\">{cap}<thead><tr>" + "".join(f"<th>{e(h)}</th>" for h in heads) + "</tr></thead><tbody>"
         + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>")
    return t, decoys


NAV = {"he": ["דף הבית", "חדשות", "צור קשר", "אודות"], "fa": ["صفحه اصلی", "اخبار", "تماس با ما", "درباره ما"],
       "en": ["Home", "News", "Contact", "About"]}
UPDATED = {"he": "עודכן לאחרונה", "fa": "آخرین به‌روزرسانی", "en": "Last updated"}


def to_html(title: str, paras: list[str], lang: str, rng: random.Random, block_html: str | None = None
            ) -> tuple[str, list[dict]]:
    """Wrap paragraphs into a page with nav + footer (footer date = decoy). '[[BLOCK]]' para -> block_html."""
    e = lambda x: html.escape(x, quote=False)
    d = "rtl" if lang in ("he", "fa") else "ltr"
    fdate = L.format_date(L.random_date(rng), lang, rng)[0]
    body = []
    for p in paras:
        if p.strip() == "[[BLOCK]]":
            body.append(block_html or "")
        elif p.startswith("<"):
            body.append(p)
        else:
            body.append(f"<p>{e(p)}</p>")
    page = (f"<html lang=\"{lang}\" dir=\"{d}\"><head><meta charset=\"utf-8\"><title>{e(title)}</title></head><body>"
            f"<nav>{' | '.join(f'<a href=#>{e(n)}</a>' for n in NAV[lang])}</nav>"
            f"<article><h1>{e(title)}</h1>\n" + "\n".join(body) + "</article>"
            f"<footer>{e(UPDATED[lang])}: {e(fdate)} · © {e(L.org(rng, lang))}</footer></body></html>")
    return page, [{"slot": "date", "type": "date", "value": fdate, "role": "page footer update date", "record": -1,
                   "placement": "footer"}]


WROTE = {"he": "ב-{d}, {p} כתב/ה:", "fa": "در تاریخ {d}، {p} نوشت:", "en": "On {d}, {p} wrote:"}
HDR = {"he": ("מאת", "אל", "עותק", "תאריך", "נושא"), "fa": ("از", "به", "رونوشت", "تاریخ", "موضوع"),
       "en": ("From", "To", "Cc", "Date", "Subject")}


def to_email(subject: str, body: str, lang: str, rng: random.Random, quoted: tuple[str, str] | None = None
             ) -> tuple[str, list[dict]]:
    """Long header block (subject far from body), optional quoted previous message with a decoy date (H5)."""
    f, t, cc, dt, sj = HDR[lang]
    mail = lambda: f"{rng.choice(['ops', 'info', 'desk', 'admin', 'team'])}{rng.randint(1, 99)}@{rng.choice(['corp', 'mail', 'biz'])}.example"
    send_date = L.format_date(L.random_date(rng), lang, rng)[0]
    lines = [f"{f}: {L.person(rng, lang)} <{mail()}>", f"{t}: {L.person(rng, lang)} <{mail()}>",
             f"{cc}: {', '.join(mail() for _ in range(rng.randint(1, 4)))}", f"{sj}: {subject}",
             f"{dt}: {send_date}", f"Message-ID: <{rng.getrandbits(64):x}@mail.example>",
             f"X-Mailer: {rng.choice(['Outlook 16.0', 'Thunderbird 128', 'Gmail'])}",
             f"Thread-Index: {rng.getrandbits(96):x}", f"X-Priority: {rng.randint(1, 5)}",
             f"X-MS-Has-Attach: {rng.choice(['yes', 'no'])}", f"Content-Language: {lang}"]
    rng.shuffle(lines[5:])
    decoys = [{"slot": "date", "type": "date", "value": send_date, "role": "email send date header", "record": -1,
               "placement": "header"}]
    text = "\n".join(lines) + "\n\n" + body
    if quoted:
        qdate, qline = quoted
        text += "\n\n" + WROTE[lang].format(d=qdate, p=L.person(rng, lang)) + "\n> " + qline
        decoys.append({"slot": "date", "type": "date", "value": qdate, "role": "date of a previous message in the thread",
                       "record": -1, "placement": "quoted"})
    return text, decoys


# ---- large-table family (H13-H18) ----
DIST_LABEL = {  # label of a same-type distractor column (H13); '{l}' = the GT column label
    "prev": {"he": "{l} קודם", "fa": "{l} قبلی", "en": "Previous {l}"},
    "location": [{"he": "מקום ייצור", "fa": "محل تولید", "en": "Made in"},
                 {"he": "כתובת משרד ראשי", "fa": "آدرس دفتر مرکزی", "en": "HQ address"}],
    "date": [{"he": "תאריך ייצור", "fa": "تاریخ تولید", "en": "Manufactured"},
             {"he": "תאריך בדיקה אחרונה", "fa": "تاریخ آخرین بازرسی", "en": "Last inspection"}],
    "org": [{"he": "יצרן", "fa": "سازنده", "en": "Manufacturer"}, {"he": "מבטח", "fa": "بیمه‌گر", "en": "Insurer"},
            {"he": "קבלן משנה", "fa": "پیمانکار فرعی", "en": "Subcontractor"}],
    "status": [{"he": "סטטוס בדוח הקודם", "fa": "وضعیت در گزارش قبلی", "en": "Status in last report"}],
    "id": [{"he": "מספר ישן", "fa": "شماره قدیمی", "en": "Old number"}, {"he": "מספר הזמנה", "fa": "شماره سفارش", "en": "Order no."}],
    "person": [{"he": "איש קשר", "fa": "رابط", "en": "Contact person"}, {"he": "מאשר", "fa": "تأییدکننده", "en": "Approved by"}],
    "amount": [{"he": "הצעה קודמת", "fa": "پیشنهاد قبلی", "en": "Previous offer"}, {"he": "קנס", "fa": "جریمه", "en": "Fine"}],
}
GROUP = {"cur": {"he": "נתונים עדכניים", "fa": "اطلاعات فعلی", "en": "Current data"},
         "prev": {"he": "נתונים קודמים", "fa": "اطلاعات قبلی", "en": "Previous data"}}
NOTES = {"he": ["לבדוק שוב", "דחוף", "אושר בטלפון", "ללא הערות", "ממתין למסמכים", "עודכן ידנית"],
         "fa": ["بررسی مجدد", "فوری", "تأیید تلفنی", "بدون توضیح", "در انتظار مدارک", "به‌روزرسانی دستی"],
         "en": ["recheck", "urgent", "confirmed by phone", "no notes", "awaiting documents", "updated manually"]}
NOISE = [  # (labels, value generator)
    ({"he": "טלפון", "fa": "تلفن", "en": "Phone"}, lambda rng, lang: L.phone(rng, lang)),
    ({"he": "כמות", "fa": "تعداد", "en": "Qty"}, lambda rng, lang: str(rng.randint(1, 60))),
    ({"he": "הערות", "fa": "توضیحات", "en": "Notes"}, lambda rng, lang: rng.choice(NOTES[lang])),
    ({"he": "קוד פנימי", "fa": "کد داخلی", "en": "Internal code"},
     lambda rng, lang: f"{rng.choice('ABKMTZ')}{rng.randint(10, 99)}-{rng.randint(100, 999)}"),
    ({"he": "עדיפות", "fa": "اولویت", "en": "Priority"}, lambda rng, lang: str(rng.randint(1, 5))),
    ({"he": "מס׳ שורה במערכת", "fa": "ردیف سیستم", "en": "Sys. row"}, lambda rng, lang: str(rng.randint(1000, 99999))),
]
DITTO = {"he": ["〃", "כנ״ל", "\""], "fa": ["〃", "همان", "\""], "en": ["〃", "ditto", "\""]}
NULL_CELL = ["", "—", "-"]
SECTION = {"he": "{l}: {v}", "fa": "{l}: {v}", "en": "{l}: {v}"}


def big_table(schema: Schema, plan: dict, lang: str, rng: random.Random, as_text: bool) -> str:
    """Render the H13-family table from plan['table'] + plan['decoys'] (HTML, or a pipe table if as_text)."""
    cfg, recs, cases = plan["table"], plan["records"], plan["hard_cases"]
    dec = {(d["col"], d["record"]): d["value"] for d in plan["decoys"] if d.get("col")}
    cont = schema.container_slot if cfg["sections"] else None
    null_cell, ditto = rng.choice(NULL_CELL), rng.choice(DITTO[lang])

    def gt_cell(name):
        def f(i):
            v = recs[i]["slots"].get(name)
            if not v:
                return null_cell
            if name == cfg["ditto_slot"] and i > 0 and (recs[i - 1]["slots"].get(name) or {}).get("value") == v["value"] \
                    and not any(st == i for st, _ in cfg["sections"]):
                return ditto
            return v["value"]
        return f

    # columns: (top_group|None, leaf label, cell fn)
    main = (None, label("main", lang), lambda i: recs[i]["main_entity"])
    singles = [(None, label(s.name, lang), gt_cell(s.name)) for s in schema.slots
               if s.name != cont and s.name not in cfg["prev_slots"]]
    for name in cfg["dist_cols"]:
        s = schema.slot(name)
        opts = DIST_LABEL.get(s.type) or []
        lab = rng.choice(opts)[lang] if opts and rng.random() < 0.6 else DIST_LABEL["prev"][lang].format(l=label(name, lang))
        singles.append((None, lab, lambda i, n=name: dec[(f"dist:{n}", i)]))
    for lab, gen in rng.sample(NOISE, k=rng.randint(2, 3)):
        vals = [gen(rng, lang) for _ in recs]
        singles.append((None, lab[lang], lambda i, v=vals: v[i]))
    blocks = [[c] for c in singles]
    if cfg["prev_slots"]:
        cur = [(GROUP["cur"][lang], label(n, lang), gt_cell(n)) for n in cfg["prev_slots"]]
        prv = [(GROUP["prev"][lang], label(n, lang), lambda i, n=n: dec[(f"prev:{n}", i)]) for n in cfg["prev_slots"]]
        blocks += [cur, prv]
    rng.shuffle(blocks)
    cols = [main] + [c for b in blocks for c in b]
    sec_at = {st: v for st, v in cfg["sections"]}
    sec_lab = label(schema.container_slot, lang) if cont else ""
    e = lambda x: html.escape(x or "", quote=False)

    if "H17" in cases:  # transposed: attribute rows x entity columns
        rows = [[c[1]] + [c[2](i) for i in range(len(recs))] for c in cols[1:]]
        if as_text:
            return "\n".join(" | ".join(r) for r in [[cols[0][1]] + [r["main_entity"] for r in recs]] + rows)
        head = "<tr>" + f"<th>{e(cols[0][1])}</th>" + "".join(f"<th>{e(r['main_entity'])}</th>" for r in recs) + "</tr>"
        body = "".join("<tr>" + f"<th>{e(r[0])}</th>" + "".join(f"<td>{e(c)}</td>" for c in r[1:]) + "</tr>" for r in rows)
        return f"<table border=\"1\"><thead>{head}</thead><tbody>{body}</tbody></table>"

    two = bool(cfg["prev_slots"])
    if as_text:
        lines = []
        if two:
            top, last = [], None
            for c in cols:
                top.append(c[0] if c[0] and c[0] != last else "")
                last = c[0]
            lines.append(" | ".join(top))
        lines.append(" | ".join(c[1] for c in cols))
        lines.append("-" * 40)
        for i in range(len(recs)):
            if i in sec_at:
                lines.append(f"■ {SECTION[lang].format(l=sec_lab, v=sec_at[i])}")
            lines.append(" | ".join(c[2](i) for c in cols))
        return "\n".join(lines)
    if two:
        top, j = "", 0
        while j < len(cols):
            g = cols[j][0]
            if g is None:
                top += f"<th rowspan=\"2\">{e(cols[j][1])}</th>"
                j += 1
            else:
                n = sum(1 for _ in __import__("itertools").takewhile(lambda c: c[0] == g, cols[j:]))
                top += f"<th colspan=\"{n}\">{e(g)}</th>"
                j += n
        head = f"<tr>{top}</tr><tr>" + "".join(f"<th>{e(c[1])}</th>" for c in cols if c[0]) + "</tr>"
    else:
        head = "<tr>" + "".join(f"<th>{e(c[1])}</th>" for c in cols) + "</tr>"
    body = ""
    for i in range(len(recs)):
        if i in sec_at:
            body += f"<tr><th colspan=\"{len(cols)}\">{e(SECTION[lang].format(l=sec_lab, v=sec_at[i]))}</th></tr>"
        body += "<tr>" + "".join(f"<td>{e(c[2](i))}</td>" for c in cols) + "</tr>"
    return f"<table border=\"1\"><thead>{head}</thead><tbody>{body}</tbody></table>"


# ---- H19 long multi-section document ----
PART = {"he": "חלק {k}: {p}", "fa": "بخش {k}: {p}", "en": "Part {k}: {p}"}
PLURAL_NOM = {"he": lambda p: p.removeprefix("ה"), "fa": lambda p: p, "en": lambda p: p}


def long_doc(schema: Schema, plan: dict, lang: str, rng: random.Random, as_text: bool) -> str:
    """Render plan['long'] sections (table / list / list_org / cards). Gold = plan records."""
    recs, out = plan["records"], []
    e = (lambda x: x) if as_text else (lambda x: html.escape(x or "", quote=False))
    sep, nul = rng.choice([" — ", " | ", " – "]), rng.choice(NULL_CELL[1:])
    for si, sec in enumerate(plan["long"]["sections"]):
        sub = recs[sec["start"]:sec["end"]]
        p = PLURAL[schema.id][lang]
        head = SUBTITLE[lang].format(p=p, org=sec["org"]) if sec["org"] else \
            PART[lang].format(k=si + 1, p=PLURAL_NOM[lang](p))
        out.append(f"\n## {head}" if as_text else f"<h2>{e(head)}</h2>")
        pre = f"s{si}:"
        sdec = [{**d, "col": d["col"][len(pre):], "record": d["record"] - sec["start"]}
                for d in plan["decoys"] if d.get("col", "").startswith(pre)]
        if sec["kind"] == "table":
            sp = {"table": sec["table"], "records": sub, "hard_cases": sec["sub_cases"], "decoys": sdec}
            out.append(big_table(schema, sp, lang, rng, as_text))
            continue
        skip = {schema.container_slot} if sec["org"] else set()
        order = [s for s in schema.slots if s.name not in skip]
        rng.shuffle(order)
        dlab = {}
        for name in sec["dist"]:
            opts = DIST_LABEL.get(schema.slot(name).type) or []
            dlab[name] = rng.choice(opts)[lang] if opts else DIST_LABEL["prev"][lang].format(l=label(name, lang))
        items = []
        for i, r in enumerate(sub):
            kv = [(label(s.name, lang), (r["slots"][s.name] or {}).get("value") or nul) for s in order]
            for name in sec["dist"]:
                dv = next(d["value"] for d in sdec if d["record"] == i and d["col"] == f"dist:{name}")
                kv.insert(rng.randint(0, len(kv)), (dlab[name], dv))
            if sec["kind"] == "cards":
                if as_text:
                    items.append(r["main_entity"] + "\n" + "\n".join(f"  {k}: {v}" for k, v in kv))
                else:
                    items.append(f"<h3>{e(r['main_entity'])}</h3><p>" + "<br>".join(f"{e(k)}: {e(v)}" for k, v in kv) + "</p>")
            else:
                line = r["main_entity"] + sep + "; ".join(f"{k}: {v}" for k, v in kv)
                items.append(f"• {line}" if as_text else f"<li>{e(line)}</li>")
        if sec["kind"] == "cards":
            out.append(("\n\n" if as_text else "\n").join(items))
        else:
            out.append("\n".join(items) if as_text else "<ul>" + "".join(items) + "</ul>")
    return "\n".join(out).strip()
