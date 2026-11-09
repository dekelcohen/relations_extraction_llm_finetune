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
