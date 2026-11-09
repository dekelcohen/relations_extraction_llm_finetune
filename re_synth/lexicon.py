"""Locale value generators (he/fa/en): names, places, orgs, IDs, dates, statuses.

All generated values are synthetic (counterfactual, H8): plates/IDs are random, orgs are invented,
places are real cities in random combinations.
"""
from __future__ import annotations

import random
import string
from datetime import date, timedelta

LANGS = ("he", "fa", "en")
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


def to_fa_digits(s: str) -> str:
    return "".join(FA_DIGITS[int(c)] if c.isdigit() and c.isascii() else c for c in s)


# --------------------------------------------------------------------------- calendars
def g2j(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Gregorian -> Jalali (jalaali algorithm)."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + gd + g_d_m[gm - 1]
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        return jy, 1 + days // 31, 1 + days % 31
    return jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30


def j2g(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    """Jalali -> Gregorian (jalaali algorithm)."""
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd
    days += (jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 1
    for md in month_days:
        if gd <= md:
            break
        gd -= md
        gm += 1
    return gy, gm, gd


EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
HE_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר",
             "דצמבר"]
FA_G_MONTHS = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر",
               "دسامبر"]
FA_J_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن",
               "اسفند"]
HEB_CAL_MONTHS = ["תשרי", "חשוון", "כסלו", "טבת", "שבט", "אדר", "ניסן", "אייר", "סיוון", "תמוז", "אב", "אלול"]

_GEM = [(400, "ת"), (300, "ש"), (200, "ר"), (100, "ק"), (90, "צ"), (80, "פ"), (70, "ע"), (60, "ס"), (50, "נ"),
        (40, "מ"), (30, "ל"), (20, "כ"), (10, "י"), (9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"),
        (3, "ג"), (2, "ב"), (1, "א")]


def gematria(n: int) -> str:
    n %= 1000
    out = ""
    for v, ch in _GEM:
        while n >= v:
            if n in (15, 16):  # avoid spelling divine names: ט"ו / ט"ז
                out += "טו" if n == 15 else "טז"
                n = 0
                break
            out += ch
            n -= v
    return out[:-1] + '"' + out[-1] if len(out) > 1 else out + "'"


def hebrew_calendar_date(rng: random.Random) -> str:
    """A random Hebrew-calendar date string, e.g. כ"ג בשבט תשפ"ו (scored as a string, not converted)."""
    return f"{gematria(rng.randint(1, 29))} ב{rng.choice(HEB_CAL_MONTHS)} {gematria(rng.choice([785, 786, 787]))}"


def random_date(rng: random.Random, start=date(2023, 1, 1), end=date(2027, 12, 31)) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def format_date(d: date, lang: str, rng: random.Random) -> tuple[str, str]:
    """Return (surface, style). Styles include H10 locale variants (jalali, persian digits, hebrew calendar)."""
    if lang == "en":
        style = rng.choice(["iso", "mdy_long", "dmy_long", "dmy_short"])
        s = {"iso": d.isoformat(), "mdy_long": f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}",
             "dmy_long": f"{d.day} {EN_MONTHS[d.month - 1]} {d.year}",
             "dmy_short": f"{d.day} {EN_MONTHS[d.month - 1][:3]} {d.year}"}[style]
        return s, style
    if lang == "he":
        style = rng.choices(["dmy_slash2", "dmy_slash4", "dmy_dot", "d_month_y", "hebcal"], [3, 3, 2, 3, 1])[0]
        if style == "hebcal":
            return hebrew_calendar_date(rng), style
        s = {"dmy_slash2": f"{d.day}/{d.month}/{d.year % 100:02d}", "dmy_slash4": f"{d.day:02d}/{d.month:02d}/{d.year}",
             "dmy_dot": f"{d.day}.{d.month}.{d.year}", "d_month_y": f"{d.day} ב{HE_MONTHS[d.month - 1]} {d.year}"}[style]
        return s, style
    style = rng.choices(["jalali_num", "jalali_num_fa", "jalali_long", "jalali_long_fa", "greg_num", "greg_long"],
                        [3, 3, 2, 2, 1, 1])[0]
    jy, jm, jd = g2j(d.year, d.month, d.day)
    s = {"jalali_num": f"{jy}/{jm:02d}/{jd:02d}", "jalali_num_fa": to_fa_digits(f"{jy}/{jm:02d}/{jd:02d}"),
         "jalali_long": f"{jd} {FA_J_MONTHS[jm - 1]} {jy}", "jalali_long_fa": to_fa_digits(f"{jd} {FA_J_MONTHS[jm - 1]} {jy}"),
         "greg_num": f"{d.year}/{d.month:02d}/{d.day:02d}", "greg_long": f"{d.day} {FA_G_MONTHS[d.month - 1]} {d.year}"}[style]
    return s, style


# --------------------------------------------------------------------------- gazetteers
CITIES = {
    "he": ["יהוד", "חיפה", "אשדוד", "באר שבע", "נתניה", "רחובות", "הרצליה", "עפולה", "טבריה", "אילת", "כפר סבא",
           "רעננה", "לוד", "רמלה", "חולון", "בת ים", "פתח תקווה", "ראשון לציון", "מודיעין", "קריית גת", "נהריה", "עכו",
           "דימונה", "קריית שמונה", "אור יהודה", "גבעתיים", "רמת גן", "הוד השרון", "ראש העין", "יבנה"],
    "fa": ["یزد", "شیراز", "اصفهان", "تبریز", "مشهد", "کرج", "قم", "اهواز", "رشت", "کرمان", "همدان", "اراک", "زنجان",
           "قزوین", "ساری", "گرگان", "بندرعباس", "بوشهر", "کاشان", "سنندج", "ارومیه", "کرمانشاه", "خرم‌آباد", "بیرجند",
           "سمنان", "زاهدان", "یاسوج", "ایلام", "اردبیل", "بجنورد"],
    "en": ["Leeds", "Austin", "Denver", "Rotterdam", "Lyon", "Porto", "Tucson", "Glasgow", "Hamburg", "Dayton", "Reno",
           "Boise", "Bristol", "Malmo", "Ghent", "Turin", "Cork", "Halifax", "Fresno", "Akron"],
}
STREETS = {
    "he": ["הרצל", "ויצמן", "ז'בוטינסקי", "רוטשילד", "בן גוריון", "הנשיא", "העצמאות", "אלנבי", "הבנים", "המלאכה"],
    "fa": ["خیابان ولیعصر", "خیابان انقلاب", "خیابان آزادی", "بلوار کشاورز", "خیابان شریعتی", "خیابان فردوسی",
           "بلوار امام رضا", "خیابان سعدی"],
    "en": ["Mill Lane", "Station Road", "Elm Street", "Harbor Drive", "King Street", "Oak Avenue", "Canal Street"],
}
SITES = {  # sub-locations
    "he": ["חניון {c}", "אזור התעשייה של {c}", "הנמל ב{c}", "המחסן המרכזי ב{c}"],
    "fa": ["پارکینگ {c}", "شهرک صنعتی {c}", "انبار مرکزی {c}", "بندر {c}"],
    "en": ["the {c} depot", "{c} Central Parking", "the {c} industrial park", "{c} harbor", "the {c} warehouse"],
}


def location(rng: random.Random, lang: str) -> str:
    c = rng.choice(CITIES[lang])
    k = rng.random()
    if k < 0.6:
        return c
    if k < 0.8:
        n = rng.randint(1, 140)
        if lang == "en":
            return f"{n} {rng.choice(STREETS[lang])}, {c}"
        if lang == "he":
            return f"רחוב {rng.choice(STREETS[lang])} {n}, {c}"
        return f"{c}، {rng.choice(STREETS[lang])}، پلاک {n}"
    return rng.choice(SITES[lang]).format(c=c)


ORG_PARTS = {
    "he": (["קבוצת", "חברת", "תעשיות", "השכרת רכב", "לוגיסטיקה", "מוסכי", "שירותי"],
           ["אורן", "גל-אור", "רמות", "אופק", "שחר", "נחשון", "תבור", "ארבל", "כרמל", "גלבוע", "אלון", "דקל", "ברק",
            "לביא", "צור"], ["", "", ' בע"מ']),
    "fa": (["شرکت", "گروه", "حمل‌ونقل", "نمایندگی", "خدمات"],
           ["آریا", "پارس", "البرز", "سپهر", "دنا", "کوثر", "ارغوان", "مهرگان", "سیمرغ", "نوین", "زاگرس", "بهار",
            "آسمان", "روشن", "پویا"], ["", "", " (سهامی خاص)"]),
    "en": ([""], ["Northwind", "Bluecrest", "Ironbridge", "Silverline", "Redwood", "Harborview", "Summit", "Keystone",
                  "Brightpath", "Oakmont", "Falcon", "Granite"],
           [" Logistics", " Rentals", " Motors", " Holdings", " Freight", " Services", " Ltd", " Group"]),
}
BANKS = {"he": ["בנק אופק", "בנק הכרמל", "בנק תבור", "בנק נחשון"], "fa": ["بانک آریا", "بانک سپهر", "بانک البرز", "بانک پارس نوین"],
         "en": ["Bluecrest Bank", "Granite Savings Bank", "Summit Credit Union", "Keystone Bank"]}


def org(rng: random.Random, lang: str) -> str:
    a, b, c = ORG_PARTS[lang]
    return f"{rng.choice(a)} {rng.choice(b)}{rng.choice(c)}".strip()


FIRST = {"he": ["יוסי", "מיכל", "אבי", "נועה", "דני", "שירה", "רון", "תמר", "איתי", "הדס", "עומר", "ליאת", "גיל", "מאיה", "אלון"],
         "fa": ["علی", "مریم", "رضا", "زهرا", "حسین", "فاطمه", "محمد", "سارا", "امیر", "نرگس", "مهدی", "لیلا", "کاوه", "شیرین", "بهرام"],
         "en": ["James", "Maria", "David", "Sarah", "Tom", "Emily", "Daniel", "Laura", "Chris", "Anna", "Mark", "Julia"]}
LAST = {"he": ["כהן", "לוי", "מזרחי", "פרץ", "ביטון", "אברהם", "פרידמן", "שפירא", "גולן", "אזולאי", "דהן", "בן דוד"],
        "fa": ["احمدی", "محمدی", "حسینی", "رضایی", "کریمی", "موسوی", "جعفری", "صادقی", "رحیمی", "نوری", "تهرانی", "کاظمی"],
        "en": ["Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Clark", "Hall", "Young"]}


def person(rng: random.Random, lang: str) -> str:
    return f"{rng.choice(FIRST[lang])} {rng.choice(LAST[lang])}"


# --------------------------------------------------------------------------- IDs
def _alnum(rng, n, chars=string.ascii_uppercase + string.digits):
    return "".join(rng.choice(chars) for _ in range(n))


def plate(rng: random.Random, lang: str) -> str:
    if lang == "he":
        return rng.choice([f"{rng.randint(10, 99)}-{rng.randint(100, 999)}-{rng.randint(10, 99)}",
                           f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(100, 999)}"])
    if lang == "fa":
        s = f"{rng.randint(10, 99)}{rng.choice('بجدسصطقلمنوهی')}{rng.randint(100, 999)}-{rng.randint(10, 99)}"
        return to_fa_digits(s) if rng.random() < 0.5 else s
    return rng.choice([f"{rng.randint(1, 9)}{_alnum(rng, 3, string.ascii_uppercase)}{rng.randint(100, 999)}",
                       f"{_alnum(rng, 2, string.ascii_uppercase)}{rng.randint(10, 99)} {_alnum(rng, 3, string.ascii_uppercase)}"])


def tracking_id(rng, lang):
    return rng.choice([f"TRK-{_alnum(rng, 6)}", f"{_alnum(rng, 4, string.ascii_uppercase)}{rng.randint(1000000, 9999999)}",
                       f"SHP{rng.randint(10000000, 99999999)}"])


def national_id(rng, lang):
    if lang == "en":
        return f"EMP-{rng.randint(10000, 99999)}"
    s = "".join(str(rng.randint(0, 9)) for _ in range(9 if lang == "he" else 10))
    return to_fa_digits(s) if lang == "fa" and rng.random() < 0.4 else s


def serial(rng, lang):
    return rng.choice([f"SN{_alnum(rng, 8)}", f"{_alnum(rng, 3, string.ascii_uppercase)}-{rng.randint(100000, 999999)}"])


def account_number(rng, lang):
    return f"{rng.randint(10, 99)}-{rng.randint(100, 999)}-{rng.randint(100000, 999999)}"


def imo(rng, lang):
    return f"IMO {rng.randint(9000000, 9999999)}"


def contract_id(rng, lang):
    return rng.choice([f"CN-{rng.randint(2023, 2027)}-{rng.randint(100, 9999)}", f"{rng.randint(1000, 9999)}/{rng.randint(10, 99)}"])


def phone(rng, lang):
    if lang == "he":
        return f"05{rng.randint(0, 9)}-{rng.randint(1000000, 9999999)}"
    if lang == "fa":
        return f"09{rng.randint(10, 39)}{rng.randint(1000000, 9999999)}"
    return f"+1 {rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def amount(rng, lang):
    v = rng.choice([rng.randint(5, 950) * 1000, rng.randint(1, 90) * 100000])
    if lang == "he":
        return f"{v:,} ש\"ח" if rng.random() < 0.5 else f"₪{v:,}"
    if lang == "fa":
        s = f"{v:,} تومان" if rng.random() < 0.5 else f"{v:,} ریال"
        return to_fa_digits(s) if rng.random() < 0.4 else s
    return f"${v:,}" if rng.random() < 0.5 else f"{v:,} USD"


# --------------------------------------------------------------------------- domain vocab
VEHICLES = {
    "he": ["מאזדה 3", "טויוטה קורולה", "יונדאי i20", "סובארו אימפרזה", "קיה פיקנטו", "סקודה אוקטביה", "פולקסווגן גולף",
           "ניסאן מיקרה", "מיצובישי לנסר", "פורד טרנזיט"],
    "fa": ["پژو ۲۰۶", "سمند", "پراید", "تیبا", "دنا پلاس", "کوییک", "شاهین", "ساینا", "هایما", "رانا"],
    "en": ["Mazda 3", "Toyota Corolla", "Subaru Impreza", "Kia Picanto", "Ford Transit", "Honda Civic", "VW Golf",
           "Hyundai i20", "Nissan Leaf", "Skoda Octavia"],
}
COLORS = {"he": ["שחורה", "לבנה", "אדומה", "כחולה", "ירוקה", "כסופה", "אפורה"],
          "fa": ["سفید", "مشکی", "نقره‌ای", "آبی", "قرمز", "خاکستری", "سبز"],
          "en": ["black", "white", "red", "blue", "green", "silver", "grey"]}
GOODS = {"he": ["משלוח ציוד רפואי", "מכולת חלקי חילוף", "משטח מוצרי חשמל", "משלוח ספרים", "ארגז כלי עבודה"],
         "fa": ["محموله تجهیزات پزشکی", "کانتینر قطعات یدکی", "پالت لوازم برقی", "محموله کتاب", "جعبه ابزار"],
         "en": ["medical equipment shipment", "container of spare parts", "pallet of electronics", "book consignment",
                "crate of power tools"]}
ASSETS = {"he": ["מחשב נייד Dell Latitude", "מקרן Epson", "טאבלט Samsung", "מדפסת HP", "רחפן DJI"],
          "fa": ["لپ‌تاپ Dell Latitude", "ویدئو پروژکتور Epson", "تبلت Samsung", "پرینتر HP", "پهپاد DJI"],
          "en": ["Dell Latitude laptop", "Epson projector", "Samsung tablet", "HP printer", "DJI drone"]}
ROLES = {"he": ["מנהלת חשבונות", "מהנדס תוכנה", "נהג משאית", "מנהל מחסן", "אחות", "טכנאי"],
         "fa": ["حسابدار", "مهندس نرم‌افزار", "راننده کامیون", "مدیر انبار", "پرستار", "تکنسین"],
         "en": ["accountant", "software engineer", "truck driver", "warehouse manager", "nurse", "technician"]}
VESSELS = {"he": ["אוניית המשא גלית", "המכלית נגה", "ספינת המכולות אביב", "אוניית הצובר תמר"],
           "fa": ["کشتی باری ستاره", "نفتکش دریا", "کشتی کانتینری مهتاب", "کشتی فله‌بر البرز"],
           "en": ["cargo ship Galit", "tanker Nova Star", "container ship Aurora", "bulk carrier Tamar"]}
MEETINGS = {"he": ["ישיבת תקציב", "כנס ספקים", "דיון ועדת תכנון", "סדנת בטיחות"],
            "fa": ["جلسه بودجه", "همایش تأمین‌کنندگان", "نشست کمیسیون برنامه‌ریزی", "کارگاه ایمنی"],
            "en": ["budget meeting", "suppliers conference", "planning committee hearing", "safety workshop"]}

STATUSES = {
    "vehicle": {"he": ["חונה", "ממתין לאיסוף", "בתיקון", "נגנב", "מעוקל", "נמכר"],
                "fa": ["پارک شده", "در انتظار تحویل", "در تعمیرگاه", "سرقت شده", "توقیف شده", "فروخته شده"],
                "en": ["parked", "awaiting pick-up", "under repair", "stolen", "impounded", "sold"]},
    "shipment": {"he": ["בדרך", "נמסר", "מעוכב במכס", "הוחזר לשולח", "ממתין לאיסוף"],
                 "fa": ["در راه", "تحویل شده", "در گمرک متوقف", "مرجوع شده", "آماده تحویل"],
                 "en": ["in transit", "delivered", "held at customs", "returned to sender", "ready for pickup"]},
    "asset": {"he": ["פעיל", "בתיקון", "אבד", "הושבת", "הוחזר"], "fa": ["فعال", "در تعمیر", "مفقود", "از رده خارج", "مسترد شده"],
              "en": ["active", "in repair", "lost", "decommissioned", "returned"]},
    "meeting": {"he": ["מתוכנן", "בוטל", "נדחה", "הסתיים"], "fa": ["برنامه‌ریزی شده", "لغو شده", "به تعویق افتاده", "برگزار شده"],
                "en": ["scheduled", "cancelled", "postponed", "held"]},
    "contract": {"he": ["נחתם", "בתוקף", "הוקפא", "בוטל", "פג תוקף"], "fa": ["امضا شده", "در جریان", "معلق", "لغو شده", "منقضی"],
                 "en": ["signed", "in force", "suspended", "terminated", "expired"]},
    "account": {"he": ["פעיל", "מוקפא", "נסגר", "מוגבל"], "fa": ["فعال", "مسدود", "بسته شده", "محدود"],
                "en": ["active", "frozen", "closed", "restricted"]},
    "vessel": {"he": ["עוגנת", "בהפלגה", "בתיקון במספנה", "מעוכבת"], "fa": ["پهلو گرفته", "در حال حرکت", "در تعمیرگاه", "توقیف شده"],
               "en": ["docked", "at sea", "in dry dock", "detained"]},
}

GENERATORS = {
    "location": location, "org": org, "person": person, "plate": plate, "tracking_id": tracking_id,
    "national_id": national_id, "serial": serial, "account_number": account_number, "imo": imo,
    "contract_id": contract_id, "phone": phone, "amount": amount,
    "bank": lambda rng, lang: rng.choice(BANKS[lang]),
    "color": lambda rng, lang: rng.choice(COLORS[lang]),
    "role": lambda rng, lang: rng.choice(ROLES[lang]),
}
