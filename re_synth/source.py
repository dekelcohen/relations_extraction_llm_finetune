"""Seed texts from HF datasets-server (no `datasets` install needed): Wikipedia + FineWeb-2 (he/fa/en)."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)
API = "https://datasets-server.huggingface.co/rows"

SOURCES = {
    "he": [("wikimedia/wikipedia", "20231101.he", "wiki"), ("HuggingFaceFW/fineweb-2", "heb_Hebr", "web")],
    "fa": [("wikimedia/wikipedia", "20231101.fa", "wiki"), ("HuggingFaceFW/fineweb-2", "fas_Arab", "web")],
    "en": [("wikimedia/wikipedia", "20231101.en", "wiki"), ("HuggingFaceFW/fineweb", "sample-10BT", "web")],
}


def _get(params: dict, retries: int = 7) -> dict:
    tok = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    for i in range(retries):
        try:
            r = requests.get(API, params=params, timeout=60, headers=headers)
            if r.status_code == 200:
                return r.json()
            log.warning("datasets-server %s: %s", r.status_code, r.text[:200])
        except requests.RequestException as e:
            log.warning("datasets-server error %s", e)
        time.sleep(min(90, 5 * 2 ** i))  # 429 rate limits: exponential backoff
    raise RuntimeError(f"datasets-server failed for {params}")


def excerpt(text: str, max_chars: int = 1800, min_chars: int = 400) -> str | None:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) > 60]
    out = ""
    for p in paras:
        if len(out) + len(p) > max_chars:
            if not out:
                out = p[:max_chars]
            break
        out += p + "\n\n"
    out = out.strip()
    return out if len(out) >= min_chars else None


def fetch_seeds(lang: str, n: int, out_path: Path, seed: int = 0, page: int = 12) -> int:
    """Append ~n seeds for `lang` to out_path (jsonl). Random pages from each source, half wiki / half web."""
    rng = random.Random(f"{seed}-{lang}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing, have = set(), {}
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            o = json.loads(l)
            existing.add(o["url"])
            have[(o["lang"], o["dataset"])] = have.get((o["lang"], o["dataset"]), 0) + 1
    got = 0
    per_src = (n + len(SOURCES[lang]) - 1) // len(SOURCES[lang])
    with out_path.open("a", encoding="utf-8") as f:
        for ds, cfg, kind in SOURCES[lang]:
            got_src, tries = have.get((lang, ds), 0), 0
            if got_src >= per_src:
                continue
            try:
                total = _get({"dataset": ds, "config": cfg, "split": "train", "offset": 0, "length": 1})["num_rows_total"]
            except RuntimeError as e:
                log.warning("skip source %s/%s: %s", ds, cfg, e)
                continue
            while got_src < per_src and tries < per_src * 3:
                tries += 1
                off = rng.randrange(0, max(1, total - page))
                try:
                    rows = _get({"dataset": ds, "config": cfg, "split": "train", "offset": off, "length": page})["rows"]
                except RuntimeError as e:
                    log.warning("page failed: %s", e)
                    break
                rng.shuffle(rows)
                for r in rows[:4]:  # max 4 per page for diversity
                    row = r["row"]
                    url = row.get("url") or f"{ds}/{cfg}/{off + r['row_idx']}"
                    ex = excerpt(row.get("text", ""))
                    if not ex or url in existing:
                        continue
                    existing.add(url)
                    f.write(json.dumps({"lang": lang, "source": kind, "dataset": ds, "title": row.get("title", ""),
                                        "url": url, "text": ex}, ensure_ascii=False) + "\n")
                    got_src += 1
                    if got_src >= per_src:
                        break
            log.info("seeds %s %s/%s: %d", lang, ds, cfg, got_src)
            got += got_src
    return got


def load_seeds(path: Path, lang: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    return [r for r in rows if lang is None or r["lang"] == lang]
