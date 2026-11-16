"""Polite INSS (Hebrew) scraper: list pages -> article pages -> data/raw/inss/articles.jsonl (git-ignored, copyrighted).

  python scripts/scrape_inss.py --max 30
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
BASE = "https://www.inss.org.il/he/publication/"
OUT = Path("data/raw/inss/articles.jsonl")


def get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=40)
    r.raise_for_status()
    return r.text


def list_urls(pages: int) -> list[str]:
    """The list page paginates via JS, so combine the feed (paged) and the per-type list pages."""
    urls: list[str] = []
    srcs = [f"{BASE}feed/?paged={p}" for p in range(1, pages + 1)] + [BASE] + \
           [f"{BASE}?ptype={t}" for t in (1175, 217, 219, 384, 755, 1184)]
    for src in srcs:
        html = get(src)
        for slug in re.findall(r"inss\.org\.il/he/publication/([^\"/?<\s]+)/", html):
            u = f"https://www.inss.org.il/he/publication/{slug}/"
            if slug not in ("feed", "page") and u not in urls:
                urls.append(u)
        time.sleep(1.5)
    return urls


def parse(html: str) -> dict:
    s = BeautifulSoup(html, "html.parser")
    paras = [p.get_text(" ", strip=True) for p in s.select("div.content_publication_post")]
    paras = [p for p in paras if len(p) > 40]
    return {"title": (s.title.string or "").split(" - המכון")[0].strip(), "text": "\n\n".join(paras)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--pages", type=int, default=5)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    got = 0
    with OUT.open("w", encoding="utf-8") as f:
        for u in list_urls(a.pages):
            if got >= a.max:
                break
            try:
                art = parse(get(u))
            except Exception as e:
                print("skip", u, e)
                continue
            time.sleep(1.5)
            if len(art["text"]) < 3000:
                continue
            f.write(json.dumps({"url": u, **art}, ensure_ascii=False) + "\n")
            got += 1
            print(got, u, len(art["text"]))


if __name__ == "__main__":
    main()
