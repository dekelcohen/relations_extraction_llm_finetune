"""Fresh seeds for data/test_long: fetch_seeds dedups only inside its output file, so preload every URL used by the
pilot / stage-1 / stage-2 seed files (dataset="preload", not counted per source), fetch, then strip the preloaded rows."""
import glob
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from re_synth.source import fetch_seeds

load_dotenv(".env")  # HF_TOKEN avoids datasets-server rate limits

OUT = Path("data/test_long")
PATTERNS = ["data/pilot/*seeds*", "data/synth*/seeds*.jsonl", "data/stage_2_RL_GRPO/seeds*.jsonl"]


def used_urls() -> set[str]:
    urls = set()
    for pat in PATTERNS:
        for p in glob.glob(pat):
            urls |= {json.loads(l)["url"] for l in open(p, encoding="utf-8") if l.strip()}
    return urls


def main(n: int = 60, seed: int = 3) -> None:
    urls = used_urls()
    print("excluded urls:", len(urls))
    for lang in ("he", "fa"):
        out = OUT / f"seeds_{lang}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for u in sorted(urls):
                f.write(json.dumps({"lang": lang, "dataset": "preload", "url": u}) + "\n")
        fetch_seeds(lang, n, out, seed)
        rows = [json.loads(l) for l in out.open(encoding="utf-8")]
        keep = [r for r in rows if r["dataset"] != "preload" and r["url"] not in urls]
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        print(lang, "seeds:", len(keep))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
