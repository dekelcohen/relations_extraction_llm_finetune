"""Build synthetic samples (parallel, cached, resumable)."""
from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .hardcases import plan_sample
from .inject import dump, generate_sample
from .source import load_seeds

log = logging.getLogger(__name__)
FORMAT_W = {"text": 0.45, "html": 0.3, "email": 0.25}


def build(seeds_path: Path, out_path: Path, langs: list[str], n_per_lang: int, model: str = "gen", seed: int = 0,
          workers: int = 8, schema_id: str | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open(encoding="utf-8")}
    jobs = []
    for lang in langs:
        seeds = load_seeds(seeds_path, lang)
        if not seeds:
            log.warning("no seeds for %s", lang)
            continue
        for i in range(n_per_lang):
            sid = f"{lang}-{seed}-{i:05d}"
            if sid in done:
                continue
            rng = random.Random(sid)
            fmt = rng.choices(list(FORMAT_W), list(FORMAT_W.values()))[0]
            plan = plan_sample(rng, lang, fmt, schema_id)
            jobs.append((sid, plan, seeds[i % len(seeds)], rng))
    log.info("build: %d jobs (%d already done) -> %s", len(jobs), len(done), out_path)
    stats = {"ok": 0, "fail": 0, "error": 0}
    with ThreadPoolExecutor(workers) as ex, out_path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(generate_sample, plan, sd, rng, model, sid): sid for sid, plan, sd, rng in jobs}
        for fu in as_completed(futs):
            sid = futs[fu]
            try:
                s = fu.result()
            except Exception as e:  # network / provider errors: skip, rerun resumes
                log.warning("sample %s error: %s", sid, e)
                stats["error"] += 1
                continue
            stats[s["qc"]["status"]] += 1
            if s["qc"]["status"] != "ok":
                log.info("sample %s QC fail: %s", sid, s["qc"]["issues"])
            f.write(dump(s) + "\n")
            f.flush()
    log.info("build stats: %s", stats)
    return stats


def load_samples(path: Path, only_ok: bool = True) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    return [r for r in rows if not only_ok or r.get("qc", {}).get("status") == "ok"]
