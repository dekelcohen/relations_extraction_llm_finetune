"""CLI: python -m re_synth <cmd> ...

  seeds         fetch seed excerpts (wiki + fineweb-2) for languages
  build         generate synthetic samples with the generator LLM (scrub+inject+verify+layout)
  predict       run an extraction model (OpenRouter id or alias weak/strong) on samples
  score         score predictions -> metrics json + markdown report
  mine          weak+strong preds -> tiers (easy/hard/ambiguous/weird)
  export        samples (+tiers) -> sft_*.jsonl / grpo_*.jsonl
  convert-redocred  Re-DocRED json -> samples in our record format
  show          pretty-print a sample
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _report_md(rep: dict, title: str) -> str:
    cols = ["n", "slot_p", "slot_r", "slot_f1", "rec_f1", "null_acc", "decoy_hit_rate", "halluc_rate", "json_valid",
            "exact_sample"]
    lines = [f"## {title}", "", "| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for g, m in rep.items():
        lines.append(f"| {g} | " + " | ".join("" if m.get(c) is None else str(m.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _chunk_args(p, flag: bool = True) -> None:
    from .chunking import CHUNK_CHARS, CHUNK_MIN, CHUNK_OVERLAP
    if flag:
        p.add_argument("--no_chunk", action="store_true", help="never split long documents")
    p.add_argument("--chunk_chars", type=int, default=CHUNK_CHARS, help="max chars per chunk")
    p.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="chars of trailing units repeated")
    p.add_argument("--chunk_min", type=int, default=CHUNK_MIN, help="docs up to this length are not chunked")


def _chunk_cfg(a) -> dict | None:
    if getattr(a, "no_chunk", False):
        return None
    return {"max_chars": a.chunk_chars, "overlap": a.chunk_overlap, "min_chars": a.chunk_min}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="re_synth")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seeds")
    p.add_argument("--langs", default="he,fa,en")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("data/seeds.jsonl"))
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("build")
    p.add_argument("--seeds", type=Path, default=Path("data/seeds.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/synth/samples.jsonl"))
    p.add_argument("--langs", default="he,fa,en")
    p.add_argument("--n", type=int, default=50, help="samples per language")
    p.add_argument("--model", default="gen")
    p.add_argument("--schema", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--focus", action="store_true", help="oversample hard cases of the FT model's errors (see hardcases.py)")
    p.add_argument("--long_p", type=float, default=None, help="--focus: share of H19 long documents (default 0.2)")

    p = sub.add_parser("predict")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    _chunk_args(p)

    p = sub.add_parser("score")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="metrics json; a .md report is written next to it")

    p = sub.add_parser("mine")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--weak", type=Path, required=True)
    p.add_argument("--strong", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("export")
    p.add_argument("--samples", type=Path, nargs="+", required=True)
    p.add_argument("--tiers", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-aug", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", action="store_true", help="train split: long docs -> their clean chunk rows")
    _chunk_args(p, flag=False)

    p = sub.add_parser("export-eval", help="all samples as one eval file (grpo-row format, canonical prompt)")
    p.add_argument("--samples", type=Path, nargs="+", required=True)
    p.add_argument("--tiers", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--langs", default=None, help="e.g. he,fa (default: all)")
    p.add_argument("--chunk", action="store_true", help="long docs -> their clean chunk rows (training files)")
    _chunk_args(p, flag=False)

    p = sub.add_parser("teacher-cot", help="teacher reasoning + verification -> thinking-mode SFT file")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True, help="output dir")
    p.add_argument("--ref", type=Path, nargs="*", default=[], help="samples.jsonl to dedup against (e.g. pilot)")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("convert-redocred")
    p.add_argument("--inp", type=Path, required=True)
    p.add_argument("--rel-info", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("show")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--id", default=None)
    p.add_argument("--i", type=int, default=0)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("llm_cache").setLevel(logging.WARNING)
    log = logging.getLogger("re_synth")

    if a.cmd == "seeds":
        from .source import fetch_seeds
        for lang in a.langs.split(","):
            log.info("seeds %s: +%d", lang, fetch_seeds(lang, a.n, a.out, a.seed))
    elif a.cmd == "build":
        from .build import build
        from .llm import usage_totals
        if a.long_p is not None:
            from . import hardcases
            hardcases.FOCUS_LONG_P = a.long_p
        build(a.seeds, a.out, a.langs.split(","), a.n, a.model, a.seed, a.workers, a.schema, a.focus)
        log.info("LLM usage: %s", usage_totals())
    elif a.cmd == "predict":
        from .build import load_samples
        from .llm import usage_totals
        from .mine import predict
        s = load_samples(a.samples)[: a.limit]
        predict(s, a.model, a.out, a.workers, chunk=_chunk_cfg(a))
        log.info("LLM usage: %s", usage_totals())
    elif a.cmd == "score":
        from .build import load_samples
        from .mine import group_report, load_preds, score_preds
        s = load_samples(a.samples)
        sc = score_preds(s, load_preds(a.preds))
        rep = group_report([x for x in s if x["id"] in sc], sc)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"report": rep, "per_sample": sc}, ensure_ascii=False, indent=1), encoding="utf-8")
        md = _report_md(rep, f"{a.preds.name}")
        a.out.with_suffix(".md").write_text(md, encoding="utf-8")
        print(md)
    elif a.cmd == "mine":
        from collections import Counter
        from .build import load_samples
        from .mine import load_preds, mine, score_preds
        s = load_samples(a.samples)
        t = mine(s, score_preds(s, load_preds(a.weak)), score_preds(s, load_preds(a.strong)))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(t, indent=1), encoding="utf-8")
        log.info("tiers: %s", dict(Counter(v["tier"] for v in t.values())))
    elif a.cmd == "export":
        from .build import load_samples
        from .export import export
        s = [x for p in a.samples for x in load_samples(p)]
        tiers = json.loads(a.tiers.read_text(encoding="utf-8")) if a.tiers else None
        print(export(s, a.out, tiers, a.seed, a.n_aug, chunk=_chunk_cfg(a) if a.chunk else None))
    elif a.cmd == "export-eval":
        from .build import load_samples
        from .export import export_eval
        s = [x for p in a.samples for x in load_samples(p)]
        if a.langs:
            s = [x for x in s if x["lang"] in a.langs.split(",")]
        tiers = json.loads(a.tiers.read_text(encoding="utf-8")) if a.tiers else None
        n = export_eval(s, a.out, tiers, chunk=_chunk_cfg(a) if a.chunk else None)
        log.info("export-eval: %d rows -> %s", n, a.out)
    elif a.cmd == "teacher-cot":
        from .build import load_samples
        from .llm import usage_totals
        from .teacher_cot import run
        ref = [x["text"] for p in a.ref for x in load_samples(p, only_ok=False)]
        print(run(load_samples(a.samples), a.model, a.out, ref, a.workers))
        log.info("LLM usage: %s", usage_totals())
    elif a.cmd == "convert-redocred":
        from .convert.redocred import convert
        n = convert(a.inp, a.rel_info, a.out, a.limit)
        log.info("converted %d docs -> %s", n, a.out)
    elif a.cmd == "show":
        from .build import load_samples
        s = load_samples(a.samples, only_ok=False)
        x = next(r for r in s if r["id"] == a.id) if a.id else s[a.i]
        print(x["text"])
        print("\nGT:", json.dumps(x["gt"], ensure_ascii=False, indent=1))
        print("DECOYS:", json.dumps(x.get("decoys"), ensure_ascii=False))
        print("HARD:", x["hard_cases"], "QC:", x["qc"])


if __name__ == "__main__":
    main()
