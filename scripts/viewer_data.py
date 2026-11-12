"""Build the dataset-viewer payload (samples + schema metadata + prompt-chain previews) for
`viewer/server.py`, read straight from the source files on every request -- nothing is copied or exported.

A dataset is a directory with samples.jsonl (+ optional tiers.json, preds_<model>.jsonl). Predictions and
flags that live elsewhere (e.g. outputs/eval/*.preds.jsonl) are referenced, not copied, via an optional
`viewer_sources.json` in the dataset dir (written by eval/compare_report.py):
    {"preds": {"<label>": "<path>" | ["<path>", ...], ...}, "flags": "<report .json with a 'flags' key>"}
Relative paths are resolved against the repo root.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from re_synth.hardcases import HARD_CASES  # noqa: E402
from re_synth.score import align, match, parse_prediction, sample_f1, score_sample, slot_types_for  # noqa: E402
from re_synth.prompt_chain import build_prompt_chain  # noqa: E402
from re_synth.schemas import SCHEMAS  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_pred(raw: str) -> list[dict] | None:
    """Best-effort parse of a model's raw output (incl. <think>...</think> + fenced JSON) into records."""
    recs, valid = parse_prediction(raw)
    return recs if valid else None


def reasoning_of(raw) -> str | None:
    m = re.search(r"<think>(.*?)(?:</think>|$)", raw, flags=re.S) if isinstance(raw, str) else None
    return m.group(1).strip() or None if m else None


def build_schema_meta() -> dict:
    return {
        sid: {
            "main_type": s.main_type,
            "slots": [{"name": sl.name, "type": sl.type} for sl in s.slots],
        }
        for sid, s in SCHEMAS.items()
    }


def _f1(s: dict, raw: str) -> float | None:
    try:
        recs, valid = parse_prediction(raw)
        m = score_sample(recs, valid, s["gt"], s.get("decoys", []), s["text"],
                         slot_types_for(s["schema_id"], inline=s.get("schema")))
        return round(sample_f1(m) if valid else 0.0, 4)
    except Exception:
        return None


def _verdicts(s: dict, recs: list[dict] | None) -> list[dict] | None:
    """Per gold record: {slot: {"v": ok|wrong|missing|null_ok|spurious, "pred": value}}, using the scorer's own
    record alignment and value matching (so the viewer agrees with slot_f1, incl. partial-span matches)."""
    if recs is None or "gt" not in s:
        return None
    gold = s["gt"].get("records", [])
    types = slot_types_for(s["schema_id"], inline=s.get("schema"))
    to_pred = {j: i for i, j in align(recs, gold, types)}
    out = []
    for j, g in enumerate(gold):
        p = recs[to_pred[j]] if j in to_pred else {}
        row = {}
        for k, typ in [("main_entity", "text"), *types.items()]:
            gv, pv = g.get(k), p.get(k)
            if pv in ("", []):
                pv = None
            v = ("ok" if match(pv, gv, typ) else "wrong") if gv is not None and pv is not None else \
                "missing" if gv is not None else "spurious" if pv is not None else "null_ok"
            row[k] = {"v": v, "pred": pv}
        out.append(row)
    return out


def build_sample(s: dict, tiers: dict, preds_by_model: dict[str, dict], flags: dict | None = None) -> dict:
    sid = s["id"]
    preds = {}
    for model, by_id in preds_by_model.items():
        raw = by_id.get(sid)
        if raw is not None:
            recs = parse_pred(raw)
            preds[model] = {"raw": raw, "records": recs, "reasoning": reasoning_of(raw),
                            "f1": _f1(s, raw) if "gt" in s else None, "verdicts": _verdicts(s, recs)}
    tier_info = tiers.get(sid, {})
    return {
        "id": sid,
        "lang": s.get("lang"),
        "format": s.get("format"),
        "schema_id": s.get("schema_id"),
        "genre": s.get("genre"),
        "hard_cases": s.get("hard_cases", []),
        "tier": tier_info.get("tier"),
        "weak_f1": tier_info.get("weak_f1"),
        "strong_f1": tier_info.get("strong_f1"),
        "text": s.get("text", ""),
        "records": s.get("gt", {}).get("records", []),
        "decoys": s.get("decoys", []),
        "preds": preds,
        "flags": (flags or {}).get(sid, []),
    }


def build_prompt_chains(samples_payload: list[dict]) -> dict:
    """One illustrative prompt chain per schema actually present, grounded with a real example
    sample (prefers an 'easy'-tier one so the fewshot/output example is a clean, uncontested case)."""
    by_schema: dict[str, dict] = {}
    for s in samples_payload:
        sid = s["schema_id"]
        if sid not in SCHEMAS:
            continue
        if sid not in by_schema or (s.get("tier") == "easy" and by_schema[sid].get("tier") != "easy"):
            by_schema[sid] = s
    return {sid: build_prompt_chain(SCHEMAS[sid], example) for sid, example in by_schema.items()}


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_samples(in_dir: Path) -> list[dict]:
    """samples.jsonl, or the per-language samples_<x>.jsonl files of a dataset dir (stage 2)."""
    f = in_dir / "samples.jsonl"
    return read_jsonl(f) if f.exists() else [r for p in sorted(in_dir.glob("samples_*.jsonl")) for r in read_jsonl(p)]


def build_dataset_payload(in_dir: Path, limit: int = 0) -> dict:
    samples = read_samples(in_dir)
    if limit:
        samples = samples[:limit]

    tiers_path = in_dir / "tiers.json"
    tiers = json.loads(tiers_path.read_text(encoding="utf-8")) if tiers_path.exists() else {}

    # preds_<model>.jsonl stored in the dataset dir itself, plus sources referenced by viewer_sources.json.
    pred_paths = {p.stem[len("preds_"):]: [p] for p in sorted(in_dir.glob("preds_*.jsonl"))}
    src_path = in_dir / "viewer_sources.json"
    sources = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    for label, p in sources.get("preds", {}).items():  # manifest label wins over a preds_<x>.jsonl of the same file
        paths = [_resolve(x) for x in ([p] if isinstance(p, str) else p)]  # a label may span several files
        resolved = {x.resolve() for x in paths}
        pred_paths = {k: v for k, v in pred_paths.items() if not {x.resolve() for x in v} & resolved} | {label: paths}
    preds_by_model = {}
    for model, path in pred_paths.items():
        rows = [r for x in path if x.exists() for r in read_jsonl(x)]
        if rows:
            preds_by_model[model] = {r["id"]: r["raw"] for r in rows}

    # Interesting-sample flags {sample_id: [flag, ...]} from eval/compare_report.py's .json output.
    flags = {}
    if sources.get("flags") and _resolve(sources["flags"]).exists():
        flags = json.loads(_resolve(sources["flags"]).read_text(encoding="utf-8")).get("flags", {})

    samples_payload = [build_sample(s, tiers, preds_by_model, flags) for s in samples]
    return {
        "source": str(in_dir).replace("\\", "/"),
        "hard_case_descriptions": HARD_CASES,
        "schemas": build_schema_meta(),
        "prompt_chains": build_prompt_chains(samples_payload),
        "samples": samples_payload,
    }


def discover_datasets(root: Path, max_depth: int = 3) -> list[dict]:
    """Find every directory under `root` that contains samples.jsonl (e.g. data/pilot,
    data/synth/build), for the viewer's dataset-selector dropdown. data/pilot is labeled
    "testset" and is the default."""
    root = Path(root)
    found = []
    if not root.exists():
        return found
    base_depth = len(root.parts)
    dirs = sorted({p.parent for pat in ("samples.jsonl", "samples_*.jsonl") for p in root.rglob(pat)})
    for d in dirs:
        if len(d.parts) - base_depth > max_depth:
            continue
        rel = d.relative_to(root).as_posix()
        is_default = rel == "pilot"
        found.append({
            "name": rel,
            "path": str(d).replace("\\", "/"),
            "label": "testset (pilot)" if is_default else rel,
            "default": is_default,
        })
    if found and not any(d["default"] for d in found):
        found[0]["default"] = True
    return found
