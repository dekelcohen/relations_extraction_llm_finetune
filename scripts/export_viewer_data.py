"""Build viewer/data.js from a samples.jsonl (+ optional preds/tiers) directory for the HTML dataset viewer.

Usage:
    python scripts/export_viewer_data.py --in data/pilot --out viewer/data.js
    python scripts/export_viewer_data.py --in data/synth/build --out viewer/data.js --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from re_synth.hardcases import HARD_CASES  # noqa: E402
from re_synth.schemas import SCHEMAS  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_pred(raw: str) -> list[dict] | None:
    """Best-effort parse of a model's raw JSON output into a list of {main_entity, <slots>} records."""
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(out, dict):
        out = out.get("records", out.get("entities"))
    return out if isinstance(out, list) else None


def build_schema_meta() -> dict:
    return {
        sid: {
            "main_type": s.main_type,
            "slots": [{"name": sl.name, "type": sl.type} for sl in s.slots],
        }
        for sid, s in SCHEMAS.items()
    }


def build_sample(s: dict, tiers: dict, preds_by_model: dict[str, dict]) -> dict:
    sid = s["id"]
    preds = {}
    for model, by_id in preds_by_model.items():
        raw = by_id.get(sid)
        if raw is not None:
            preds[model] = {"raw": raw, "records": parse_pred(raw)}
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pilot", help="directory with samples.jsonl (+ preds_*.jsonl, tiers.json)")
    ap.add_argument("--out", default="viewer/data.js")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    args = ap.parse_args()

    in_dir = Path(args.inp)
    samples = read_jsonl(in_dir / "samples.jsonl")
    if args.limit:
        samples = samples[: args.limit]

    tiers_path = in_dir / "tiers.json"
    tiers = json.loads(tiers_path.read_text(encoding="utf-8")) if tiers_path.exists() else {}

    preds_by_model = {}
    for model in ("weak", "strong"):
        rows = read_jsonl(in_dir / f"preds_{model}.jsonl")
        if rows:
            preds_by_model[model] = {r["id"]: r["raw"] for r in rows}

    data = {
        "source": str(in_dir).replace("\\", "/"),
        "hard_case_descriptions": HARD_CASES,
        "schemas": build_schema_meta(),
        "samples": [build_sample(s, tiers, preds_by_model) for s in samples],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `var` (not `const`) so the global binding is a real property of `window`/`globalThis` in
    # every environment, including when the two scripts are `eval`'d separately (e.g. in tests).
    out_path.write_text("var DATASET = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Wrote {len(data['samples'])} samples ({len(preds_by_model)} pred model(s)) to {out_path}")


if __name__ == "__main__":
    main()
