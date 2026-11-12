"""Build the dataset-viewer payload (samples + schema metadata + prompt-chain previews) from a
samples.jsonl (+ optional preds/tiers) directory.

Used two ways:
- As a library by `viewer/server.py` (dynamic mode: loads straight from disk on each request).
- As a CLI that writes a static `viewer/data.js`, for opening the viewer with no server at all:
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
from re_synth.prompt_chain import build_prompt_chain  # noqa: E402
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


def build_dataset_payload(in_dir: Path, limit: int = 0) -> dict:
    samples = read_jsonl(in_dir / "samples.jsonl")
    if limit:
        samples = samples[:limit]

    tiers_path = in_dir / "tiers.json"
    tiers = json.loads(tiers_path.read_text(encoding="utf-8")) if tiers_path.exists() else {}

    # Auto-discover any preds_<model>.jsonl file (not just weak/strong), so new baselines / model
    # runs (e.g. Phase 0's Qwen3-8B, Qwen3.8-27B, Llama-3.3-70B, frontier models) show up automatically.
    preds_by_model = {}
    for path in sorted(in_dir.glob("preds_*.jsonl")):
        model = path.stem[len("preds_"):]
        rows = read_jsonl(path)
        if rows:
            preds_by_model[model] = {r["id"]: r["raw"] for r in rows}

    samples_payload = [build_sample(s, tiers, preds_by_model) for s in samples]
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
    for path in sorted(root.rglob("samples.jsonl")):
        d = path.parent
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pilot", help="directory with samples.jsonl (+ preds_*.jsonl, tiers.json)")
    ap.add_argument("--out", default="viewer/data.js")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    args = ap.parse_args()

    data = build_dataset_payload(Path(args.inp), args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `var` (not `const`) so the global binding is a real property of `window`/`globalThis` in
    # every environment, including when the two scripts are `eval`'d separately (e.g. in tests).
    out_path.write_text("var DATASET = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Wrote {len(data['samples'])} samples ({len(data['prompt_chains'])} prompt chain(s)) to {out_path}")


if __name__ == "__main__":
    main()
