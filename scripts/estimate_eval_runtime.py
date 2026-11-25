"""Extrapolate evaluation time from an HF/Unsloth smoke run's prompt throughput."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from re_synth import chunking  # noqa: E402

DEFAULT_DATA = (
    ROOT / "data/pilot/export/pilot_all.jsonl",
    ROOT / "data/test_long/test_long.jsonl",
    ROOT / "data/hard_inss/dev.jsonl",
)


def prompt_count(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(
            max(1, len(chunking.split(json.loads(line)["text"])))
            for line in f if line.strip()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="Smoke metrics; one per model variant")
    parser.add_argument("--data", action="append", type=Path, help="Evaluation JSONL; repeat to override default datasets")
    parser.add_argument("--generation-seconds", type=float, help="Override measured generation time for a single smoke")
    parser.add_argument("--load-seconds", type=float, help="Override measured model load time for a single smoke")
    args = parser.parse_args()
    if len(args.metrics) > 1 and (args.generation_seconds is not None or args.load_seconds is not None):
        parser.error("timing overrides require exactly one smoke metrics file")
    workloads = [(path.name, prompt_count(path)) for path in (args.data or DEFAULT_DATA)]
    total = 0.0
    for path in args.metrics:
        generation = json.loads(path.read_text(encoding="utf-8")).get("generation", {})
        prompts = generation.get("prompts", 0)
        seconds = args.generation_seconds if args.generation_seconds is not None else generation.get("generation_seconds", 0)
        load = args.load_seconds if args.load_seconds is not None else generation.get("load_seconds", 0)
        if prompts <= 0 or not math.isfinite(seconds) or seconds <= 0:
            parser.error(f"{path}: need a completed smoke with prompts and positive generation time")
        if not math.isfinite(load) or load < 0:
            parser.error(f"{path}: expected nonnegative model load time")
        rate = prompts / seconds
        estimate = load + sum(n / rate for _, n in workloads)
        total += estimate
        print(f"{path.name}: {prompts} prompts in {seconds:.1f}s ({rate * 60:.2f} prompts/min); load {load:.1f}s")
        for dataset, n in workloads:
            print(f"  {dataset}: {n} prompts, ~{n / rate / 3600:.2f} h")
        print(f"  Variant total: ~{estimate / 3600:.2f} h")
    if len(args.metrics) > 1:
        print(f"Sequential total: ~{total / 3600:.2f} h")
    print("Rough estimate: assumes matching settings and similar generated lengths; excludes scoring/report overhead.")


if __name__ == "__main__":
    main()
