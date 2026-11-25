"""Build the Qwen3-8B before/after report against cached baseline predictions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
BASE = "qwen3-8b-bf16"
STAGE1 = "qwen3-8b-ft-bf16"
STAGE2 = "qwen3-8b-ft-stage2-bf16"
DATASETS = {
    "pilot": ("data/pilot", "export/pilot_all.jsonl", "-think"),
    "test_long": ("data/test_long", "test_long.jsonl", ".test_long"),
    "hard_inss": ("data/hard_inss", "dev.jsonl", ".hard_inss"),
}
BASELINES = ("llama-3.3-70b", "sonnet-5.5", "qwen3-4b-ft-stage2-awq", "qwen3.8-27b-ft-stage2-int4-medium")


def ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def build_report(eval_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        draft = Path(tmp) / out.name
        links = "".join(
            f'<li>{name}: <a href="#{name}">scores</a> | '
            f'<a href="#{name}-before">base errors</a> | <a href="#{name}-after">stage-2 errors</a></li>'
            for name in DATASETS
        )
        draft.write_text(
            '<!doctype html><meta charset="utf-8"><title>Qwen3-8B evaluation</title><body>'
            "<h1>Qwen3-8B (vLLM bf16 base + LoRA): base vs stage-1 vs stage-2 SFT</h1>"
            "<p>References: cached Llama 3.3 70B, Sonnet 5.5, Qwen3-4B stage 2 (AWQ), "
            "and Qwen3.8-27B stage 2 (INT4, medium reasoning).</p>"
            f"<ul>{links}</ul></body>",
            encoding="utf-8",
        )
        for name, (directory, row_file, suffix) in DATASETS.items():
            directory = ROOT / directory
            rows = directory / row_file
            expected = ids(rows)
            models = {
                BASE: eval_dir / f"{BASE}{suffix}.preds.jsonl",
                STAGE1: eval_dir / f"{STAGE1}{suffix}.preds.jsonl",
                STAGE2: eval_dir / f"{STAGE2}{suffix}.preds.jsonl",
                BASELINES[0]: directory / "preds_weak.jsonl",
                BASELINES[1]: directory / "preds_strong.jsonl",
                **{label: ROOT / "outputs/eval" / f"{label}{suffix}.preds.jsonl" for label in BASELINES[2:]},
            }
            common = ["--report", str(draft), "--rows", str(rows)]
            for label, path in models.items():
                actual = ids(path)
                if actual != expected:
                    raise ValueError(f"{name}/{label}: {len(expected - actual)} missing, "
                                     f"{len(actual - expected)} unexpected sample IDs")
                common.extend(("--model", f"{label}={path}"))
            subprocess.run([sys.executable, str(ROOT / "eval/append_section.py"), *common,
                            "--id", name, "--title", f"{name}: scores"], check=True)
            for phase, model, other in (("before", BASE, STAGE2), ("after", STAGE2, BASE)):
                refs = [arg for label in (other, STAGE1, *BASELINES) for arg in ("--ref", label)]
                subprocess.run([sys.executable, str(ROOT / "eval/append_errors.py"), *common,
                                "--id", f"{name}-{phase}", "--title", f"{name}: {model} errors",
                                "--em", model, "--viewer_dataset", name, *refs], check=True)
        draft.replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "outputs/eval")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/report/qwen3-8b.html")
    args = parser.parse_args()
    build_report(args.eval_dir.resolve(), args.out.resolve())
