"""Publish completed pipeline artifacts. Credentials are inherited, never written."""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_sync  # noqa: E402

EVAL = Path("outputs/eval")
REPORT = Path("outputs/report")
COAUTHOR = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
SETS = {
    "pilot": ("data/pilot/export/pilot_all.jsonl", "data/pilot"),
    "grpo_dev": ("data/stage_2_RL_GRPO/grpo_dev.jsonl", "data/stage_2_RL_GRPO"),
    "test_long": ("data/test_long/test_long.jsonl", "data/test_long"),
    "hard_inss": ("data/hard_inss/dev.jsonl", "data/hard_inss"),
}
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def git(*args: str, **kwargs):
    return subprocess.run(
        ["git", "-c", "credential.helper=", "-c",
         'credential.helper=!f() { test "$1" = get || return 0; host=; https=; '
         'while IFS= read -r line && test -n "$line"; do case "$line" in '
         'host=github.com) host=yes;; protocol=https) https=yes;; esac; done; '
         'if test "$host:$https" = yes:yes && test -n "$GH_TOKEN"; then '
         'printf "username=x-access-token\\npassword=%s\\n" "$GH_TOKEN"; fi; }; f',
         "-c", "user.name=Pipeline Publisher", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
         *args], cwd=ROOT, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, check=True, **kwargs)


def preflight():
    from huggingface_hub import HfApi
    if not os.environ.get("HF_TOKEN") or not os.environ.get("GH_TOKEN"):
        raise ValueError("Publishing requires runtime HF_TOKEN and GH_TOKEN")
    # Do not call hf auth login: it would leave credentials on the shared machine.
    who = HfApi(token=os.environ["HF_TOKEN"]).whoami()
    if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
        raise ValueError("HF_TOKEN must have write access to the private model repositories")
    git("push", "--dry-run")
    print("Hugging Face authentication and Git push preflight passed")


def commit(paths: list[Path], message: str):
    files = [p.as_posix() for p in paths]
    for path in paths:
        if (ROOT / path).stat().st_size >= 100_000_000:
            raise ValueError(f"{path} is too large for GitHub; keep it outside Git")
    git("add", "-f", "--", *files)
    if git("diff", "--cached", "--name-only", "--", *files, capture_output=True).stdout:
        git("commit", "--only", "-m", message, "-m", COAUTHOR, "--", *files)
    git("push")


def model(out: str, repo: str | None):
    p = Path(out)
    if p.parts[:1] != ("outputs",) or ".." in p.parts:
        raise ValueError("Model publication requires a repo-relative directory under outputs/")
    p = p.as_posix()
    repo = repo or hf_sync._manifest().get(p, {}).get("repo")
    if not repo:
        name = Path(p).name
        match = re.fullmatch(r"sft-cot-(.+?)(-stage\d+)?", name)
        if match:
            base, stage = match.groups()
            name = f"{base}-sft-cot{stage or ''}"
        repo = f"Shani123/re-{name}"
    hf_sync.push(p, repo, "model", final_model=True)
    commit([Path("hf_artifacts.json")], f"Record published model {p}")


def eval_path(p: Path):
    if p.parent != EVAL or not p.name.endswith(".preds.jsonl"):
        raise ValueError(f"Not a publishable full evaluation: {p}")
    for dataset in SETS:
        suffix = "-think.preds.jsonl" if dataset == "pilot" else f".{dataset}.preds.jsonl"
        if p.name.endswith(suffix):
            name = p.name[:-len(suffix)]
            if LABEL.fullmatch(name) and not name.startswith("smoke."):
                return dataset, name
    raise ValueError(f"Unknown evaluation dataset or label: {p}")


def report(run: str, family: str, outputs: list[str]):
    from tempfile import TemporaryDirectory

    if not LABEL.fullmatch(run) or not LABEL.fullmatch(family):
        raise ValueError("Invalid run name or model family")
    current = {Path(p) for p in outputs}
    for path in current:
        eval_path(path)
    tracked = git("ls-files", "-z", "--", EVAL.as_posix(), capture_output=True).stdout.decode().split("\0")
    by_set: dict[str, dict[str, Path]] = {}
    for path in sorted(current | {Path(x) for x in tracked if x.endswith(".preds.jsonl")}):
        try:
            dataset, name = eval_path(path)
        except ValueError:
            continue
        if path in current or name == family or name.startswith(f"{family}-"):
            if (ROOT / path).is_file() and (ROOT / f"{path.as_posix()}.metrics.json").is_file():
                by_set.setdefault(dataset, {})[name] = path
    if not current <= {p for entries in by_set.values() for p in entries.values()}:
        raise ValueError("Missing predictions or metrics for a completed evaluation")
    files = sorted(current | {Path(f"{p.as_posix()}.metrics.json") for p in current})
    dst = REPORT / f"{run}.html"
    (ROOT / REPORT).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ROOT / REPORT) as tmp:
        draft = Path(tmp) / dst.name
        title = html.escape(run)
        draft.write_text(f"<!doctype html><meta charset=utf-8><title>{title}</title><body>"
                         f"<h1>{title} evaluation</h1></body>", encoding="utf-8")
        for dataset, entries in by_set.items():
            rows, baseline = SETS[dataset]
            specs = []
            for label, kind in (("sonnet-5.5", "strong"), ("llama-3.3-70b", "weak")):
                p = f"{baseline}/preds_{kind}.jsonl"
                if (ROOT / p).is_file():
                    specs.extend(["--model", f"{label}={p}"])
            for label, p in entries.items():
                specs.extend(["--model", f"{label}={p.as_posix()}"])
            common = ["--report", str(draft), "--rows", rows, *specs]
            subprocess.run([sys.executable, "eval/append_section.py", *common, "--id", dataset,
                            "--title", dataset], cwd=ROOT, check=True)
            for label in entries:
                refs = [b for b in entries if b == family or
                        (b.startswith(f"{family}-") and not b.startswith(f"{family}-ft"))]
                subprocess.run([sys.executable, "eval/append_errors.py", *common,
                                "--id", f"errors-{dataset}-{label}", "--title", f"Errors: {label} on {dataset}",
                                "--em", label, "--viewer_dataset", dataset,
                                *(arg for b in refs if b != label for arg in ("--ref", b))],
                               cwd=ROOT, check=True)
        draft.replace(ROOT / dst)
    commit([*files, dst], f"Publish evaluation results for {run}")


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("model")
    p.add_argument("out")
    p.add_argument("--repo")
    p = sub.add_parser("eval")
    p.add_argument("run")
    p.add_argument("family")
    p.add_argument("outputs", nargs="+")
    a = parser.parse_args()
    if a.action == "preflight":
        preflight()
        return
    lock = ROOT / "outputs/pipeline/publish.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if a.action == "model":
            model(a.out, a.repo)
        else:
            report(a.run, a.family, a.outputs)


if __name__ == "__main__":
    main()
