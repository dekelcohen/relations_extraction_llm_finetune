"""Push / pull large artifacts (adapters, checkpoints, big datasets) to private Hugging Face Hub repos.

Git holds code, docs and small data; anything over GitHub's 100 MB file limit goes to HF Hub and is listed in
hf_artifacts.json ({local_path: {"repo": "<user>/<name>", "type": "model"|"dataset"}}).
Token: HF_TOKEN in the repo-root .env (git-ignored), or a prior `hf auth login`.

  python scripts/hf_sync.py push outputs/sft-cot-qwen3-4b --repo Shani123/re-qwen3-4b-sft-cot
  python scripts/hf_sync.py pull                 # every artifact listed in hf_artifacts.json
  python scripts/hf_sync.py pull outputs/sft-cot-qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "hf_artifacts.json"


def _token() -> str | None:
    env = ROOT / ".env"
    if not os.environ.get("HF_TOKEN") and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def push(path: str, repo: str, kind: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / path), repo_id=repo, repo_type=kind,
                      commit_message=f"upload {path}", ignore_patterns=["*.lock", "__pycache__/*"])
    m = _manifest()
    m[path] = {"repo": repo, "type": kind}
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
    print(f"pushed {path} -> https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


def pull(paths: list[str]) -> None:
    from huggingface_hub import snapshot_download
    m = _manifest()
    for p in paths or list(m):
        a = m[p]
        snapshot_download(a["repo"], repo_type=a["type"], local_dir=str(ROOT / p), token=_token())
        print(f"pulled {a['repo']} -> {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("path", help="folder relative to the repo root")
    p.add_argument("--repo", required=True)
    p.add_argument("--type", default="model", choices=["model", "dataset"])
    p = sub.add_parser("pull")
    p.add_argument("paths", nargs="*")
    a = ap.parse_args()
    push(a.path, a.repo, a.type) if a.cmd == "push" else pull(a.paths)


if __name__ == "__main__":
    main()
