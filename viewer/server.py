"""Dynamic dataset-viewer server: serves the static frontend and loads a chosen dataset from disk
on each request (so new samples/preds/tiers show up without regenerating anything).

Install:  pip install -r viewer/requirements.txt
Run:      python viewer/server.py [--data-root data] [--host 127.0.0.1] [--port 8000]
Then open http://127.0.0.1:8000 and pick a dataset from the dropdown (defaults to "testset" = data/pilot).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from scripts.viewer_data import build_dataset_payload, discover_datasets  # noqa: E402

app = FastAPI(title="RE dataset viewer")


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-cache"  # always revalidate: viewer code/data change often
    return resp
DATA_ROOT = ROOT / "data"


@app.get("/api/datasets")
def list_datasets():
    return discover_datasets(DATA_ROOT)


@app.get("/api/dataset/{name:path}")
def get_dataset(name: str, limit: int = 0):
    datasets = {d["name"]: d for d in discover_datasets(DATA_ROOT)}
    if name not in datasets:
        raise HTTPException(404, f"unknown dataset {name!r}; available: {sorted(datasets)}")
    return build_dataset_payload(Path(datasets[name]["path"]), limit=limit)


# Static frontend (index.html, app.js, style.css) last, so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="static")


def main() -> None:
    import uvicorn

    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="directory to scan for datasets (default: data/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    DATA_ROOT = Path(args.data_root)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
