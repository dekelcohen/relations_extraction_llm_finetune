"""Build the cross-dataset summary table (one row per model, one column per eval set) and an HTML nav/dropdown,
inserted near the top of the report (right after <h1>). Reuses outputs/eval/*.metrics.json when present (same
strict scorer as the per-set sections); computes and caches the rest (Llama/Sonnet on sets without a saved file).

python eval/build_summary.py --report outputs/report/pilot_compare.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from re_synth.mine import load_preds, score_preds  # noqa: E402
from re_synth.score import aggregate  # noqa: E402

E = ROOT / "outputs" / "eval"

# label -> {set: preds path}. pilot uses the "-think" naming; grpo_dev/test_long/hard_inss use ".<set>.".
MODELS = {
    "qwen3-4b-ft (s1)": {"pilot": E / "qwen3-4b-ft-think.preds.jsonl", "grpo_dev": E / "sft.grpo_dev.preds.jsonl",
                         "test_long": E / "sft.test_long.preds.jsonl"},
    "qwen3-4b-grpo": {"pilot": E / "qwen3-4b-grpo-think.preds.jsonl", "grpo_dev": E / "grpo-final.grpo_dev.preds.jsonl",
                      "test_long": E / "grpo.test_long.preds.jsonl", "hard_inss": E / "grpo.hard_inss.preds.jsonl"},
    "qwen3-4b-ft (s2 self-distill)": {"pilot": E / "qwen3-4b-ft-stage2-awq-think.preds.jsonl",
                                      "grpo_dev": E / "qwen3-4b-ft-stage2-awq.grpo_dev.preds.jsonl",
                                      "test_long": E / "qwen3-4b-ft-stage2-awq.test_long.preds.jsonl",
                                      "hard_inss": E / "qwen3-4b-ft-stage2-awq.hard_inss.preds.jsonl"},
    "qwen3.5-9b-ft (s1)": {"pilot": E / "qwen3.5-9b-ft-think.preds.jsonl",
                           "grpo_dev": E / "qwen3.5-9b-ft.grpo_dev.preds.jsonl",
                           "test_long": E / "qwen3.5-9b-ft.test_long.preds.jsonl",
                           "hard_inss": E / "qwen3.5-9b-ft.hard_inss.preds.jsonl"},
    "qwen3.5-9b-ft (s2 self-distill)": {"pilot": E / "qwen3.5-9b-ft-stage2-think.preds.jsonl",
                                        "grpo_dev": E / "qwen3.5-9b-ft-stage2.grpo_dev.preds.jsonl",
                                        "test_long": E / "qwen3.5-9b-ft-stage2.test_long.preds.jsonl",
                                        "hard_inss": E / "qwen3.5-9b-ft-stage2.hard_inss.preds.jsonl"},
    "qwen3.8-27b base (medium)": {"pilot": E / "qwen3.8-27b-int4-medium-think.preds.jsonl",
                                  "grpo_dev": E / "qwen3.8-27b-int4-medium.grpo_dev.preds.jsonl",
                                  "test_long": E / "qwen3.8-27b-int4-medium.test_long.preds.jsonl",
                                  "hard_inss": E / "qwen3.8-27b-int4-medium.hard_inss.preds.jsonl"},
    "qwen3.8-27b-ft s1 (medium)": {"pilot": E / "qwen3.8-27b-ft-int4-medium-think.preds.jsonl",
                                   "grpo_dev": E / "qwen3.8-27b-ft-int4-medium.grpo_dev.preds.jsonl",
                                   "test_long": E / "qwen3.8-27b-ft-int4-medium.test_long.preds.jsonl",
                                   "hard_inss": E / "qwen3.8-27b-ft-int4-medium.hard_inss.preds.jsonl"},
    "qwen3.8-27b-ft s2 (medium)": {"pilot": E / "qwen3.8-27b-ft-stage2-int4-medium-think.preds.jsonl",
                                   "grpo_dev": E / "qwen3.8-27b-ft-stage2-int4-medium.grpo_dev.preds.jsonl",
                                   "test_long": E / "qwen3.8-27b-ft-stage2-int4-medium.test_long.preds.jsonl",
                                   "hard_inss": E / "qwen3.8-27b-ft-stage2-int4-medium.hard_inss.preds.jsonl"},
    "llama-3.3-70b": {"pilot": ROOT / "data/pilot/preds_weak.jsonl", "grpo_dev": ROOT / "data/stage_2_RL_GRPO/preds_weak.jsonl",
                      "test_long": ROOT / "data/test_long/preds_weak.jsonl", "hard_inss": ROOT / "data/hard_inss/preds_weak.jsonl"},
    "sonnet-5.5": {"pilot": ROOT / "data/pilot/preds_strong.jsonl", "grpo_dev": ROOT / "data/stage_2_RL_GRPO/preds_strong.jsonl",
                   "test_long": ROOT / "data/test_long/preds_strong.jsonl"},
}
SETS = ["pilot", "grpo_dev", "test_long", "test_long (70, no trunc)", "hard_inss"]

SAMPLE_SRC = {
    "pilot": ("data/pilot/samples.jsonl",), "grpo_dev": ("data/stage_2_RL_GRPO/samples_he.jsonl", "data/stage_2_RL_GRPO/samples_fa.jsonl"),
    "test_long": ("data/test_long/samples_he.jsonl", "data/test_long/samples_fa.jsonl"),
    "hard_inss": ("data/hard_inss/pool.jsonl",),
}
EVAL_IDS = {"pilot": "data/pilot/export/pilot_all.jsonl", "grpo_dev": "data/stage_2_RL_GRPO/grpo_dev.jsonl",
           "test_long": "data/test_long/test_long.jsonl", "hard_inss": "data/hard_inss/dev.jsonl"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def samples_for(set_name: str) -> list[dict]:
    pool = {s["id"]: s for f in SAMPLE_SRC[set_name] for s in read_jsonl(ROOT / f)}
    ids = [r["id"] for r in read_jsonl(ROOT / EVAL_IDS[set_name])]
    return [pool[i] for i in ids if i in pool]  # drops duplicate ids (e.g. hard_inss dev has one repeat)


def truncated_ids(path: Path) -> set[str]:
    """Doc ids in a test_long preds file whose chunked output(s) never close <think> or whose last chunk isn't
    valid JSON at the end -- the model ran out of tokens."""
    bad = set()
    for r in read_jsonl(path):
        chunks = r.get("chunks") or [r["raw"]]
        if isinstance(chunks, str):
            import ast
            chunks = ast.literal_eval(chunks)
        if any("</think>" not in c or not c.rstrip().endswith("}") for c in chunks):
            bad.add(r["id"])
    return bad


def slot_f1(samples: list[dict], preds_path: Path, keep_ids: set[str] | None = None) -> float | None:
    preds = load_preds(preds_path)
    if keep_ids is not None:
        samples = [s for s in samples if s["id"] in keep_ids]
    res = score_preds(samples, preds)
    if not res:
        return None
    return aggregate(list(res.values())).get("slot_f1")


def metrics_slot_f1(path: Path) -> float | None:
    mp = path.with_suffix(path.suffix + ".metrics.json")
    if mp.exists():
        return json.load(mp.open()).get("report", {}).get("all", {}).get("slot_f1")
    return None


def training_runs_table() -> str:
    """Render docs/reproduce.md's "## 3. Training runs" markdown table as HTML, so the report stays in sync with
    the single source of truth (no hand-duplicated hyperparams)."""
    text = (ROOT / "docs/reproduce.md").read_text(encoding="utf-8")
    m = re.search(r"## 3\. Training runs\n+(\|.*\n)+", text)
    if not m:
        return ""
    lines = [l for l in m.group(0).splitlines()[1:] if l.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[2:]:  # skip the --- separator row
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows.append(cells)
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in header)

    def cell(c: str) -> str:
        c = html.escape(c)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    tbody = "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        "<h2 id='train-runs'>Training runs</h2>"
        "<p class=muted>Source of truth: <code>docs/reproduce.md</code> &sect;3 (regenerate this section by re-running "
        "<code>eval/build_summary.py</code> after editing that file).</p>"
        f"<table class=small><tr>{thead}</tr>{tbody}</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "outputs/report/pilot_compare.html"))
    a = ap.parse_args()

    cache = {s: samples_for(s) for s in ("pilot", "grpo_dev", "test_long", "hard_inss")}
    trunc_ids = truncated_ids(E / "qwen3-4b-ft-stage2-awq.test_long.preds.jsonl")
    keep70 = {s["id"] for s in cache["test_long"]} - trunc_ids
    print(f"test_long: excluding {len(trunc_ids)} truncated docs -> {len(keep70)} kept")

    table = {}
    for label, by_set in MODELS.items():
        row = {}
        for s in ("pilot", "grpo_dev", "test_long", "hard_inss"):
            p = by_set.get(s)
            if not p or not p.exists():
                row[s] = None
                continue
            v = metrics_slot_f1(p)
            if v is None:  # not pre-scored (Llama/Sonnet on most sets) -> compute once, cache next to the preds
                v = slot_f1(cache[s], p)
                if v is not None:
                    mp = p.with_suffix(p.suffix + ".metrics.json")
                    mp.write_text(json.dumps({"report": {"all": {"slot_f1": round(v, 4), "n": len(cache[s])}}}, indent=1))
            row[s] = v
        p = by_set.get("test_long")
        row["test_long (70, no trunc)"] = slot_f1(cache["test_long"], p, keep70) if p and p.exists() else None
        table[label] = row

    def fmt(v):
        return f"{v:.3f}" if v is not None else "&ndash;"

    best = {s: max((table[l][s] for l in table if table[l][s] is not None), default=None) for s in SETS}
    rows = ""
    for label, row in table.items():
        cells = "".join(f"<td{' class=best' if row[s] == best[s] and row[s] is not None else ''}>{fmt(row[s])}</td>" for s in SETS)
        rows += f"<tr><td class=g>{html.escape(label)}</td>{cells}</tr>"
    summary = (
        "<h2 id='summary'>Summary: slot-F1 per model &times; eval set</h2>"
        "<p class=muted>pilot = 298 synthetic samples (all tiers/langs); grpo_dev = 60 held-out hard docs (never trained on); "
        "test_long = 80 held-out long realistic docs, he+fa (never trained on); <b>test_long (70, no trunc)</b> = the same "
        "set minus the 10 docs where qwen3-4b-ft (s2) hit the token limit, so every model is compared on the identical 70 docs; "
        "hard_inss = 39 hard INSS-hosted docs, he+fa, up to 9.6k chars (never trained on). "
        "Bold/shaded = best in that column. &ndash; = not run on that set.</p>"
        "<table><tr><th>model</th>" + "".join(f"<th>{html.escape(s)}</th>" for s in SETS) + "</tr>" + rows + "</table>")

    nav_targets = [("summary", "Summary"), ("train-runs", "Training runs"), ("", "Pilot overview"),
                   ("err-" + "qwen3.8-27b-ft-int4", "Pilot errors (27B-ft-s1)"),
                   ("err35_pilot", "Pilot errors (9B s2 vs s1)"),
                   ("grpo_dev", "grpo_dev"), ("err27_grpo_dev", "grpo_dev errors (27B-ft-s1)"),
                   ("err35_grpo_dev", "grpo_dev errors (9B s2 vs s1)"), ("test_long", "test_long"),
                   ("err27_test_long", "test_long errors (27B-ft-s1)"), ("err4b_test_long", "test_long errors (4B-ft-s2)"),
                   ("err35_test_long", "test_long errors (9B s2 vs s1)"),
                   ("hard_inss", "hard_inss"), ("err27_hard_inss", "hard_inss errors (27B-ft-s1)"),
                   ("err4b_hard_inss", "hard_inss errors (4B-ft-s2)"), ("err35_hard_inss", "hard_inss errors (9B s2 vs s1)"),
                   ("gap27", "Strong-gap (27B wrong, Sonnet+Llama right)")]
    nav = ("<nav style='margin:8px 0'><label>Jump to: <select onchange=\"if(this.value) location.hash=this.value\">"
           + "".join(f"<option value='{html.escape(t)}'>{html.escape(n)}</option>" for t, n in nav_targets)
           + "</select></label></nav>")
    train_runs = training_runs_table()

    h = a.report if isinstance(a.report, str) else str(a.report)
    doc = Path(h).read_text(encoding="utf-8")
    doc = re.sub(r"<!--sec:top-summary-->.*?<!--/sec:top-summary-->", "", doc, flags=re.S)
    block = f"<!--sec:top-summary-->{nav}{summary}{train_runs}<!--/sec:top-summary-->"
    m = re.search(r"</h1>", doc)
    doc = doc[:m.end()] + block + doc[m.end():] if m else block + doc
    if "td.best{" not in doc:
        doc = doc.replace("</style>", "td.best{background:#dcfce7;font-weight:600}nav select{font-size:13px;padding:2px}"
                                       "table.small{font-size:12px}table.small td,table.small th{max-width:260px}</style>")
    Path(h).write_text(doc, encoding="utf-8")
    print(f"summary -> {h}")
    for label, row in table.items():
        print(f"  {label:<32}" + " ".join(f"{s}={fmt(row[s])}" for s in SETS))


if __name__ == "__main__":
    main()
