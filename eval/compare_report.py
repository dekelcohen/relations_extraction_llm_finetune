"""Compare several models' predictions on one eval set -> summary HTML + .json (incl. flags read by the dataset viewer).

  python eval/compare_report.py --samples data/pilot/samples.jsonl \
      --model sonnet-5.5=data/pilot/preds_strong.jsonl --model llama-3.3-70b=data/pilot/preds_weak.jsonl \
      --model qwen3-4b=outputs/eval/qwen3-4b-think.preds.jsonl --model qwen3-4b-ft=outputs/eval/qwen3-4b-ft-think.preds.jsonl \
      --base qwen3-4b --ft qwen3-4b-ft --viewer_dir data/pilot --out outputs/report/pilot_compare.html

Preds files: jsonl rows with {"id", "raw"} (re_synth predict or eval/predict.py output; <think> is stripped).
--viewer_dir: writes <viewer_dir>/viewer_sources.json pointing at the preds files and this report's .json (flags),
so the dataset viewer reads the same files (no copies) and shows the models side-by-side with interesting samples marked.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from re_synth.hardcases import HARD_CASES  # noqa: E402
from re_synth.mine import load_preds, score_preds  # noqa: E402
from re_synth.score import aggregate, parse_prediction_lenient, sample_f1, score_sample, slot_types_for  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from error_analysis import CAT_DESC, CATS, ROOTS, analyze  # noqa: E402

LENIENT_LEGEND = (
    "Lenient = same scorer, but model outputs that fail strict JSON parsing are repaired first by a deterministic "
    "post-processor (re_synth.score.parse_prediction_lenient), never by an LLM: (1) <b>inner_quotes</b> &ndash; a "
    "<code>\"</code> inside a string is escaped unless the next non-space char is <code>, : } ]</code> (Hebrew "
    "acronyms such as <code>בע\"מ</code>, <code>ש\"ח</code>, <code>תשפ\"ה</code> written with a plain ASCII quote); "
    "(2) <b>trailing_data</b> &ndash; junk after the first complete JSON object is ignored (e.g. an extra <code>}</code>); "
    "(3) <b>trailing_commas</b> removed. Values are not changed. <b>unclosed_think</b> = the model never closed "
    "<code>&lt;think&gt;</code> (token budget exhausted) &ndash; no answer exists, so it stays a failure.")


def score_lenient(samples: list[dict], preds: dict[str, str]) -> tuple[dict[str, dict], dict[str, int]]:
    out, repairs = {}, defaultdict(int)
    for s in samples:
        if s["id"] not in preds:
            continue
        recs, valid, rep = parse_prediction_lenient(preds[s["id"]])
        if rep:
            repairs[rep] += 1
        out[s["id"]] = score_sample(recs, valid, s["gt"], s.get("decoys", []), s["text"],
                                    slot_types_for(s["schema_id"], inline=s.get("schema")))
    return out, dict(repairs)

OK, BAD = 0.99, 0.7  # sample-F1 thresholds: >= OK is correct, < BAD is a clear error
METRICS = ["slot_f1", "slot_p", "slot_r", "rec_f1", "exact_sample", "null_acc", "decoy_hit_rate", "halluc_rate",
           "json_valid"]
LOWER_BETTER = {"decoy_hit_rate", "halluc_rate"}
METRIC_DESC = {
    "n": "Number of pilot samples in this row/group.",
    "slot_f1": "Slot-level F1 (micro, over all samples): harmonic mean of slot_p and slot_r. A slot = one non-main field "
               "(date, org, location, id, status...) of one extracted record. Predicted records are first aligned to gold "
               "records; a value counts as correct if it matches gold after type-aware normalization (dates, digits, "
               "prefixes). A wrong value counts as both a FP and a FN. Higher is better.",
    "slot_p": "Slot precision = correct predicted values / all non-null predicted values (incl. values in unmatched "
              "extra records). Low = model fills slots it shouldn't or with wrong values.",
    "slot_r": "Slot recall = correct predicted values / all non-null gold values (incl. values of missed records). "
              "Low = model misses values (leaves them null or misses whole records).",
    "rec_f1": "Record-level F1: did the model find the right number of main entities (records)? A predicted record is a "
              "TP if it aligns to a gold record by its main entity; slot values are ignored here.",
    "exact_sample": "Exact-sample accuracy: fraction of samples whose output is valid JSON AND every slot of every record "
                    "is right (sample F1 = 1). The strictest metric: one wrong field fails the whole sample.",
    "null_acc": "Null accuracy: among slots whose gold value is null (not stated in the text, e.g. H4/H9), the fraction "
                "the model correctly left null. Low = the model invents / bleeds values.",
    "decoy_hit_rate": "Decoy-hit rate: fraction of planted distractors (same-type decoy values injected near the entity, "
                      "e.g. another org/date) that the model output as a slot value. Lower is better.",
    "halluc_rate": "Hallucination rate: fraction of predicted non-null values that do not appear in the source text "
                   "(after normalization). Lower is better.",
    "json_valid": "Fraction of outputs that strictly parse as the expected JSON (after stripping <think> and code fences).",
    "slot F1 strict": "Slot F1 with the strict parser: unparseable output = empty prediction.",
    "slot F1 lenient": "Slot F1 after deterministic formatting repair (see legend above); values are never changed.",
    "exact lenient": "Exact-sample accuracy after the formatting repair.",
    "Δ": "Fine-tuned minus base Qwen (green = improvement, red = regression; for lower-is-better metrics sign is flipped).",
}
TIER_DESC = {
    "easy": "Both Llama-3.3-70B and Sonnet got the sample fully right (sample F1 >= 0.99).",
    "hard": "Sonnet right, Llama-70B wrong: the target set for a small fine-tuned model.",
    "ambiguous": "Both large models wrong: very hard or likely a gold/label issue.",
    "weird": "Llama-70B right but Sonnet wrong.",
    "frontier_hard": "Sonnet wrong although the gold is deterministic (rendered table/list block, H6/H7/H13): "
                     "hardest trustworthy samples.",
}
HC_DESC = {**HARD_CASES, "H7c": "H7 variant: the shared org is stated only in the table caption"}
NOTES: dict[str, str] = {}


def tip(text: str, desc: str | None, tag: str = "span", attrs: str = "") -> str:
    t = f' title="{html.escape(desc)}"' if desc else ""
    cls = " class=tip" if desc and tag == "span" else ""
    return f"<{tag}{attrs}{cls}{t}>{text}</{tag}>"


def group_desc(g: str) -> str | None:
    k, _, v = g.partition("=")
    if g == "all":
        return "All pilot samples."
    return {"lang": f"Samples in language '{v}'.",
            "tier": f"Difficulty tier '{v}': {TIER_DESC.get(v, '')}",
            "format": f"Document format '{v}' (text / email / html...).",
            "schema": f"Extraction schema '{v}' (which entity type and slots to extract).",
            "hc": f"Hard case {v}: {HC_DESC.get(v, '')}"}.get(k)


def model_th(l: str) -> str:
    return tip(html.escape(l), NOTES.get(l), "th")
FLAG_DESC = {
    "ft_improved": "fine-tuned Qwen fixed it (base < {bad}, FT >= {ok})",
    "ft_regressed": "fine-tuning broke it (base >= {ok}, FT < {bad})",
    "ft_err_frontier_ok": "FT Qwen errs (< {bad}) but the frontier model is correct",
    "ft_err_70b_ok": "FT Qwen errs (< {bad}) but Llama-70B is correct",
    "ft_beats_large": "FT Qwen correct while a large model errs",
    "all_fail": "every model < {bad}: very hard or a GT issue",
}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def groups_of(s: dict, tier: str | None) -> list[str]:
    g = ["all", f"lang={s['lang']}", f"format={s['format']}", f"schema={s['schema_id']}"]
    g += [f"hc={h}" for h in s.get("hard_cases", [])]
    if tier:
        g.append(f"tier={tier}")
    return g


def flags_for(f1: dict[str, float | None], base: str, ft: str, frontier: str | None, oss70: str | None) -> list[str]:
    out = []
    b, t = f1.get(base), f1.get(ft)
    if b is not None and t is not None:
        if b < BAD and t >= OK:
            out.append("ft_improved")
        if b >= OK and t < BAD:
            out.append("ft_regressed")
    if t is not None and t < BAD:
        if frontier and (f1.get(frontier) or 0) >= OK:
            out.append("ft_err_frontier_ok")
        if oss70 and (f1.get(oss70) or 0) >= OK:
            out.append("ft_err_70b_ok")
    if t is not None and t >= OK and any(f1.get(m) is not None and f1[m] < BAD for m in (frontier, oss70) if m):
        out.append("ft_beats_large")
    vals = [v for v in f1.values() if v is not None]
    if vals and max(vals) < BAD:
        out.append("all_fail")
    return out


def fmt(v, metric=None):
    return "" if v is None else f"{v:.3f}"


def cell_color(v, best, worst, lower=False):
    if v is None or best is None or best == worst:
        return ""
    x = (v - worst) / (best - worst)
    return f' style="background:hsl({int(120 * x)},70%,{90 - 8 * x:.0f}%)"'


def table(rows: list[tuple[str, dict]], labels: list[str], metric: str, base: str | None, ft: str | None) -> str:
    h = [f"<table><caption>{tip(metric, METRIC_DESC.get(metric))}</caption><tr>"
         + tip("group", "Subset of samples: language, difficulty tier, format, schema or hard case (hover a row).", "th")
         + tip("n", METRIC_DESC["n"], "th") + "".join(model_th(l) for l in labels)]
    if base and ft:
        h[0] += tip("&Delta; FT&minus;base", METRIC_DESC["Δ"], "th")
    h[0] += "</tr>"
    lower = metric in LOWER_BETTER
    for g, by in rows:
        vals = [by.get(l, {}).get(metric) for l in labels]
        nums = [v for v in vals if v is not None]
        best, worst = ((min(nums), max(nums)) if lower else (max(nums), min(nums))) if nums else (None, None)
        n = next((by[l]["n"] for l in labels if l in by), 0)
        r = f"<tr>{tip(html.escape(g), group_desc(g), 'td', ' class=g')}<td>{n}</td>"
        r += "".join(f"<td{cell_color(v, best, worst)}>{fmt(v)}</td>" for v in vals)
        if base and ft:
            bv, tv = by.get(base, {}).get(metric), by.get(ft, {}).get(metric)
            if bv is None or tv is None:
                r += "<td></td>"
            else:
                d = tv - bv
                good = (d < 0) if lower else (d > 0)
                r += f'<td class="{"up" if good else "down" if d else ""}">{d:+.3f}</td>'
        h.append(r + "</tr>")
    return "\n".join(h) + "</table>"


def bars(labels, by_all, metrics=("slot_f1", "exact_sample", "rec_f1", "null_acc")) -> str:
    W, H, P = 640, 230, 30
    gw = (W - P) / len(metrics)
    bw = gw * 0.8 / len(labels)
    cols = ["#6366f1", "#f59e0b", "#94a3b8", "#10b981", "#ef4444", "#0ea5e9"]
    s = [f'<svg width="{W}" height="{H + 40}" font-family="sans-serif" font-size="11">']
    for i in range(0, 11, 2):
        y = H - i / 10 * (H - 10)
        s.append(f'<line x1="{P}" x2="{W}" y1="{y}" y2="{y}" stroke="#e5e7eb"/><text x="0" y="{y + 4}">{i / 10:.1f}</text>')
    for gi, m in enumerate(metrics):
        x0 = P + gi * gw + gw * 0.1
        for li, l in enumerate(labels):
            v = by_all.get(l, {}).get(m)
            if v is None:
                continue
            bh = v * (H - 10)
            x = x0 + li * bw
            s.append(f'<rect x="{x:.1f}" y="{H - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" fill="{cols[li % 6]}">'
                     f'<title>{html.escape(l)} {m}={v:.3f}</title></rect>'
                     f'<text x="{x + bw / 2 - 1:.1f}" y="{H - bh - 3:.1f}" text-anchor="middle" font-size="9">{v:.2f}</text>')
        s.append(f'<text x="{x0 + gw * 0.4:.1f}" y="{H + 15}" text-anchor="middle" style="cursor:help">{m}'
                 f'<title>{html.escape(METRIC_DESC.get(m, ""))}</title></text>')
    for li, l in enumerate(labels):
        s.append(f'<rect x="{P + li * 150}" y="{H + 25}" width="10" height="10" fill="{cols[li % 6]}"/>'
                 f'<text x="{P + li * 150 + 14}" y="{H + 34}">{html.escape(l)}</text>')
    return "".join(s) + "</svg>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/pilot/samples.jsonl")
    ap.add_argument("--tiers", default="data/pilot/tiers.json")
    ap.add_argument("--eval_ids", default="data/pilot/export/pilot_all.jsonl", help="restrict to ids in this jsonl")
    ap.add_argument("--model", action="append", required=True, help="label=preds.jsonl (repeatable, column order)")
    ap.add_argument("--note", action="append", default=[], help="label=free text (settings shown in the report)")
    ap.add_argument("--base", help="label of the base small model")
    ap.add_argument("--ft", help="label of the fine-tuned small model")
    ap.add_argument("--frontier", help="label of the frontier reference model")
    ap.add_argument("--oss70", help="label of the large open model")
    ap.add_argument("--viewer_dir", help="dataset dir to write viewer_sources.json (paths of preds + flags) into")
    ap.add_argument("--viewer_url", default="", help="base viewer url for sample links, e.g. http://host:8111/")
    ap.add_argument("--out", default="outputs/report/pilot_compare.html")
    a = ap.parse_args()

    samples = read_jsonl(a.samples)
    if a.eval_ids and Path(a.eval_ids).exists():
        keep = {r["id"] for r in read_jsonl(a.eval_ids)}
        samples = [s for s in samples if s["id"] in keep]
    tiers = json.loads(Path(a.tiers).read_text()) if a.tiers and Path(a.tiers).exists() else {}
    models = dict(m.split("=", 1) for m in a.model)
    notes = dict(n.split("=", 1) for n in a.note)
    NOTES.update(notes)
    labels = list(models)

    per = {}  # label -> {id: metrics}
    per_len, repairs, errors, raw_preds = {}, {}, {}, {}
    for l, p in models.items():
        preds = raw_preds[l] = load_preds(p)
        per[l] = score_preds(samples, preds)
        per_len[l], repairs[l] = score_lenient(samples, preds)
        errors[l] = analyze(samples, preds)
        missing = len(samples) - len(per[l])
        if missing:
            print(f"WARNING {l}: {missing} samples have no prediction (scored only on {len(per[l])})")

    groups = defaultdict(lambda: defaultdict(list))  # group -> label -> [metrics]
    groups_len = defaultdict(lambda: defaultdict(list))
    for s in samples:
        for g in groups_of(s, tiers.get(s["id"], {}).get("tier")):
            for l in labels:
                if s["id"] in per[l]:
                    groups[g][l].append(per[l][s["id"]])
                    groups_len[g][l].append(per_len[l][s["id"]])
    agg = {g: {l: aggregate(v) for l, v in by.items() if v} for g, by in groups.items()}
    agg_len = {g: {l: aggregate(v) for l, v in by.items() if v} for g, by in groups_len.items()}
    order = sorted(agg, key=lambda g: (g != "all", g.split("=")[0], len(g), g))

    f1 = {s["id"]: {l: (sample_f1(per[l][s["id"]]) if per[l][s["id"]].get("valid_json") else 0.0)
                    if s["id"] in per[l] else None for l in labels} for s in samples}
    flags = {}
    if a.base and a.ft:
        for sid, d in f1.items():
            fl = flags_for(d, a.base, a.ft, a.frontier, a.oss70)
            if fl:
                flags[sid] = fl
    flag_counts = defaultdict(int)
    for fl in flags.values():
        for f in fl:
            flag_counts[f] += 1

    out = Path(a.out)
    if a.viewer_dir:  # reference (not copy) the sources, so the viewer reads the same files as this report
        def rel(p):
            p = Path(p).resolve()
            return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        man = Path(a.viewer_dir) / "viewer_sources.json"
        man.write_text(json.dumps({"preds": {l: rel(p) for l, p in models.items()}, "flags": rel(out.with_suffix(".json"))},
                                  indent=1), encoding="utf-8")
        print(f"viewer: sources -> {man} ({len(flags)} flagged samples)")
    out.parent.mkdir(parents=True, exist_ok=True)
    main_rows = [(g, agg[g]) for g in order if g == "all" or g.startswith(("lang=", "tier="))]
    detail_rows = [(g, agg[g]) for g in order if g.startswith(("format=", "schema=", "hc="))]
    overall = ("<table><tr><th>model</th>" + tip("n", METRIC_DESC["n"], "th")
               + "".join(tip(m, METRIC_DESC.get(m), "th") for m in METRICS)
               + tip("settings", "Model id, provider and decoding settings.", "th") + "</tr>")
    for l in labels:
        r = agg["all"].get(l, {})
        overall += f"<tr>{tip(html.escape(l), notes.get(l), 'td', ' class=g')}<td>{r.get('n', 0)}</td>" + "".join(
            f"<td>{fmt(r.get(m))}</td>" for m in METRICS) + f"<td class=note>{html.escape(notes.get(l, ''))}</td></tr>"
    overall += "</table>"

    def link(sid):
        return f'<a href="{html.escape(a.viewer_url)}#{sid}" target="_blank">{sid}</a>' if a.viewer_url else sid

    sample_lists = []
    by_id = {s["id"]: s for s in samples}
    for f in FLAG_DESC:
        ids = [i for i, fl in flags.items() if f in fl]
        if not ids:
            continue
        rows = "".join(f"<tr><td>{link(i)}</td><td>{by_id[i]['lang']}</td><td>{by_id[i]['schema_id']}</td>"
                       f"<td>{' '.join(tip(h, HC_DESC.get(h)) for h in by_id[i].get('hard_cases', []))}</td>"
                       + "".join(f"<td>{fmt(f1[i][l])}</td>" for l in labels) + "</tr>" for i in sorted(ids))
        sample_lists.append(
            f"<details><summary><b>{f}</b> ({len(ids)}) &mdash; {FLAG_DESC[f].format(ok=OK, bad=BAD)}</summary>"
            f"<table><tr><th>id</th><th>lang</th><th>schema</th><th>hard cases</th>"
            + "".join(tip(html.escape(l) + " sample F1", f"Sample F1 of {l} (slot F1 within this one sample; 0 if invalid JSON). {notes.get(l, '')}", "th") for l in labels) + f"</tr>{rows}</table></details>")

    # ---- lenient (tolerant JSON) scores, global + per language
    lang_groups = ["all"] + sorted(g for g in agg_len if g.startswith("lang="))
    len_html = "<table><tr><th rowspan=2>model</th>" + "".join(
        tip(g, group_desc(g), "th", " colspan=3") for g in lang_groups) + tip(
        "repairs applied", "How many outputs needed each repair type (see legend above).", "th", " rowspan=2") + "</tr><tr>" + "".join(
        "".join(tip(c, METRIC_DESC[c], "th") for c in ("slot F1 strict", "slot F1 lenient", "exact lenient"))
        for _ in lang_groups) + "</tr>"
    for l in labels:
        len_html += f"<tr>{tip(html.escape(l), notes.get(l), 'td', ' class=g')}"
        for g in lang_groups:
            s_, n_ = agg[g].get(l, {}), agg_len[g].get(l, {})
            d = (n_.get("slot_f1") or 0) - (s_.get("slot_f1") or 0)
            len_html += (f"<td>{fmt(s_.get('slot_f1'))}</td><td><b>{fmt(n_.get('slot_f1'))}</b>"
                         f"{f' <span class=up>(+{d:.3f})</span>' if d > 0.0005 else ''}</td>"
                         f"<td>{fmt(n_.get('exact_sample'))}</td>")
        len_html += f"<td class=note>{html.escape(', '.join(f'{k}: {v}' for k, v in repairs[l].items()) or 'none needed')}</td></tr>"
    len_html += "</table>"

    # ---- error analysis
    err_model = a.ft or labels[-1]

    def count_table(key, keys, desc):
        h = "<table><tr>" + tip("category", "Error category; hover a row (or see 'meaning').", "th") + "".join(
            tip(html.escape(l), f"Number of slot errors of {l} in this category. {notes.get(l, '')}", "th") for l in labels)
        h += tip(f"{html.escape(err_model)}: he / fa / en", f"Errors of {err_model} split by language.", "th") + "<th>meaning</th></tr>"
        for k in keys:
            cnt = {l: sum(1 for e in errors[l] if e[key] == k) for l in labels}
            if not any(cnt.values()):
                continue
            by_lang = " / ".join(str(sum(1 for e in errors[err_model] if e[key] == k and e["lang"] == lg))
                                 for lg in ("he", "fa", "en"))
            h += f"<tr>{tip(k, desc[k], 'td', ' class=g')}" + "".join(f"<td>{cnt[l]}</td>" for l in labels)
            h += f"<td>{by_lang}</td><td class=note>{html.escape(desc[k])}</td></tr>"
        tot = "".join(f"<td><b>{len(errors[l])}</b></td>" for l in labels)
        return h + f"<tr><td class=g>total slot errors</td>{tot}<td></td><td></td></tr></table>"

    ex_html = []
    for r in ROOTS:
        es = [e for e in errors[err_model] if e["root"] == r]
        if not es:
            continue
        rows = "".join(f"<tr><td>{link(e['id'])}</td><td>{e['lang']}</td>{tip(e['cat'], CAT_DESC.get(e['cat']), 'td')}<td>{e['slot'] or ''}</td>"
                       f"<td class=v>{html.escape(str(e['gold']))}</td><td class=v>{html.escape(str(e['pred']))}</td>"
                       f"<td>{' '.join(tip(h, HC_DESC.get(h)) for h in e['hc'])}</td></tr>" for e in es[:12])
        ex_html.append(f"<details><summary><b>{r}</b> ({len(es)}) &mdash; {html.escape(ROOTS[r])}</summary><table>"
                       f"<tr><th>id</th><th>lang</th>{tip('category', 'Mechanical error category (see table below).', 'th')}<th>slot</th><th>gold</th><th>pred</th>"
                       f"<th>hard cases</th></tr>{rows}</table></details>")
    (out.with_suffix(".errors.json")).write_text(json.dumps(errors, ensure_ascii=False, indent=0), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({"agg": agg, "agg_lenient": agg_len, "repairs": repairs,
                                                      "flags": flags, "sample_f1": f1}, indent=1, ensure_ascii=False))
    err_html = (f"<h2>Error analysis</h2><p class=muted>Every gold/pred slot disagreement (on leniently-parsed outputs) "
                f"gets one category, using the scorer's own record alignment and value matching "
                f"(eval/error_analysis.py). Root causes group them by why they happen; examples are for "
                f"<b>{html.escape(err_model)}</b>. Counts are slot-level (a missed record counts once).</p>"
                f"<h3>Root causes</h3>{count_table('root', list(ROOTS), ROOTS)}"
                f"<h3>Examples per root cause ({html.escape(err_model)})</h3>{''.join(ex_html)}"
                f"<details><summary>Mechanical error categories</summary>{count_table('cat', CATS, CAT_DESC)}</details>")

    dl = lambda d: "<dl class=leg>" + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in d.items()) + "</dl>"
    legend = ("<details open><summary><b>Legend</b> &mdash; what the metrics and labels mean "
              "(hover any dotted-underlined header / label for the same text)</summary>"
              "<h3>Metrics</h3>" + dl({k: v for k, v in METRIC_DESC.items() if k in METRICS or k in ('n', 'Δ')})
              + "<h3>Models</h3>" + dl({l: notes.get(l, '') for l in labels})
              + "<h3>Difficulty tiers (from the large models' pilot results)</h3>" + dl({f'tier={k}': v for k, v in TIER_DESC.items()})
              + "<h3>Interesting-sample flags</h3>" + dl({k: v.format(ok=OK, bad=BAD) for k, v in FLAG_DESC.items()})
              + "<h3>Hard cases (hc=)</h3>" + dl(HC_DESC) + "</details>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Pilot comparison</title><style>
body{{font-family:sans-serif;margin:20px;max-width:1300px;color:#222}}table{{border-collapse:collapse;font-size:13px;margin:8px 0}}
td,th{{border:1px solid #d1d5db;padding:3px 7px;text-align:right}}th{{background:#f3f4f6}}td.g{{text-align:left;font-weight:600}}
td.note{{text-align:left;font-size:11px;color:#555;max-width:380px}}td.up{{color:#15803d;font-weight:600}}td.down{{color:#b91c1c;font-weight:600}}
details{{margin:6px 0}}summary{{cursor:pointer}}td.v{{text-align:left;max-width:280px}}span.up{{color:#15803d}}code{{background:#f3f4f6}}.tip,th[title],td[title]{{cursor:help}}.tip,th[title]{{text-decoration:underline dotted #888}}caption{{text-align:left;font-weight:600;padding:2px 0}}dl.leg{{display:grid;grid-template-columns:max-content 1fr;gap:3px 12px;font-size:13px}}dl.leg dt{{font-weight:600;font-family:monospace}}dl.leg dd{{margin:0}}.muted{{color:#666;font-size:12px}}</style></head><body>
<h1>Pilot testset &mdash; model comparison</h1>
<p class=muted>{len(samples)} samples &middot; {html.escape(a.samples)} &middot; generated {time.strftime('%Y-%m-%d %H:%M')}
&middot; sample F1 thresholds: correct &ge; {OK}, error &lt; {BAD}
{f'&middot; <a href="{html.escape(a.viewer_url)}" target="_blank">open sample viewer</a>' if a.viewer_url else ''}</p>
{legend}
<h2>Overall</h2>{overall}{bars(labels, agg['all'])}
<h2>Lenient score: tolerant JSON parsing (global + per language)</h2><p class=muted>{LENIENT_LEGEND}</p>{len_html}
<h2>Slot F1 by language / tier</h2>{table(main_rows, labels, 'slot_f1', a.base, a.ft)}
<h2>Exact-sample accuracy by language / tier</h2>{table(main_rows, labels, 'exact_sample', a.base, a.ft)}
<h2>Interesting samples</h2><p>{' &middot; '.join(f'{tip(k, FLAG_DESC[k].format(ok=OK, bad=BAD))}: <b>{v}</b>' for k, v in flag_counts.items()) or 'none'}</p>
{''.join(sample_lists)}
{err_html}
<h2>Slot F1 by format / schema / hard case</h2>{table(detail_rows, labels, 'slot_f1', a.base, a.ft)}
<details><summary>Decoy-hit rate (lower is better)</summary>{table(main_rows + detail_rows, labels, 'decoy_hit_rate', a.base, a.ft)}</details>
<details><summary>Null accuracy</summary>{table(main_rows + detail_rows, labels, 'null_acc', a.base, a.ft)}</details>
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"report: {out}")
    for l in labels:
        r = agg["all"].get(l, {})
        print(f"  {l:<16} slot_f1={r.get('slot_f1')} exact={r.get('exact_sample')} n={r.get('n')}")


if __name__ == "__main__":
    main()
