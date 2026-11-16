"""Where does Qwen3.8-27B fail while BOTH Sonnet-5.5 and Llama-70B are right?

python eval/strong_gap.py [--report outputs/report/pilot_compare.html]
Per set (pilot, grpo_dev, test_long) and per model (27B-ft-stage-1 = ft-int4, 27B INT4 base): slot/record errors whose
(id, record, slot) is not an error of Sonnet nor of Llama (and the strong models have no record-level error in that sample).
Items the audit marked gold_wrong / gold_incomplete / ambiguous / scorer_issue are dropped. Writes outputs/audit/27b_vs_strong.{md,json}
and (with --report) a section into the HTML report.
"""
import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
from error_analysis import ROOTS, analyze, audit_verdict, load_audit  # noqa: E402
from compare_report import HC_DESC  # noqa: E402

E, P, D, T = "outputs/eval", "data/pilot", "data/stage_2_RL_GRPO", "data/test_long"
MODELS = {"27B-ft-stage-1": "qwen3.8-27b-ft-int4", "27B-INT4-base": "qwen3.8-27b-int4"}
DROP = {"gold_wrong", "gold_incomplete", "ambiguous", "scorer_issue"}
SETS = {
    "pilot": dict(rows=None, strong=f"{P}/preds_strong.jsonl", weak=f"{P}/preds_weak.jsonl",
                  m=lambda l: f"{E}/{l}-think.preds.jsonl"),
    "grpo_dev": dict(rows=f"{D}/grpo_dev.jsonl", strong=f"{D}/preds_strong.jsonl", weak=f"{D}/preds_weak.jsonl",
                     m=lambda l: f"{E}/{l}.grpo_dev.preds.jsonl"),
    "test_long": dict(rows=f"{T}/test_long.jsonl", strong=f"{T}/preds_strong.jsonl", weak=f"{T}/preds_weak.jsonl",
                      m=lambda l: f"{E}/{l}.test_long.preds.jsonl"),
}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def samples_of(cfg):
    if cfg["rows"] is None:
        keep = {r["id"] for r in load(f"{P}/export/pilot_all.jsonl")}
        return [s for s in load(f"{P}/samples.jsonl") if s["id"] in keep]
    out = []
    for r in load(cfg["rows"]):
        m = r.get("meta", {})
        out.append({"id": r["id"], "lang": m.get("lang"), "schema_id": m.get("schema_id"), "hard_cases": m.get("hard_cases", []),
                    "gt": json.loads(r["gt"]), "decoys": json.loads(r["decoys"]), "text": r["text"]})
    return out


def gap(errs, other_errs):
    """Errors of `errs` not shared with `other_errs` (same id+record+slot, or a record-level error in that sample)."""
    by_id = defaultdict(list)
    for o in other_errs:
        by_id[o["id"]].append(o)
    out = []
    for e in errs:
        os_ = by_id.get(e["id"], [])
        if e["slot"] is None:
            shared = any(o["slot"] is None and o["record"] == e["record"] for o in os_)
        else:
            shared = any(o["slot"] is None or (o["slot"] == e["slot"] and o["record"] == e["record"]) for o in os_)
        if not shared:
            out.append(e)
    return out


def groups(items):
    g = defaultdict(list)
    for e in items:
        g[f"root:{e['root']}"].append(e)
        g[f"lang:{e['lang']}"].append(e)
        g[f"schema:{e['schema']}"].append(e)
        for h in e["hc"]:
            g[f"hc:{h}"].append(e)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--viewer_url", default="http://127.0.0.1:8111/")
    ap.add_argument("--out", default="outputs/audit/27b_vs_strong")
    ap.add_argument("--audit", default="outputs/audit/label_noise_audit.jsonl")
    a = ap.parse_args()
    res = {}
    for sname, cfg in SETS.items():
        samples = samples_of(cfg)
        audit = load_audit(a.audit, sname)
        sp = {x["id"]: x["raw"] for x in load(cfg["strong"])}
        wp = {x["id"]: x["raw"] for x in load(cfg["weak"])}
        sub = [s for s in samples if s["id"] in sp and s["id"] in wp]
        es, ew = analyze(sub, sp), analyze(sub, wp)
        res[sname] = {"n_samples": len(sub), "strong_errors": len(es), "llama_errors": len(ew)}
        for mname, label in MODELS.items():
            preds = {x["id"]: x["raw"] for x in load(cfg["m"](label))}
            errs = analyze([s for s in sub if s["id"] in preds], preds)
            both = gap(gap(errs, es), ew)
            items, dropped = [], 0
            for e in both:
                v = audit_verdict(e, label, audit) if audit else None
                if v and v["category_name"] in DROP:
                    dropped += 1
                    continue
                items.append(e)
            res[sname][mname] = {"total_errors": len(errs), "wrong_while_both_right": len(both), "excluded_by_audit": dropped,
                                 "items": items}
    jout = {}
    for s, d in res.items():
        jout[s] = {k: v for k, v in d.items() if not isinstance(v, dict)}
        for m in MODELS:
            g = groups(d[m]["items"])
            jout[s][m] = {"total_errors": d[m]["total_errors"], "wrong_while_both_right": d[m]["wrong_while_both_right"],
                          "excluded_by_audit": d[m]["excluded_by_audit"], "kept": len(d[m]["items"]),
                          "groups": {k: {"count": len(v), "ids": sorted({e["id"] for e in v})}
                                     for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}}
    json.dump(jout, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)

    cell = lambda e: f"`{e['id']}` {e['slot'] or '(record)'}: gold={e['gold']!r} pred={e['pred']!r}"  # noqa: E731
    md = ["# 27B vs strong models: errors where Qwen3.8-27B is wrong but Sonnet-5.5 AND Llama-3.3-70B are right\n",
          "27B-ft-stage-1 = `qwen3.8-27b-ft-int4`; 27B-INT4-base = `qwen3.8-27b-int4`. An item = one slot (or missed/extra record) error; "
          "it counts when neither strong model has an error on that id+record+slot (nor a record-level error in the sample). "
          "Audit verdicts gold_wrong / gold_incomplete / ambiguous / scorer_issue are excluded "
          "(`outputs/audit/label_noise_audit.jsonl`; only a subset of errors is audited, mostly grpo_dev and pilot). "
          "Generated by `eval/strong_gap.py`.\n", "## Summary\n",
          "| set | model | all slot errors | wrong, both strong right | audit-excluded | kept |", "|---|---|---|---|---|---|"]
    for s in res:
        for m in MODELS:
            j = jout[s][m]
            md.append(f"| {s} (n={res[s]['n_samples']}) | {m} | {j['total_errors']} | {j['wrong_while_both_right']} | "
                      f"{j['excluded_by_audit']} | **{j['kept']}** |")
    for s in res:
        for m in MODELS:
            items = res[s][m]["items"]
            md.append(f"\n## {s} — {m} ({len(items)} items)\n")
            md.append("| group | count | share | examples (gold → pred) |\n|---|---|---|---|")
            for k, v in sorted(groups(items).items(), key=lambda kv: -len(kv[1])):
                if len(v) < 2 and len(items) > 20:
                    continue
                ex = "<br>".join(cell(e).replace("|", "/") for e in v[:3])
                md.append(f"| {k} | {len(v)} | {len(v) / max(1, len(items)):.0%} | {ex} |")
    hcs, langs, schemas, roots = (defaultdict(int) for _ in range(4))
    for s in res:
        for e in res[s]["27B-ft-stage-1"]["items"]:
            for h in e["hc"]:
                hcs[h] += 1
            langs[e["lang"]] += 1
            schemas[e["schema"]] += 1
            roots[e["root"]] += 1
    tot = sum(langs.values()) or 1
    rk = lambda d, n=99: sorted(d.items(), key=lambda kv: -kv[1])[:n]  # noqa: E731
    md.append("\n## Training emphasis for stage-2 SFT (27B-ft-stage-1 gap items, pooled over the 3 sets)\n")
    md.append("Hard-case tags are multi-label (an item counts once per tag), so shares do not sum to 100%.\n")
    md.append("| rank | hard case | items | meaning |\n|---|---|---|---|")
    for i, (h, c) in enumerate(rk(hcs, 8), 1):
        md.append(f"| {i} | {h} | {c} | {HC_DESC.get(h, '')} |")
    md.append("\n**Root causes (ranked):** " + ", ".join(f"{r} {c}" for r, c in rk(roots)))
    md.append("\n**Languages to oversample:** " + ", ".join(f"{l} {c} ({c / tot:.0%})" for l, c in rk(langs)))
    md.append("\n**Schemas with most gap items:** " + ", ".join(f"{l} {c}" for l, c in rk(schemas, 6)))
    md.append("\n**New sample types to generate:** for each top root cause generate long, multi-entity docs with the phenomenon, "
              "planted distractors and null-tempting slots, keeping only teacher-verified rows. Top root causes: "
              + "; ".join(f"{r} ({ROOTS[r].split(';')[0]})" for r, _ in rk(roots, 4)) + ".\n")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md))
    print("wrote", a.out + ".md/.json")
    print({s: {m: jout[s][m]["kept"] for m in MODELS} for s in res})

    if a.report:
        esc = html.escape
        link = lambda i: f'<a href="{a.viewer_url}#{i}" target="_blank">{i}</a>'  # noqa: E731
        h = ("<h2 id='gap27'>27B fails where Sonnet/Llama succeed</h2><p class=muted>Slot/record errors of Qwen3.8-27B whose "
             "id+record+slot both Sonnet-5.5 and Llama-3.3-70B get right; audited gold_wrong/gold_incomplete/ambiguous/scorer_issue "
             "items removed. Full tables + training emphasis: <code>outputs/audit/27b_vs_strong.md</code>. Links open the viewer's "
             "selected dataset.</p><table><tr><th>set</th><th>model</th><th>all errors</th><th>both strong right</th>"
             "<th>audit-excluded</th><th>kept</th></tr>")
        for s in res:
            for m in MODELS:
                j = jout[s][m]
                h += (f"<tr><td>{s}</td><td>{m}</td><td>{j['total_errors']}</td><td>{j['wrong_while_both_right']}</td>"
                      f"<td>{j['excluded_by_audit']}</td><td><b>{j['kept']}</b></td></tr>")
        h += "</table>"
        for s in res:
            for m in MODELS:
                items = res[s][m]["items"]
                body = ""
                for k, v in sorted(groups(items).items(), key=lambda kv: -len(kv[1])):
                    rows = "".join(f"<tr><td>{link(e['id'])}</td><td>{e['slot'] or '(record)'}</td><td class=v>{esc(str(e['gold']))}</td>"
                                   f"<td class=v>{esc(str(e['pred']))}</td></tr>" for e in v[:3])
                    body += (f"<details><summary><b>{esc(k)}</b> ({len(v)})</summary><table><tr><th>id</th><th>slot</th>"
                             f"<th>gold</th><th>pred</th></tr>{rows}</table></details>")
                h += f"<details><summary><b>{s} &mdash; {m}</b> ({len(items)} items)</summary>{body}</details>"
        t = open(a.report, encoding="utf-8").read()
        t = re.sub(r"<!--sec:gap27-->.*?<!--/sec:gap27-->", "", t, flags=re.S)
        i = t.rfind("</body>")
        t = t[:i] + f"<!--sec:gap27-->{h}<!--/sec:gap27-->" + t[i:]
        open(a.report, "w", encoding="utf-8").write(t)


if __name__ == "__main__":
    main()
