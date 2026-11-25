"""Quality gate for a small eval smoke: python smoke_gate.py <preds.jsonl.metrics.json> [--min-json F] [--min-closed F] [--min-f1 F].
Prints one stats line (json validity, think closure, length hits, mean tokens); exits 1 if a threshold is missed."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("metrics")
ap.add_argument("--min-json", type=float, default=0)
ap.add_argument("--min-closed", type=float, default=0)
ap.add_argument("--min-f1", type=float, default=0)
a = ap.parse_args()
m = json.load(open(a.metrics, encoding="utf-8"))
rep, gen = m["report"]["all"], m.get("generation", {})
n = max(1, gen.get("prompts", rep.get("n", 1)))
stats = {"n": rep.get("n"), "json_valid": rep["json_valid"], "slot_f1": rep["slot_f1"],
         "closed_think": gen.get("closed_think", 0) / n, "hit_limit": gen.get("hit_limit", 0),
         "mean_tokens": round(gen.get("tokens", 0) / n)}
print("SMOKE_STATS", json.dumps(stats))
fails = [k for k, v, t in (("json_valid", stats["json_valid"], a.min_json), ("closed_think", stats["closed_think"], a.min_closed),
                           ("slot_f1", stats["slot_f1"], a.min_f1)) if v + 1e-9 < t]
if fails:
    raise SystemExit(f"SMOKE_GATE FAIL: {', '.join(fails)}")
print("SMOKE_GATE PASS")
