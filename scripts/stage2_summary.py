"""F1 / json_valid / hit% (+avg gen tokens) per model and set from outputs/eval metrics and the pipeline logs -> markdown."""
import json, os, re, sys
E = "outputs/eval"
SETS = {"pilot": "{m}-think.preds.jsonl", "grpo_dev": "{m}.grpo_dev.preds.jsonl", "test_long": "{m}.test_long.preds.jsonl", "hard_inss": "{m}.hard_inss.preds.jsonl"}
MODELS = ["qwen3.8-27b-int4", "qwen3.8-27b-int4-medium", "qwen3.8-27b-ft-int4", "qwen3.8-27b-ft-int4-medium", "qwen3.8-27b-ft-stage2-int4", "qwen3.8-27b-ft-stage2-int4-medium"]
logs = "".join(open(f, errors="ignore").read() for f in ("outputs/pipeline/a100/pipeline.log",) if os.path.exists(f))
print("| model | " + " | ".join(SETS) + " |\n|---|" + "---|" * len(SETS))
for m in MODELS:
    cells = []
    for s, pat in SETS.items():
        f = f"{E}/{pat.format(m=m)}.metrics.json"
        if not os.path.exists(f):
            cells.append("–"); continue
        a = json.load(open(f))["report"]["all"]
        hit = re.findall(re.escape(pat.format(m=m)) + r" DONE .*?hit_max_tokens=(\d+)/(\d+) avg_gen_tok=(\d+)", logs)
        h = f"{100 * int(hit[-1][0]) / int(hit[-1][1]):.1f}% hit, {hit[-1][2]} tok" if hit else "hit n/a"
        cells.append(f"{a['slot_f1']:.4f} / jv {a['json_valid']:.3f} / {h}")
    print(f"| {m} | " + " | ".join(cells) + " |")
