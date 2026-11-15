"""Print 'step=i/N s_per_step=x eta=Ymin loss=l' from a Trainer progress metrics.jsonl."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
tr = [r for r in rows if "loss" in r]
if tr:
    n, r = int(sys.argv[2]), tr[-1]
    sp = r["time"] / max(r["step"], 1)
    print(f"step={r['step']}/{n} s_per_step={sp:.1f} eta={(n - r['step']) * sp / 60:.0f}min loss={r['loss']:.3f}")
