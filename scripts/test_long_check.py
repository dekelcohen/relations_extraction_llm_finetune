"""Separation checks for data/test_long (TEST ONLY): seed URLs vs all training/pilot seeds, 8-gram overlap vs all
training samples (>= 0.3 -> sample marked qc=fail 'ngram_overlap'), entity-name reuse vs grpo_train.
Writes data/test_long/separation.json. Run after `build --realistic_long`."""
import glob
import html
import json
import re
import sys
from pathlib import Path

N, THRESH = 8, 0.3
OUT = Path("data/test_long")
SEED_PAT = ["data/pilot/*seeds*", "data/synth*/seeds*.jsonl", "data/stage_2_RL_GRPO/seeds*.jsonl"]
TRAIN_PAT = ["data/synth_hefa*/samples*.jsonl", "data/stage_2_RL_GRPO/samples_*.jsonl"]
EXTRA_NAME_SRC = ["data/stage_2_RL_GRPO/grpo_train.jsonl"]


def words(text: str) -> list[str]:
    return re.findall(r"\w+", html.unescape(re.sub(r"<[^>]+>", " ", text)).lower())


def grams(text: str) -> set:
    w = words(text)
    return {tuple(w[i:i + N]) for i in range(len(w) - N + 1)}


def rows(pattern: str):
    for p in sorted(glob.glob(pattern)):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                yield json.loads(l)


def main() -> None:
    used = {r["url"] for pat in SEED_PAT for r in rows(pat) if "test_long" not in pat}
    from re_synth.schemas import SCHEMAS
    train_grams, train_names, train_keys = set(), set(), set()
    n_train = 0
    for pat in TRAIN_PAT:
        for r in rows(pat):
            n_train += 1
            train_grams |= grams(r["text"])
            recs = r.get("gt", {}).get("records", [])
            train_names |= {x["main_entity"] for x in recs if x.get("main_entity")}
            ids = SCHEMAS[r["schema_id"]].id_slot if r.get("schema_id") in SCHEMAS else None
            train_keys |= {(x["main_entity"], x.get(ids)) for x in recs if ids and x.get(ids)}
    for pat in EXTRA_NAME_SRC:
        for r in rows(pat):
            gt = json.loads(r["gt"]) if isinstance(r["gt"], str) else r["gt"]
            train_names |= {x["main_entity"] for x in gt.get("records", []) if x.get("main_entity")}
    rep = {"excluded_seed_urls": len(used), "train_samples": n_train, "train_8grams": len(train_grams), "files": {}}
    for lang in ("he", "fa"):
        path = OUT / f"samples_{lang}.jsonl"
        data = [json.loads(l) for l in path.open(encoding="utf-8")]
        url_hits = overlap_fail = 0
        ov, reuse_n, reuse_tot, key_n = [], 0, 0, 0
        for s in data:
            url_hits += s["seed"]["url"] in used
            g = grams(s.get("text", ""))
            o = len(g & train_grams) / max(1, len(g))
            ov.append(o)
            if o >= THRESH and s["qc"]["status"] == "ok":
                s["qc"] = {"status": "fail", "issues": sorted(set(s["qc"]["issues"]) | {"ngram_overlap"})}
                overlap_fail += 1
            if s["qc"]["status"] == "ok":
                names = [r["main_entity"] for r in s["gt"]["records"]]
                reuse_n += sum(n in train_names for n in names)
                ids = SCHEMAS[s["schema_id"]].id_slot
                key_n += sum((r["main_entity"], r.get(ids)) in train_keys for r in s["gt"]["records"])
                reuse_tot += len(names)
        path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in data), encoding="utf-8")
        rep["files"][lang] = {"samples": len(data), "ok": sum(s["qc"]["status"] == "ok" for s in data),
                              "seed_url_in_training": url_hits, "ngram_overlap_rejected": overlap_fail,
                              "max_ngram_overlap": round(max(ov), 4), "mean_ngram_overlap": round(sum(ov) / len(ov), 4),
                              "entity_name_reuse_rate": round(reuse_n / max(1, reuse_tot), 3),
                              "name_and_id_reuse_rate": round(key_n / max(1, reuse_tot), 4),
                              "entities_checked": reuse_tot}
    (OUT / "separation.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if any(f["seed_url_in_training"] for f in rep["files"].values()):
        sys.exit("seed URL leakage!")


if __name__ == "__main__":
    main()
