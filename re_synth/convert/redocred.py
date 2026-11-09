"""Re-DocRED / DocRED (-CF) -> our record format.

Input: Re-DocRED json list of docs {title, sents, vertexSet, labels:[{h,t,r,evidence}]}; rel_info.json {P17: "country", ...}.
Per doc: main type = entity type with most head relations; records = head entities of that type with relations;
slots = relation names seen for that type in the doc + up to 2 absent relations (-> null). Multi-tail -> list.
Get data: https://github.com/tonytan48/Re-DocRED (data/*.json), rel_info.json from DocRED.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TYPE_MAP = {"TIME": "date", "LOC": "location", "ORG": "org", "PER": "person", "NUM": "text", "MISC": "text"}
MAIN_DEF = {"PER": "A person", "ORG": "An organization", "LOC": "A location / geopolitical entity",
            "MISC": "A named work, event, product or other named entity"}


def _detok(sents: list[list[str]]) -> str:
    s = " ".join(" ".join(x) for x in sents)
    s = re.sub(r" ([.,;:!?%)\]'])", r"\1", s)
    return re.sub(r"([(\[$]) ", r"\1", s).replace(" 's", "'s").replace(" n't", "n't")


def convert(inp: Path, rel_info: Path, out: Path, limit: int | None = None, seed: int = 0) -> int:
    rng = random.Random(seed)
    rels = json.loads(Path(rel_info).read_text(encoding="utf-8"))
    docs = json.loads(Path(inp).read_text(encoding="utf-8"))[:limit]
    by_type_rels: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        for lab in d.get("labels", []):
            by_type_rels[d["vertexSet"][lab["h"]][0]["type"]][lab["r"]] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for di, d in enumerate(docs):
            vs, labels = d["vertexSet"], d.get("labels", [])
            if not labels:
                continue
            head_types = Counter(vs[l["h"]][0]["type"] for l in labels)
            mtype = head_types.most_common(1)[0][0]
            doc_rels = sorted({l["r"] for l in labels if vs[l["h"]][0]["type"] == mtype})
            extra = [r for r, _ in by_type_rels[mtype].most_common(15) if r not in doc_rels]
            slots_r = doc_rels + rng.sample(extra, k=min(2, len(extra)))
            tail_type = {}
            recs: dict[int, dict] = {}
            for l in labels:
                if vs[l["h"]][0]["type"] != mtype:
                    continue
                rec = recs.setdefault(l["h"], {"main_entity": vs[l["h"]][0]["name"]})
                name = rels.get(l["r"], l["r"]).replace(" ", "_")
                tail_type.setdefault(name, TYPE_MAP.get(vs[l["t"]][0]["type"], "text"))
                rec.setdefault(name, []).append(vs[l["t"]][0]["name"])
            slot_names = [rels.get(r, r).replace(" ", "_") for r in slots_r]
            records = []
            for rec in recs.values():
                o = {"main_entity": rec["main_entity"]}
                for sn in slot_names:
                    v = sorted(set(rec.get(sn, [])))
                    o[sn] = None if not v else v[0] if len(v) == 1 else v
                records.append(o)
            schema = {"main_type": mtype, "main_definition": MAIN_DEF.get(mtype, mtype) +
                      " that is the subject (head) of at least one of the listed relations.",
                      "slots": [{"name": sn, "type": tail_type.get(sn, "text"),
                                 "definition": f"Wikidata relation '{rels.get(r, r)}' ({r}); list if several, "
                                               "null if not stated."} for sn, r in zip(slot_names, slots_r)]}
            f.write(json.dumps({"id": f"redocred-{di}", "lang": "en", "format": "text",
                                "schema_id": f"redocred-{di}", "schema": schema, "hard_cases": ["public"],
                                "seed": {"source": "redocred", "title": d.get("title")}, "text": _detok(d["sents"]),
                                "gt": {"records": records}, "decoys": [], "qc": {"status": "ok", "issues": []}},
                               ensure_ascii=False) + "\n")
            n += 1
    return n
