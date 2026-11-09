"""Extraction prompt (shared by data export, baselines/mining, training and eval)."""
from __future__ import annotations

import json

SYSTEM = ("You are an information extraction engine. Extract records about the requested main entity type from the "
          "document, following the schema and guidelines exactly. Output only JSON.")

GUIDELINES = [
    "Create one record per distinct main entity of the requested type mentioned in the document.",
    "Copy values verbatim from the document (same digits, same calendar, same spelling). Do not translate, convert "
    "or normalize. Do not include grammatical prefixes attached to a word (e.g. Hebrew ב/ל/מ/ה/ו/ש/כ).",
    "Use null when the document does not state the slot for THAT entity. Never copy a value that belongs to another "
    "entity, and never guess.",
    "Respect each slot definition: ignore distractor values of the same type (e.g. manufacture dates, dates of earlier "
    "messages, previous locations, manufacturers, phone numbers).",
    "A heading or subtitle (e.g. 'The following X belong to ORG:') applies to the list/table items below it.",
    "If a status or value changed over time, report the latest one.",
    "Email headers (subject, etc.) are part of the document.",
]


def render_user(schema_dict: dict, text: str, fewshot: list[dict] | None = None, guidelines: list[str] | None = None
                ) -> str:
    g = guidelines if guidelines is not None else GUIDELINES
    keys = ["main_entity"] + [s["name"] for s in schema_dict["slots"]]
    out = {"records": [{k: "..." for k in keys}]}
    parts = ["## Schema", json.dumps(schema_dict, ensure_ascii=False, indent=1), "## Guidelines",
             "\n".join(f"- {x}" for x in g),
             "## Output format", f"Return a JSON object: {json.dumps(out, ensure_ascii=False)} "
                                 "(use null for missing slots; {\"records\": []} if no such entity)."]
    for i, ex in enumerate(fewshot or []):
        parts += [f"## Example {i + 1}", "Document:\n<<<\n" + ex["text"] + "\n>>>",
                  "Output:\n" + json.dumps(ex["output"], ensure_ascii=False)]
    parts += ["## Document", "<<<\n" + text + "\n>>>", "Output:"]
    return "\n\n".join(parts)


def messages(schema_dict: dict, text: str, fewshot=None, guidelines=None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_user(schema_dict, text, fewshot, guidelines)}]


def target_json(gt: dict, rename: dict | None = None, slot_order: list[str] | None = None) -> str:
    rename = rename or {}
    recs = []
    for r in gt["records"]:
        o = {"main_entity": r.get("main_entity")}
        for k in slot_order or [k for k in r if k != "main_entity"]:
            o[rename.get(k, k)] = r.get(k)
        recs.append(o)
    return json.dumps({"records": recs}, ensure_ascii=False)
