"""Illustrative multi-stage extraction-prompt chain, rendered for human review alongside the
dataset (viewer "Prompt chain" panel).

This is a *review/design* artifact, not wired into training/eval/mining (those use the
single-stage schema+guidelines+fewshot prompt in `prompting.py`). It renders:

1. A **Background** prompt: a short domain-specific backstory paragraph, plus the main entity and
   each relation/slot with its description and a few generated example surface forms (one per
   language) — so a prompt reviewer can see what the model is expected to recognize before reading
   the extraction rules.
2. A **chain of Extraction prompts**. Schemas that have a natural identifier slot (`Schema.id_slot`)
   get a 2-stage chain mirroring a "kb_dict first, then enrich" pipeline:
   - Stage 1 ("id_lookup"): extract only `main_entity` + the id slot (a cheap, high-precision
     lookup, e.g. a keyword/regex/dictionary matcher in a real system).
   - Stage 2 ("full_extract"): given stage 1's output, extract the remaining relations for those
     *same* entities only (added instruction: "do not invent new ones").
   Schemas without an id slot get a single extraction stage with all relations.
   Each stage's prompt only asks for the JSON fields it adds, which keeps later stages focused and
   cheap when a document has to be scanned for only a handful of new attributes.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from . import prompting
from .schemas import Schema, Slot, gen_main, gen_value

BACKSTORY = {
    "vehicle": "Internal fleet-management notes about vehicles the company owns or leases: where each one "
               "currently is, who it is assigned to, and its handling status.",
    "shipment": "Logistics / freight-forwarding correspondence about shipments moving from an origin to a "
                "destination, tracked by a carrier.",
    "person_employment": "HR and personnel records, or internal messages, about employees and their current job.",
    "asset": "IT/equipment asset-tracking notes: who holds a piece of equipment, where it is, and its condition.",
    "meeting": "Scheduling notices and meeting-minutes summaries for internal or external events.",
    "contract": "Procurement / legal correspondence about contracts between a supplier and a client.",
    "account": "Banking-operations notes about customer accounts and recent account actions.",
    "vessel": "Port and maritime-logistics operations notes about vessels, their voyages and operators.",
}

EXAMPLE_LANGS = ["en", "he", "fa"]


def _examples(schema: Schema, slot: Slot | None, langs: list[str] = EXAMPLE_LANGS) -> list[str]:
    """A couple of generated example surface forms (one per language) for the background prompt."""
    out = []
    for lang in langs:
        seed = hash((schema.id, slot.name if slot else "__main__", lang)) & 0xFFFFFFFF
        rng = random.Random(seed)
        try:
            val = gen_main(schema, rng, lang) if slot is None else gen_value(schema, slot, rng, lang)["value"]
        except Exception:
            continue
        if val:
            out.append(val)
    return out


def background_prompt(schema: Schema) -> str:
    story = BACKSTORY.get(schema.id, schema.main_definition)
    lines = [
        story,
        "",
        f"Main entity \u2014 {schema.main_type}: {schema.main_definition}",
        f"  Example surface forms: {', '.join(_examples(schema, None)) or '(n/a)'}",
        "",
        "Relations to extract for each main entity:",
    ]
    for slot in schema.slots:
        lines.append(f"- {slot.name} ({slot.type}): {slot.definition}")
        exs = _examples(schema, slot)
        if exs:
            lines.append(f"  Example surface forms: {', '.join(exs)}")
    return "\n".join(lines)


@dataclass
class Stage:
    name: str
    title: str
    slot_names: list[str]
    uses_prev: bool


def build_chain(schema: Schema) -> list[Stage]:
    if schema.id_slot:
        rest = [s.name for s in schema.slots if s.name != schema.id_slot]
        return [
            Stage("id_lookup", "Stage 1 \u2014 kb_dict: identify entities by ID", [schema.id_slot], False),
            Stage("full_extract", "Stage 2 \u2014 extract remaining relations", rest, True),
        ]
    return [Stage("full_extract", "Extraction", [s.name for s in schema.slots], False)]


def _schema_subset(schema: Schema, slot_names: list[str]) -> dict:
    slots = [s for s in schema.slots if s.name in slot_names]
    return {
        "main_entity": {"type": schema.main_type, "definition": schema.main_definition},
        "slots": [{"name": s.name, "type": s.type, "definition": s.definition} for s in slots],
    }


def _filter_record(rec: dict | None, slot_names: list[str]) -> dict | None:
    if not rec:
        return None
    out = {"main_entity": rec.get("main_entity")}
    for k in slot_names:
        out[k] = rec.get(k)
    return out


def render_stage(schema: Schema, stage: Stage, fewshot_record: dict | None, fewshot_text: str | None,
                  prev_output: list[dict] | None) -> str:
    schema_dict = _schema_subset(schema, stage.slot_names)
    parts = ["## Schema", json.dumps(schema_dict, ensure_ascii=False, indent=1)]
    if stage.uses_prev:
        parts += [
            "## Previously identified entities (output of the previous stage)",
            "Only extract the relations below for these exact entities; do not invent new ones and do "
            "not change the entity identifiers:",
            json.dumps(prev_output or [], ensure_ascii=False, indent=1),
        ]
    parts += ["## Guidelines", "\n".join(f"- {g}" for g in prompting.GUIDELINES)]
    keys = ["main_entity"] + stage.slot_names
    out_shape = {"records": [{k: "..." for k in keys}]}
    parts += ["## Output format", f"Return a JSON object: {json.dumps(out_shape, ensure_ascii=False)} "
                                  "(use null for missing slots; {\"records\": []} if no such entity)."]
    if fewshot_text is not None and fewshot_record is not None:
        parts += ["## Example", "Document:\n<<<\n" + fewshot_text + "\n>>>",
                  "Output:\n" + json.dumps({"records": [fewshot_record]}, ensure_ascii=False)]
    parts += ["## Document", "<<<\n{document}\n>>>", "Output:"]
    return "\n\n".join(parts)


def build_prompt_chain(schema: Schema, example_sample: dict | None = None) -> dict:
    """example_sample: a real sample dict (from the loaded dataset, with 'text' and 'records') of
    the same schema, used to ground the fewshot / previous-stage-output examples in real data."""
    stages = build_chain(schema)
    ex_text = (example_sample or {}).get("text")
    ex_records = (example_sample or {}).get("records") or []
    ex_record = ex_records[0] if ex_records else None

    rendered = []
    prev_output = None
    for stage in stages:
        fewshot_record = _filter_record(ex_record, stage.slot_names)
        rendered.append({
            "name": stage.name,
            "title": stage.title,
            "added_fields": stage.slot_names,
            "uses_prev_stage": stage.uses_prev,
            "prev_stage_output_example": prev_output,
            "fewshot_doc": ex_text if fewshot_record else None,
            "fewshot_output": {"records": [fewshot_record]} if fewshot_record else None,
            "prompt_preview": render_stage(schema, stage, fewshot_record, ex_text, prev_output),
        })
        prev_output = [fewshot_record] if fewshot_record else None
    return {"schema_id": schema.id, "background": background_prompt(schema), "stages": rendered}
