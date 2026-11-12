# Test set: realistic long documents (H20)

**TEST ONLY — never add `data/test_long/` to any training/SFT/GRPO export.**

## Purpose
H19 long docs were LLM intro/closing + ONE deterministic block with all entities. Real documents spread entities over narrative sections. H20 builds such documents for a final held-out evaluation (he + fa).

## H20 design (`re_synth/realistic.py`, CLI `build --realistic_long --seed 3`)
- Genres: annual/quarterly report, tender committee minutes, investigation/audit report, municipal council protocol, news round-up, internal memo with appendices, court/regulatory decision. Formats: html ~50% / text ~25% / email (with appendix) ~25%.
- 10–20 entities, 4–7 sections (1–2 structured, rest prose; first is prose; email appendix last), 4.5k–16k chars plain text.
- **prose** sections: LLM writes 1–3 paragraphs mentioning 2–4 planned entities (`inject.build_prompt(section=...)`), values grounded by `inject.verify_and_ground`; hard cases H1/H2/H4/H11/H12 applied.
- **structured** sections: deterministic `layout.long_doc` renderers (table/list/org-list/cards, H13/H15/H18 etc., decoy columns).
- **cross-section entities** (~20%, ≥1 per doc): ID + some slots in prose, remaining slots in a later table row → ONE merged gold record. ~30% are "updates" (prose has old status, table the current one; old value is a decoy). Surname-only short form in the table only if the full name appeared earlier and the surname is unique in the doc.
- LLM writes title, headings, lead-ins, footnote (one frame call).
- Gold = plan records (merged). QC: gold in text, script ratio, length, no markers.
- `DETERMINISTIC_GOLD`: **H20 not added** (gold is mixed: structured parts deterministic, prose parts only grounded).
- Limitation: `chunking.chunk_gold` finds entities by full name, so a surname-only table chunk drops its record in the *oracle/training* chunk export (not used for H20). Model chunk outputs merge by ID (`merge_records`).

## Counts and separation
- Generator `copilot-claude-sonnet-5.5`; 40 he + 40 fa passing QC (ids `he-t3-NNNNN`); `samples_*.jsonl` also keep QC-failed rows (all `length`, docs slightly < 4.5k); `spare_*.jsonl` = extra passing docs not in the test set.
- QC pass rate over generated: he 46/66 (70%), fa 58/66 (88%). Avg ≈6k chars raw, ~16 entities/doc. Formats (104 ok docs): html 47 / text 32 / email 25.
- Seeds: 1420 training/pilot URLs excluded (`scripts/test_long_seeds.py`); 0 seed leaks.
- 8-gram overlap with the 1800 training samples: max 0.012 (he) / 0.021 (fa), threshold 0.3, none rejected (`scripts/test_long_check.py`, `separation.json`).
- Entity-name reuse vs training: 92% (he) / 96% (fa) of names appear in training docs (generic lexicon names); name+ID pair reuse 0%.

## Baselines (chunked, slot-F1; 40 docs per language)
| model | all | he | fa |
|---|---|---|---|
| Llama-3.3-70B | ~0.974 | 0.978 | 0.969 |
| Sonnet 5.5 | ~0.988 | 0.992 | 0.984 |

Sonnet exact-sample 0.53/0.55, Llama 0.20/0.18. Per-format/schema breakdown: `data/test_long/score_{weak,strong}_{he,fa}.json/.md`.
Sonnet is just under the 0.99 "too easy" bar overall (he 0.992 above it).

## Open items
- Per-hard-case (H20 sub-case) breakdown not computed.
- Sonnet ≈0.99 → consider harder constructs (footnote overrides, abbreviations, more cross-section updates) in a v2.
- Viewer: `viewer_sources.json` added; not checked in the UI.

## Commands
```
export PYTHONPATH=/home/dekelco/NLP/llm_providers RE_ENV_FILE=$PWD/.env
python3 scripts/test_long_seeds.py
python3 -m re_synth build --realistic_long --seed 3 ...   # see __main__.py
PYTHONPATH=. python3 scripts/test_long_check.py
python3 -m re_synth export-eval --samples data/test_long/samples_he.jsonl data/test_long/samples_fa.jsonl --out data/test_long/test_long.jsonl
python3 -m re_synth predict --samples data/test_long/samples_he.jsonl --model meta-llama/llama-3.3-70b-instruct --out data/test_long/preds_weak.jsonl
python3 -m re_synth score --samples data/test_long/samples_he.jsonl --preds data/test_long/preds_weak.jsonl --out data/test_long/score_weak_he.json
```
