# hard_inss — hard extraction samples hosted on real INSS Hebrew prose (he + fa)

Built by `scripts/build_hard_inss.py` (scraper: `scripts/scrape_inss.py`). Rows have the same format as
`data/stage_2_RL_GRPO/grpo_dev.jsonl` (`id, prompt, gt, decoys, text, slot_types, meta`, via `re_synth.export.export_eval`),
so existing eval/scoring (`python -m re_synth score`, `eval/`) works. Gold-as-prediction scores slot-F1 1.0 on all rows.

**Stats:** 92 verified samples → `train.jsonl` 74 / `dev.jsonl` 18 (dev stratified by language).
he 66 / fa 26 (fa = LLM translation of Hebrew chunks, Hebrew→Persian mapping in `meta.jsonl: src_chunk`).
Schemas: meeting 31, contract 31, person_employment 30. 30 distinct INSS articles.

**Difficulty types** (3 sampled per document; he / fa): null_tempting 38/20, table_prose 32/10, coref 29/8, hebcal (Hebrew/Jalali + Gregorian
mixed calendars, geresh/gershayim) 27/14, negation/hypothetical 26/9, interleaved same-type entities 23/7, near_miss distractors 23/10.

**Pipeline:** 30 INSS articles (list/feed → article page, 1.5 s delay, browser UA) → sentence-packed chunks ≤ 2500 chars (`CHUNK_CHARS`) →
(≈40 chunks → Persian) → LLM rewrite keeping ≥ 45% of the host text and weaving in 2–3 records + decoys + gold
(`copilot-claude-sonnet-5.5`) → verifier: programmatic (gold/decoys verbatim in text, keys, ≥1 null, language, length, host overlap,
decoy ≠ gold) + cross-model blind extraction by `copilot-gpt-6.1-terra` scored with `re_synth.score` (keep slot-F1 ≥ 0.7;
mean 0.93, 35/92 perfect — disagreements are mostly the intended hard spots). Invented facts are fictional; they are not real events.

**Generator comparison (10 chunks each, same chunks/tags, cross-judged):** Sonnet 5.5: 7/10 programmatic pass, 6/10 final, blind F1 mean ≈0.89 on
passing (harder, richer coref/negation, 1 "fictional" artifact). GPT Terra: 7/10 pass, 7/10 final, blind F1 mean ≈0.95 (easier;
2/10 texts contain the literal word "فرضی"/fictional in entity names). Manual audit of 5 each: Sonnet 1 dubious gold (venue vs organizer ambiguity),
Terra 2 flaws (fictional-word leak, one trivial sample). → Sonnet used for the bulk. The Terra batch is kept in `*_terra.jsonl` only.

**Files:** `train.jsonl`, `dev.jsonl`, `meta.jsonl` (source URL, src chunk, tags, feature notes, blind F1, generator), `cand_*`/`verified_*`
(raw candidates + QC). Source articles (copyrighted, ~270 KB) are NOT in git: `data/raw/inss/` is git-ignored and pushed to private
HF dataset `Shani123/re-inss-raw`. Samples embed ≤2.5k-char chunks of the host prose.
Source URLs: see `url` in `meta.jsonl` (inss.org.il/he/publication/<slug>/).
