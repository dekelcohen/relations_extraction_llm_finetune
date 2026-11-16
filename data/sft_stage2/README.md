# sft_stage2 — SFT data for Qwen3.8-27B stage 2 (thinking-mode CoT + gold JSON)

`train.jsonl`: 860 rows, same `{id, messages:[system,user,assistant], meta}` format as `data/synth_hefa/sft_cot_train.jsonl`
(assistant = `<think>teacher reasoning</think>` + gold JSON), shuffled with seed 0. Built by `scripts/build_sft_stage2.py`
(`build_info.json`); lengths/CoT stats by `scripts/sft_stage2_stats.py` → `length_stats.md`.

## Composition
| source | lang | rows |
|---|---|---|
| stage2_synth (`stage_2_RL_GRPO/cot_{he,fa}`, 535 → 481 after exclusion; 164 duplicated, see below) | he / fa | 300 / 345 |
| inss (teacher-CoT of INSS train samples; 10 + 6 duplicated) | he / fa | 44 / 33 |
| replay (random 25% of `synth_hefa/sft_cot_train.jsonl`, seed 0 = 138) | he / fa | 62 / 76 |

**Exclusions:** 54 stage-2 rows dropped because their parent id (id minus `#chunk`) is in `grpo_dev`, `test_long` or pilot; 0 further
near-duplicates (8-gram overlap ≥ 0.5 vs dev/test_long/pilot/hard_inss-dev texts); the replay rows have no id/text overlap with those sets.
`hard_inss/dev` is never used for training.

## INSS part (`data/hard_inss/`)
Pool 186 verified samples (he 97 / fa 89 ≈ 48% fa): 92 round-1 short (he 66 / fa 26), +32 whole-sample he→fa translations of them
(text + gold + decoys translated together, values re-verified verbatim + blind check), +31 NEW long he docs (2–3 consecutive chunks,
4–9.6k chars, 6–12 entities; far headers, table headers, nulls, status changes) and their 28 fa translations. Split: train 147 (78/69), dev 39 (19/20).
Teacher CoT (Sonnet 5.5, blind, kept only if sample F1 ≥ TIER_F1 = 0.99): **61 kept / 85 mismatch / 1 near-dup** of 147 (long docs: 24 of 47 kept). Mismatches
are mostly genuinely hard/ambiguous samples; dropping them biases INSS toward cases the teacher solves (and the dev set remains the harder, unfiltered measure).
For docs > 4000 chars the CoT budget was raised to `len/12` words (≤900).

## Emphasis weighting (from `outputs/audit/27b_vs_strong.md`)
Audit ranking of 27B-ft gaps: H9 (null) > H20 (long multi-entity) > H12 (status change) > H2 (far value) > H1 > H13 > H11 > H18; root causes
multi_entity_bleeding, long_range_or_coref; languages fa 58% / he 40%. Applied:
- **Generation:** the new INSS long docs always carry far_header + null_tempting + a table (pipe table with top-only header) + one of status_change/near_miss/coref; half translated to fa.
- **Oversampling (≤2×):** stage-2 rows with rank-weight score (Σ over tags, weights 8..1 for the 8 types above) ≥ 12, or fa rows with score ≥ 11, are duplicated once (`#dup` ids): 58 he + 106 fa.
  INSS rows with far_header/status_change + another tag duplicated (10 he + 6 fa). Rows whose text overlaps eval texts ≥ 0.3 and replay rows are never duplicated.

## Lengths (Qwen3.8-27B `tokenizer.json`, downloaded to `~/.cache/huggingface/qwen3.8-27b-tok/`)
Max total sequence **8692 tokens** (p95 5707, mean 2741); CoT mean 424 / max 1861. INSS: max 5296 tokens. See `length_stats.md`.
CoT tokens per record fall from 162 (1–3 records) to 42 (11+ records). **76 rows are flagged** (≥4 records and <60 CoT tokens/record), mostly
the 20–30-record H19 table docs (e.g. `he-2-00194`: 20 records, 33 CoT tokens). List in `length_stats.md`; consider dropping/regenerating them (and their ~11 `#dup`s) if the model learns terse reasoning on long docs.
