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

## Self-distilled variant `train_selfdistill.jsonl` (used for the stage-2 run; replaces the short Sonnet-CoT rows)
Why: stage-1 ft thinks ~1.5k chars/chunk vs 4.3–4.9k for base-INT4 at `reasoning_effort=medium` and loses to base-medium on grpo_dev/test_long;
`train.jsonl` repeats the same short Sonnet CoT (mean 651 chars). So: **rejection-sampled self-distillation** (`scripts/selfdistill_prep.py` →
`selfdistill_src.jsonl`, 680 unique docs of `train.jsonl`, eval-format, prompt == the train row's prompt; `scripts/build_selfdistill.py`).
1. Base `RedHatAI/Qwen3.8-27B-INT4`, reasoning_effort=medium, T=0.6, same engine/eval chunking (902 chunk prompts, 0 max_tokens hits), 1 sample/doc.
2. Keep docs with merged slot-F1 == 1.0, valid JSON, every chunk closing its think: 581/680. Failed 99 docs re-sampled once (seed 1): +38 → 619/680.
3. Still failing: Sonnet row only if its CoT ≥ 1500 chars and the doc is not chunked (4 rows), else dropped (57 docs; mostly INSS and fa). Rows with a
   trace > 12000 chars are dropped (not truncated): see counts below.
4. One row per CHUNK prompt (eval chunking, ≤4000 chars unchunked), completion = `trace\n</think>\n\nJSON` (the chat template already prefills
   `<think>\n`, so no leading `<think>`); trained with the same `reasoning_effort=medium` template instruction. 2× oversampling follows `#dup` ids of `train.jsonl`.
   994 rows (stage-1 `train.jsonl` rows had a doubled `<think>`; not repeated here).

Kept rate / think length (chars) per source and lang (`selfdistill_stats.md`):

| source/lang | docs | self-distill kept | Sonnet fallback | dropped | kept rate | think chars p50/p90/mean |
|---|---|---|---|---|---|---|
| inss/fa | 27 | 16 | 1 | 10 | 59% | 2980/4450/3004 |
| inss/he | 34 | 20 | 1 | 13 | 59% | 2536/4994/2818 |
| replay_synth_hefa/fa | 76 | 72 | 0 | 4 | 95% | 1499/2528/1790 |
| replay_synth_hefa/he | 62 | 59 | 0 | 3 | 95% | 1786/3093/1963 |
| stage2_synth/fa | 239 | 215 | 1 | 23 | 90% | 3086/6892/3710 |
| stage2_synth/he | 242 | 231 | 1 | 10 | 95% | 3087/7165/3716 |

All rows: think mean 3417 / p50 2755 / p90 6713 chars (replay docs are simple, so shorter) vs base-medium 4279 (grpo_dev) / 4882 (test_long) per chunk
and ft-stage-1-medium 1458 (grpo_dev); the long/hard stage-2 + INSS rows (≈3.7k) are close to base-medium, no row is shorter than the base trace it comes from.
