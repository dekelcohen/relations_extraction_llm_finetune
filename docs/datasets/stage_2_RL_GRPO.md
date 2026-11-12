# Dataset `stage_2_RL_GRPO` — hard-case prompts for GRPO (stage 2)

Stage 1 was SFT: Qwen3-4B QLoRA trained on Sonnet teacher reasoning plus the gold JSON
(`data/synth_hefa`, adapter `outputs/sft-cot-qwen3-4b`; pilot slot-F1 0.958).
Stage 2 continues that adapter with **GRPO / RLVR**: the reward is the deterministic scorer
(`re_synth/score.py`). There is no teacher text in the loss.
This dataset supplies the stage-2 prompts. It is weighted toward:

1. the fine-tuned model's pilot error root causes (`eval/error_analysis.py`):
   distractor/role, long-range/coref, list-header propagation, multi-entity bleeding;
2. **new large-table hard cases (H13–H18)**, built so that even Sonnet / Llama-3.3-70B fail.
   The table is rendered deterministically from the plan, so the gold stays reliable
   even when frontier models get it wrong;
3. **long documents (H19)** with 10–30 distinct main entities and their slots, spread over
   several sections (tables, lists, org-headed lists, cards).

Languages: **he and fa only** (no en in stage 2). Location: `data/stage_2_RL_GRPO/`
(**committed to git**, so work can continue on another machine).

**Status (2026-11-12):** 600 samples built (300 he + 300 fa, 600/600 passed QC). This includes
93 H19 long documents and 343 H13 tables. Difficulty probe, teacher reasoning, export and
GRPO are still to do. See [status](../status.md).

## Hard-case mix (`build --focus`)

| Family | Cases | Share (600 planned) | Why |
|---|---|---|---|
| Large tables (deterministic block) | H13 + 1–3 of H14/H15/H16, or H17; H18 for text/email | ~50% (html 75%, text/email 40%) | header far from data; same-type distractor columns |
| Error-focused text | H1, H2, H4, H11, H12 (weights 2/3/4/3/4), H9 | rest | FT root causes: distractor/role, long-range, bleeding, status change |
| Email | H3 (p=.7), H5 (p=.85) | email ≈ 30% of formats | ID only in subject; quoted-thread dates |
| Small list/table | H6 (p=.5 non-email), H7 (p=.85 html), H7c | non-table samples | list-header / caption org propagation |
| Long documents (deterministic block) | H19 (+ H14–H16 inside table sections) | `--long_p` (default 0.2; he build ran with 0.35, fa with 0.2) | 10–30 entities, context length, attention across sections |

Formats: html 40%, text 30%, email 30% (`FOCUS_FORMAT_W`).

### New table hard cases (`re_synth/hardcases.py`, `re_synth/layout.py::big_table`)

| Code | Description | Failure it targets |
|---|---|---|
| H13 | Large table: 6–10 rows (transposed: 3–5 entities) × 9–14 columns. Header only at the top. 1–3 same-type distractor columns (manufacturer / insurer / old number / "Previous X" / last-inspection date …) and 2–3 noise columns (phone, qty, notes, internal code, priority). Columns shuffled. Null cells shown as `""`, `—` or `-`. | header↔value distance, column confusion |
| H14 | Section rows (`<th colspan=N>Owner: X</th>` / `■ Owner: X`). The org applies to every following row until the next section; there is no org column. | list-header propagation |
| H15 | Two-level header: **identical** leaf labels under `Current data` vs `Previous data` (colspan) | needs the top-level header, not only the leaf |
| H16 | Ditto marks (`〃`, `כנ״ל`, `همان`, `"`) in a location/org column. Gold = the value from the row above. | value not literally in the row |
| H17 | Transposed table: entities are columns, attributes (including distractor attributes) are rows | orientation |
| H18 | Plain-text pipe table inside text/email. In two-level mode the group label appears only over the first column of its group. | no HTML structure |
| H19 | Long document: 10–30 distinct main entities in 2–5 sections (`table` = a big_table section, `list`, `list_org` = org only in the section heading, `cards`), with per-section distractor fields. Names are unique (`lexicon.*_main` generators: 27–180 distinct names per schema per language). | context length, many entities, section-scoped context |

All table distractor cells become `decoys` (placement `table`). The scorer counts a decoy
hit only when a prediction is wrong. A table row may repeat an item name at most twice; the
ID column tells the rows apart, since the scorer aligns records over all slots.

## Long documents: overlapping chunks (`re_synth/chunking.py`)

Documents longer than 4,000 chars (≈2.4k Qwen3 tokens in fa) are split before extraction:
- Chunks are ~2,500 chars with 500 chars of overlap.
- Cuts fall only between units: paragraph, list item, card, or table row.
- Each chunk repeats the context its units depend on: the title or email headers, the active
  section heading, and, inside a table, the header rows plus the current group row.
- A ditto row always keeps the row it refers to.
- Transposed tables (H17) are never split.

The per-chunk outputs are then merged and de-duplicated. Records from different chunks are
merged when either:
- they have the same ID value (even if one chunk mangled the name), or
- they have the same name (modulo Hebrew prefixes and punctuation), no ID conflict, and
  their slots mostly agree.

Records from the same chunk are never merged. Each slot takes the majority non-null value.

| Where | Behaviour |
|---|---|
| `re_synth predict`, `eval/predict.py` | On by default for docs over `--chunk_min` (4000). `--no_chunk` disables it. Output rows of chunked docs also store the per-chunk raw outputs in `chunks`. |
| `export --chunk` / `export-eval --chunk` | Training rows: each long doc becomes its **clean** chunks. Gold and decoys are restricted to the chunk (`chunk_gold`). Chunks where any record is only partly present are dropped (95% are kept). Dev/test stay doc-level and are evaluated with chunked inference. |
| Scoring | Always doc-level, against the full gold. |

Checks:
- Oracle round trip (split → per-chunk gold → merge) over all 107 chunked stage-2 docs: F1 = 1.000.
- Real Sonnet outputs on two 24–29-entity docs: F1 1.000, no duplicate records. Before ID-aware
  merging there were 3 duplicates, caused by a chunk where Sonnet wrote the type name
  `bank_account` instead of each account's name.
- Unit tests: `test_chunking_roundtrip_long_doc`, `test_chunk_merge_rules`, `test_chunk_overlap_duplicates_removed`.

The pilot test set is unaffected: its longest document is 2,748 chars.

## Pipeline

```bash
# 1. fresh seeds, excluding every URL used by the pilot and stage-1 (preload-then-strip)
#    -> data/stage_2_RL_GRPO/seeds_{he,fa}.jsonl (320 per language)
# 2. generate (Copilot Sonnet 5.5, reasoning_effort=low, as in stage 1)
#    (if Sonnet gives QC/format trouble: copilot-gpt-6.1-terra, -sol, or copilot-claude-opus-5.5 effort medium)
PYTHONPATH=../llm_providers python -m re_synth build --seeds data/stage_2_RL_GRPO/seeds_he.jsonl \
  --out data/stage_2_RL_GRPO/samples_he.jsonl --langs he --n 300 --seed 2 --focus --long_p 0.35 \
  --model copilot-claude-sonnet-5.5 --workers 6          # fa: seeds_fa / samples_fa.jsonl, default --long_p
# 3. difficulty probe: frontier + 70B (Claude only via copilot-*; Llama via OpenRouter)
#    long docs are chunked by default; OpenRouter key: RE_ENV_FILE=$PWD/.env (git-ignored)
python -m re_synth predict --samples data/stage_2_RL_GRPO/samples_he.jsonl --model copilot-claude-sonnet-5.5 \
  --out data/stage_2_RL_GRPO/preds_strong.jsonl
RE_ENV_FILE=$PWD/.env python -m re_synth predict --samples data/stage_2_RL_GRPO/samples_he.jsonl \
  --model meta-llama/llama-3.3-70b-instruct --out data/stage_2_RL_GRPO/preds_weak.jsonl
python -m re_synth mine --samples data/stage_2_RL_GRPO/samples.jsonl --weak .../preds_weak.jsonl \
  --strong .../preds_strong.jsonl --out data/stage_2_RL_GRPO/tiers.json
# 4. teacher reasoning (kept for INSPECTION ONLY, never a GRPO target)
python -m re_synth teacher-cot --samples ... --model copilot-claude-sonnet-5.5 \
  --out data/stage_2_RL_GRPO/cot_{he,fa} --ref data/pilot/samples.jsonl
# 5. GRPO rows + held-out dev split
python -m re_synth export-eval --samples data/stage_2_RL_GRPO/samples_{he,fa}.jsonl --chunk \
  --tiers data/stage_2_RL_GRPO/tiers.json --out data/stage_2_RL_GRPO/grpo_all.jsonl
```

Tiers (`re_synth/mine.py`): `easy`, `hard` (Sonnet right, Llama wrong), `weird`, `ambiguous`
(both wrong, LLM-written gold: excluded), plus the new **`frontier_hard`**.
`frontier_hard` = Sonnet wrong while the gold is deterministic (H6/H7/H13/H19 block). These samples
are kept and up-weighted.

Probe result (M2, 599 scored; chunked inference):

| model | slot-F1 all | he | fa | weakest hard cases |
|---|---|---|---|---|
| Llama-3.3-70B | 0.924 | 0.893 | 0.951 | H16 0.885, H15 0.895, H14 0.904, H18 0.911, H13 0.916 |
| Sonnet 5.5 (copilot) | 0.987 | 0.986 | 0.987 | H16 0.968, H15 0.973, H14 0.981 |

Tiers: easy 331, hard 177, **frontier_hard 84**, ambiguous 3, weird 4.

Teacher reasoning: `cot_*/teacher_raw.jsonl` stores Sonnet's full reasoning for every sample,
including failures, so we can study how a strong model handles the new cases.
It is shown in the viewer as a prediction column.

## Files

| File | Content |
|---|---|
| `seeds.jsonl`, `seeds_{he,fa}.jsonl` | seed articles (new URLs only; `seeds.jsonl` = both languages) |
| `samples_{he,fa}.jsonl` | generated samples: `text`, `gt`, `decoys`, `plan` (incl. `plan.table` / `plan.long`), `hard_cases`, `qc` |
| `preds_strong.jsonl`, `preds_weak.jsonl`, `tiers.json` | difficulty probe |
| `cot_{he,fa}/teacher_raw.jsonl`, `teacher_verdicts.json` | Sonnet reasoning (inspection) |
| `prefilter.jsonl` | SFT-model rollouts (n=8, T=1) with per-sample reward mean/std |
| `grpo_train.jsonl`, `grpo_dev.jsonl` | GRPO rows: `id`, `prompt` (messages), `gt`/`decoys`/`slot_types` (JSON strings), `text`, `meta` |

`grpo_dev.jsonl` (~60 samples, stratified by language × table/non-table) is held out and
reported next to the pilot test set.

## GRPO run

Background and citations: [docs/research/grpo_hyperparameters.md](../research/grpo_hyperparameters.md).
Trainer: **Unsloth + TRL `GRPOTrainer`**, `fast_inference=False` (HF generate). vLLM 0.32 needs a
separate bnb plugin for 4-bit LoRA, and the 8 GB GPU cannot hold vLLM plus training.

| Param | Value | Rationale |
|---|---|---|
| init | SFT adapter `outputs/sft-cot-qwen3-4b` loaded directly (trainable; no new `get_peft_model`) | continue stage 1 |
| `learning_rate` | 1e-5, constant, 5% warmup | LoRA RL typically 5e-6–2e-5 |
| `beta` | **0** | With a PEFT policy, TRL's reference = adapter **disabled** = the *base* model. KL>0 would pull the model back toward pre-SFT behaviour. Drift is tracked by pilot eval instead. |
| `loss_type` / `scale_rewards` | `dr_grpo` / `batch` | no length bias (Dr. GRPO); group-std scaling amplifies noise near the reward ceiling |
| `num_generations` | 4 (6 if memory allows) | KV cache of G × ~3.5k tokens on 8 GB; the pre-filter compensates for the small G |
| `temperature` / `top_p` | 1.0 / 0.95 | rollout diversity; keep JSON intact |
| `max_completion_length` | p95 of SFT completion length on this set (from the pre-filter) | big tables need long JSON; avoid truncation |
| `mask_truncated_completions` | False | Unsloth-documented NaN/zero-mask bug |
| `num_iterations` | 1 | on-policy; ratio = 1, so clip and IS level don't matter |
| batch | `per_device_train_batch_size`=G, `gradient_accumulation_steps`=4 → 4 prompts / step | 1 GPU |
| rewards | `extraction_reward` (composite F1 + null + decoy/halluc penalties) w=1.0, `format_reward` w=0.2; `slot_f1_reward` logged with w=0 | dense reward; low format weight avoids format gaming |

Protocol:
1. **Pre-filter** (vLLM, fast): the SFT model does n=8 rollouts per prompt at T=1.0. Drop prompts
   with reward std = 0 and mean ≥ 0.98 (already solved, zero gradient). Keep all others, and
   oversample `frontier_hard` and table prompts ×2.
2. **Pilot run** (~20 steps): check memory, s/step, `frac_reward_zero_std`, `completions/clipped_ratio`,
   and length growth. Spot-check rollouts for null-hacking (null everything) and think inflation.
3. **Full run** (~2 h budget). Save the adapter every 25 steps.
4. **Eval** each checkpoint offline with vLLM on the **pilot test set** (same thinking settings as
   `qwen3-4b-ft`) and on `grpo_dev`. Select by pilot slot-F1, not training reward.
   Results go at the end of `outputs/report/pilot_compare.html` as `qwen3-4b-grpo`.

Monitoring: `outputs/grpo-qwen3-4b/progress/charts.html` (reward, reward std / zero-std groups,
completion length, truncation, KL, grad norm, LR).
