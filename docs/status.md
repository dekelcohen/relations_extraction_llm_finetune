# Project status

Living summary of where this project stands and what's next. For full detail see the
[research survey](research/entity_relation_extraction.md) and the
[plan](plans/synth_dataset_and_finetune.md) (execution log in plan §10).

- Reproduction (exact training/eval files + commands): [reproduce.md](reproduce.md)

Where we are
Research + plan are done. The synthetic pipeline, scorer, mining, SFT/GRPO training
scripts and eval harness are implemented and smoke-tested. A **300-sample pilot**
(he/fa/en, 8 schemas, text/HTML/email) has been generated and scored against
Llama-3.3-70B (weak) vs Claude Sonnet 5.5 (strong) baselines — see
[plan §10](plans/synth_dataset_and_finetune.md#10-execution-status-pilot-run-2026-11-10)
for the full table. Headline: weak/strong slot-F1 gap is real but small (0.958 vs
0.984); most pilot samples are "easy" (only 42/298 mined as hard).

## Current stage: stage 2 — GRPO on hard he/fa cases (updated 2026-11-12)

Stage 1 is done. Qwen3-4B was QLoRA-SFT'd on Sonnet teacher reasoning plus the gold JSON
(`data/synth_hefa`, adapter `outputs/sft-cot-qwen3-4b`), reaching pilot slot-F1 0.958. The
report is at `outputs/report/pilot_compare.html`. Stage 2 continues that adapter with GRPO. The
dataset is described in [`datasets/stage_2_RL_GRPO.md`](datasets/stage_2_RL_GRPO.md); the
GRPO hyperparameter research is in
[`research/grpo_hyperparameters.md`](research/grpo_hyperparameters.md).

**Convention:** commit and push at every milestone (after synthetic generation, before GRPO,
after GRPO), including the data under `data/stage_2_RL_GRPO/`, so work can continue on
another machine from the same point. Artifacts over GitHub's 100 MB file limit (adapters,
checkpoints, large datasets) go to **private HF Hub repos** with `scripts/hf_sync.py push`,
which records them in `hf_artifacts.json`.

### Milestones
| # | Milestone | State |
|---|---|---|
| M1 | Stage-2 synthetic generation: 600 he/fa samples (H13–H19 tables, long docs), chunked long-doc inference + dedup merge | ✅ done (this commit) |
| M2 | Difficulty probe (Sonnet + Llama-70B, chunked) → `tiers.json`: easy 331 / hard 177 / **frontier_hard 84** / ambiguous 3 / weird 4. Slot-F1 Llama-70B 0.924 (he 0.893, fa 0.951), Sonnet 0.987 | ✅ |
| M3 | Teacher reasoning (inspection only) → `cot_{he,fa}` (kept 270 he / 265 fa); `grpo_train` 726 chunk rows (532 source docs) / `grpo_dev` 60 docs (24 H19, 16 frontier_hard); `scripts/split_grpo.py`; stage 2 in the viewer | ✅ |
| M4 | vLLM pre-filter (SFT adapter, **n=4** on a 240-prompt subset, prompts ≤ 2500 tok; 170 kept with reward std>0; then ≤1300-token completions → 111 unique / 191 rows after hard×2). SFT completion tokens p50 1004 / p95 2239. `train/grpo_unsloth.py`, `scripts/grpo_retry.sh` (auto-resume after OOM) | ✅ |
| M5 | GRPO run: **20 steps** (2 prompts × G=4 per step, lr 1e-5, β=0, dr_grpo, strict reward, max completion 1024, ~2.5 min/step, ~50 min). Final adapter = best on grpo_dev. Pilot slot-F1 SFT 0.9584 → GRPO **0.9591** (exact 0.722 → 0.725); grpo_dev 0.853 → 0.852; test_long 0.847 → **0.858**. Adapter on HF `Shani123/re-qwen3-4b-grpo`; report sections appended | ✅ (small effect) |
| T1 | Test set: realistic long docs (H20) → `data/test_long/` (40 he + 40 fa, TEST ONLY, [doc](datasets/test_long.md)). Slot-F1 Llama-70B ~0.974 (he 0.978, fa 0.969), Sonnet ~0.988 (he 0.992, fa 0.984). Open: per-case breakdown, harder v2 constructs | ✅ |
| A1 | Qwen3.8-27B on the shared A100-40GB (`scripts/pipeline`, `configs/runs/qwen3.8-27b.env`). **Base FP8** (thinking, T=0.6, max 5120 tok): slot-F1 pilot 0.936 / grpo_dev 0.621 / test_long 0.490, mostly truncation. **INT4 works (2026-11-16):** `RedHatAI/Qwen3.8-27B-INT4`, bf16 KV (the checkpoint defaults to fp8 KV, needs sm89+), `attention_backend=TRITON_ATTN` (the auto-picked FLASHINFER JIT-compiles and needs nvcc; that was the "cudagraph profiling" crash), max_num_seqs 48, **max_tokens 12288** / max_model_len 16384. Engine load 2–9 min. **Base int4** 0.953 / 0.928 / 0.855. **ft-int4** (27B SFT adapter via vLLM LoRA on the INT4 base) **0.984 / 0.972 / 0.967**, json_valid 1.0, 0 max_tokens hits, 23–33 prompts/min. **Hit-limit retry** (`eval/retry_hits.py`, `retry_<tag>` stage): the base hits are overthinking at the template's default `reasoning_effort=xhigh`. Re-generating only the hit chunks with `reasoning_effort=medium` closes all of them, and the substituted F1 is 0.967 / 0.991 / 0.984. Clean full run of base **int4-medium**: 0.966 / **0.987** / **0.986**, 0 hits, 20–24 prompts/min. It beats ft-int4 on grpo_dev and test_long because the ft CoT is too short (teacher CoT, not max_len). See the table and verdict below | ✅ (SFT adapter on HF `Shani123/re-qwen3.8-27b-sft-cot`) |
| A2 | **Stage 2 = rejection-sampled self-distillation (2026-11-17).** Data `data/sft_stage2/train_selfdistill.jsonl` (994 chunk rows): base INT4 at `reasoning_effort=medium` re-solved 680 stage-2/INSS/replay docs (902 chunk prompts), kept slot-F1=1 docs (581 + 38 on a seed-1 retry = 619/680; 57 dropped, 4 Sonnet fallbacks; trace think mean 3.4k chars, hard rows ≈3.7k vs base-medium 4.3–4.9k, stage-1 1.5k), trained under the medium instruction (README). SFT: continued the stage-1 adapter (r32, lr 5e-5, **2 epochs = 116 steps**, bs1/ga16, max_len 8704 = no truncation, 0 rows dropped except 15 traces >12k chars, val 3%), **peak VRAM 31.3 GB**, 129 min (≈65 s/step), eval loss 0.174 → 0.062, train loss 0.088 → 0.033. Adapter `Shani123/re-qwen3.8-27b-sft-cot-stage2`. slot-F1 (json_valid 1.0, 0 max_tokens hits everywhere): stage-2 medium pilot **0.962** / grpo_dev **0.992** / test_long **0.987** / hard_inss 0.906; stage-2 default 0.967 / 0.991 / 0.985 / 0.905; stage-1 default 0.984 / 0.972 / 0.967 / –; stage-1 medium 0.979 / 0.980 / 0.972 / 0.921; base-medium 0.966 / 0.987 / 0.986 / 0.911. Avg gen tokens/chunk of the stage-2 ft: 1.0k pilot, 2.2–2.6k grpo_dev/test_long (stage-1: 0.4–1.2k). |
| A3 | **4B SFT stage 2 (self-distill) on the local RTX 3070 8GB (2026-11-17).** Continued the 4B stage-1 adapter on `train_selfdistill.jsonl` (994 rows; lr 5e-5, 2 epochs = 116 steps, bs1/ga16, max_len 9728), **peak VRAM 6.2 GB alloc / 6.5 GB reserved**, 6664 s (≈56 s/step), eval loss 0.465 → 0.208, train loss 0.362 (step 2) → 0.098 (step 116, mean 0.169). Eval (vLLM `Qwen/Qwen3-4B-AWQ` + LoRA, thinking, max_tokens 5120) slot-F1 pilot **0.965** / grpo_dev **0.959** / test_long **0.931** / hard_inss **0.821**; json_valid 1.0 / 0.983 / 1.0 / 1.0; max_tokens hits 0/298, 1/104, 10/296, 0/68 prompts. vs 4B SFT s1 0.958 / 0.853 / 0.847 / –: +0.11 grpo_dev, +0.08 test_long; vs 4B GRPO test_long 0.858 / hard_inss 0.743. Still below the 27B (test_long 0.987), especially on test_long/hard_inss. Adapter `Shani123/re-qwen3-4b-sft-cot-stage2` | ✅ |
| H1 | Hard INSS dataset (`data/hard_inss/`, [README](../data/hard_inss/README.md)): pool 186 verified (he 97 / fa 89), train 147 / dev 39 (dev = 4th eval set, `prompt/gt/decoys/text/slot_types` format; docs up to 9.6k chars → chunked inference). Round 1: 92 short samples; round 2: 31 new long audit-driven he docs + 60 he→fa whole-sample translations. Sonnet vs Terra (10 each): Sonnet 6/10 verified, blind F1 ≈0.89 (harder) vs Terra 7/10, ≈0.95 (easier, "fictional" leaks) → Sonnet. Raw articles in private HF `Shani123/re-inss-raw` | ✅ |
| S2 | Stage-2 SFT data for Qwen3.8-27B (`data/sft_stage2/`, [README](../data/sft_stage2/README.md)): 860 rows = 481 stage-2 CoT (54 eval-id rows removed; 164 dup'd by audit emphasis H9/H20/H12/H2/H1/H13/H11/H18, fa-weighted) + 61 INSS teacher-CoT (61/147 kept, 85 teacher mismatch) + 138 replay. Max 8692 tokens; 76 rows flagged short CoT for ≥4 records. **Next:** SFT on the A100, eval on grpo_dev/pilot/test_long/hard_inss-dev; consider regenerating flagged short-CoT rows | ✅ data |


### A1 results: Qwen3.8-27B on the A100 (lenient slot-F1; `outputs/report/pilot_compare.html`; retry summary `outputs/eval/qwen3.8-27b-int4.retry_summary.jsonl`)
| model | max_tok | pilot F1 / json_valid / hit% / p/min | grpo_dev | test_long |
|---|---|---|---|---|
| qwen3.8-27b-fp8 (base) | 5120 | 0.936 / 0.90 / 10.7% / 5.8 | 0.621 / 0.73 / 49% / 3.2 | 0.490 / 0.725 / 67% / 2.7 |
| qwen3.8-27b-int4 (base) | 12288 | 0.953 / 0.977 / 2.7% / 13.1 | 0.928 / 0.95 / 10.6% / 6.9 | 0.855 / 0.988 / 26.7% / 5.1 |
| int4 + hits retried, seed 1 (still hit) | 12288 | 0.963 / 0.997 (2/8) | 0.932 / 0.967 (8/11) | 0.908 / 1.0 (48/79) |
| int4 + hits retried, reasoning_effort=medium (still hit) | 12288 | 0.967 / 1.0 (0/8) | 0.991 / 1.0 (0/11) | 0.984 / 1.0 (0/79) |
| int4, non-hit docs only (upper bound) | | 0.966 (290 docs) | 0.993 (52) | 0.990 (31) |
| **qwen3.8-27b-ft-int4** (LoRA) | 12288 | **0.984** / 1.0 / 0% / 22.7 | 0.972 / 1.0 / 0% / 28.7 | 0.967 / 1.0 / 0% / 33.2 |
| **qwen3.8-27b-int4-medium** (base, all prompts at `reasoning_effort=medium`) | 12288 | 0.966 / 1.0 / 0% / 23.6 (877 tok) | **0.987** / 1.0 / 0% / 20.1 (2249 tok) | **0.986** / 1.0 / 0% / 20.7 (2289 tok) |
| qwen3.8-27b-ft-int4-medium (LoRA, medium) | 12288 | 0.979 / 1.0 / 0% / 23.9 (359 tok) | 0.980 / 1.0 / 0% / 29.4 (1170 tok) | 0.972 / 1.0 / 0% / 35.5 (1069 tok) |

hit% = share of prompts (chunks) that hit max_tokens. p/min = prompts per minute; the first set's wall time includes the engine load.
(N tok) = avg generated tokens per prompt. Thinking chars per chunk: int4 xhigh 11k/17k/24k, int4-medium 2.5k/4.3k/4.9k,
ft-int4 0.7k/1.5k/1.5k, ft-int4-medium 0.7k/1.5k/1.5k.

**Verdict (ft vs clean base-medium, 2026-11-16):**
- The stage-1 adapter helps only on pilot, which is in-distribution with its SFT data: +0.018.
- It loses on the hard sets: grpo_dev −0.015, test_long −0.019 (base-medium 0.987/0.986 vs ft 0.972/0.967).
- ft at medium is slightly better than ft at xhigh on the hard sets (0.980/0.972). Its thinking length is unchanged, so the adapter ignores the effort instruction.
- Analysis (`scripts/compare_ft_base.py`, `outputs/analysis/ft_vs_basemed.<set>.json`):
  - Errors, ft vs base-medium: 62 vs 44 on grpo_dev, 193 vs 116 on test_long.
  - Docs better per model, ft/base: grpo_dev 8/7, test_long 17/20. The losses are concentrated in a few multi-record docs.
  - ft-only errors are recall errors, by category: missed_value 67, missed_record 40 (test_long); missed_record 14, decoy 21 (grpo_dev).
  - ft-only errors by root: long_range_or_coref 61, multi_entity_bleeding 48 (test_long); multi_entity_bleeding 19, distractor_or_role 22 (grpo_dev).
  - Worst ids: test_long fa-t3-00018 (whole list of records dropped), fa-t3-00012, he-t3-00039, he-t3-00018, fa-t3-00013; grpo_dev he-2-00239, fa-2-00169.
- Hypothesis (a): the SFT max_len 3072 is NOT the cause. auto max_len covered all data (token max 2675, nothing truncated).
  - The cause is the teacher CoT itself: 651 chars mean (p90 1021) in `sft_cot_train.jsonl`. ft thinks about 1.5k chars per chunk, vs 4.3–4.9k for base-medium.
  - Within ft, losses have only slightly longer thinking than wins (1.7k vs 1.5k). So the short CoT is a model-wide habit, not a per-doc one; it under-deliberates on long, multi-record chunks.
- Hypothesis (b), LoRA drift (bnb-4bit-trained adapter on the GPTQ INT4 base): untested. vLLM 0.33 has no bitsandbytes support; see Open issues.

### 27B-ft-stage-1 error analysis (report sections, `eval/strong_gap.py`, `eval/append_errors.py`)
- Report: full Error analysis for `qwen3.8-27b-ft-int4` (root causes, loses-vs-INT4/FP8-base tables, examples with viewer links) for pilot, grpo_dev, test_long; `qwen3.8-27b-int4-medium` and `ft-int4-medium` are columns in all three tables.
- `outputs/audit/27b_vs_strong.{md,json}`: 27B errors where Sonnet AND Llama are right (audit-excluded), ft-int4 kept: pilot 14, grpo_dev 30, test_long 99 (INT4 base 42/59/389); ends with a ranked stage-2 SFT emphasis list.

### Resume on another machine
1. `git pull`. All code, docs and stage-1/2 data (`data/pilot`, `data/synth_hefa`,
   `data/stage_2_RL_GRPO`) are in the repo.
2. Create `.env` in the repo root with `OPENROUTER_API_KEY=…` and `HF_TOKEN=…` (HF account
   `Shani123`). It is git-ignored; never commit it. Pass it with `RE_ENV_FILE=$PWD/.env`.
3. Clone `llm_providers` next to this repo and run with `PYTHONPATH=../llm_providers`. Copilot
   models (`copilot-<model id>`) need the Copilot CLI to be logged in.
4. `python scripts/hf_sync.py pull` downloads every artifact listed in `hf_artifacts.json`. For
   example, the SFT adapter (+ `checkpoint-64`) comes from the private
   `Shani123/re-qwen3-4b-sft-cot` into `outputs/sft-cot-qwen3-4b`. It needs `huggingface_hub`,
   which is in the container env `/data/envs/re`.

### Stage-2 results (lenient slot-F1, same thinking settings; `outputs/report/pilot_compare.html`, last two sections)
| set | n | SFT (`qwen3-4b-ft`) | GRPO (20 steps) | ckpt-10 | Llama-70B | Sonnet 5.5 |
|---|---|---|---|---|---|---|
| pilot (all) | 298 | 0.958 | **0.959** | – | 0.963* | 0.984 |
| grpo_dev (all) | 60 | 0.853 | 0.852 | 0.826 | 0.937 | 0.989 |
| grpo_dev he / fa | 30/30 | 0.871 / 0.836 | 0.880 / 0.825 | 0.861 / 0.792 | 0.924 / 0.948 | 0.994 / 0.984 |
| grpo_dev H13 / H18 / H19 | 33/20/24 | 0.823 / 0.816 / 0.876 | 0.826 / 0.812 / 0.866 | – | 0.922 / 0.929 / 0.948 | 0.987 / 0.986 / 0.993 |
| grpo_dev frontier_hard | 16 | 0.782 | 0.795 | 0.768 | 0.923 | 0.960 |
| test_long (all) | 80 | 0.847 | **0.858** | – | 0.974 | 0.988 |
| test_long he / fa | 40/40 | 0.892 / 0.797 | 0.888 / 0.825 | – | 0.978 / 0.969 | 0.992 / 0.984 |
| test_long H13 / H15 / H18 | 34/19/18 | 0.791 / 0.765 / 0.748 | 0.822 / 0.808 / 0.757 | – | 0.964 / 0.952 / 0.952 | 0.980 / 0.970 / 0.968 |
(*Llama-70B pilot figure is the lenient one from the report.) GRPO's null accuracy on grpo_dev rose 0.456 → 0.623 and the
decoy hit rate fell 0.186 → 0.146; exact_sample 0.317 → 0.350. JSON validity 0.983 → 0.967 on dev (truncation at 1024 tokens).
**Verdict:** no regression on the pilot and a small gain on test_long (+0.011, mostly fa tables), but 20 steps are far too
few to close the gap to Llama-70B / Sonnet on big tables (SFT 4B ≈ 0.82 vs 0.92–0.99). Keep the SFT adapter as the default;
GRPO is optional (+0.01). Training curves: reward is noisy (0.12–0.99 per step with only 8 completions/step); 25–87% of
completions were clipped at 1024 tokens early (reward 0), `frac_reward_zero_std` 0–1.

### Next items
- [ ] **Qwen3.5-9B:** stage 1 completed on the shared A100; stage 2 was launched with `qwen3.5-9b-stage2`, but its training and completion remain to be verified. The stages use the 4B stage-1 data and hyperparameters, then the shared self-distillation mix, including long synthetic and INSS samples. Verify the HF evaluation backend before evaluating; evaluate both adapters on pilot, grpo_dev, test_long and hard_inss, then add their measured results to the web summary and dataset-specific error analyses, with viewer prediction sources. Commit and push the generated reports for viewing on another machine (`outputs/` is ignored). Release and verify our GPU VRAM after the final stage or a failure, without disturbing other users.
- [ ] **A2 follow-up:** stage 2 vs stage 1 (medium): grpo_dev +0.012, test_long +0.015 (now ≥ base-medium 0.987/0.986 by +0.005/+0.002, i.e. within noise), **pilot −0.017** (0.979 → 0.962 ≈ base-medium 0.966: the pilot gain over base from stage 1 is gone, so some forgetting of the short-CoT precision), hard_inss −0.015 (0.921 → 0.906, n=39 docs, base-medium 0.911; not a win: no hard_inss gain from INSS rows, 40% of INSS docs failed the F1=1 filter so the hardest cases are under-represented). Strong-gap items (27B wrong, Sonnet+Llama right; `outputs/audit/27b_vs_strong.md`, stage-2 = medium): pilot 14 → 40, grpo_dev 30 → 1, test_long 99 → 27. Per root cause stage-1 → stage-2: multi_entity_bleeding grpo_dev 19 → 1, test_long 44 → 8; long_range_or_coref test_long 52 → 9; distractor_or_role grpo_dev 10 → 0, test_long 3 → 7; fa test_long 61 → 17 (he 38 → 10), pilot fa 5 → 16 / null_slot_filled 2 → 15 (new pilot regressions: span/null conventions). Ideas: mix pilot-style short-CoT replay back in, longer INSS/hard_inss self-distill with 2–4 samples per doc, 1 more epoch on hard rows. Only 1 sample/doc and the run used the whole 2 epochs (data prep ran ahead of the 0.8-epoch override, which was not applied).
- [x] **A1:** full base INT4 eval at `reasoning_effort=medium` and ft-int4 at medium are done (`BASE_INT4_MEDIUM_*`). Base-medium beats ft on grpo_dev and test_long; see the A1 results verdict.
- [x] **A1 SFT v2 (done as A2, self-distill):** make the teacher CoT longer and more deliberate. Current mean is 651 chars; target about 3–5k chars per chunk, roughly what base-medium writes.
      - Option: self-distill rejection-sampled base-INT4-medium traces (`grpo_dev`-style data, keep only F1=1 outputs).
      - Over-sample long multi-record chunks with list/coref cases.
      - Keep auto max_len, which will rise to about 4–6k tokens.
      - Render the training data under the `reasoning_effort=medium` instruction.
      - Re-evaluate against `qwen3.8-27b-int4-medium`, which is now the baseline to beat.
- [ ] **A4 (started 2026-11-18, not finished):** Qwen3.5-4B run requested after all. `Qwen/Qwen3.5-4B` / `unsloth/Qwen3.5-4B`
      exist (`Qwen3_5ForConditionalGeneration`, hybrid linear/full-attention, VLM wrapper); confirmed `unsloth`
      2026.11.4 loads+patches it in 4-bit on local3070 and `AutoModelForCausalLM` resolves the text-only
      `Qwen3_5ForCausalLM`, so `train/sft_unsloth.py` works unchanged. No `-unsloth-bnb-4bit`/`-AWQ` checkpoint
      exists yet, so eval uses a new `hf` backend in `eval/predict.py` (bitsandbytes 4-bit + `.generate()`, no
      vLLM). Configs added (`configs/runs/qwen3.5-4b[-stage2].env`); **training/eval not yet run** — resume on the
      4xA100 server, see [reproduce.md §6](reproduce.md#6-qwen35-4b-new-base-in-progress--started-2026-11-18).
      Once results exist, add `qwen3.5-4b-ft[-stage2]` to `scripts/build_report.sh` / `eval/build_summary.py`
      summary table + per-dataset error analysis (same pattern as `qwen3-4b-ft-stage2`).
- [x] Probe done (M2). 84 `frontier_hard` samples is enough, so no harder constructs for now.
      Candidates for a later round: footnote overrides, in-cell changes (`<del>old</del> new`),
      rowspan-merged cells, cross-table ID joins.
- [ ] Teacher-CoT pass (inspection only); optionally add stage 2 to the viewer manifest
      (`viewer_sources.json`).
- [ ] `export-eval --chunk` → split out `grpo_dev` (~60, stratified, H19-heavy) and `grpo_train`.
- [x] Pre-filter, GRPO script, run, eval, report (see results above).
- [ ] **Longer GRPO**: ≥100 steps need faster rollouts (HF generate costs ~2.5 min per 2 prompts on the 3070). Options: vLLM
      colocate with a bnb plugin, a bigger GPU, shorter thinking (SFT p50 is 1000 tokens), or GRPO on a no-think SFT.
- [ ] Raise `max_completion` (1024 clipped many rollouts) once rollouts are cheaper; log per-completion lengths in the pre-filter.
- [ ] Weakest area for the 4B model: big tables with ditto marks / two-level headers / pipe tables (H15, H16, H18) and fa.
- [ ] Not done: re-scoring the pre-filter with the strict reward (completions were not stored); the first 2 launches OOM'd
      (G=4, 1536 tokens), so the pre-filter's length cut (≤1300 tokens) and the final 1024/1900 caps differ.
- **Training reward is STRICT** (user decision): exact slot = 1, lenient-only partial = 0.5 TP/FP/FN, main-entity name
      scored as a slot, exact decoy check, +0.1 all-exact bonus; eval stays lenient + `exact_sample`. Table in
      [`datasets/stage_2_RL_GRPO.md`](datasets/stage_2_RL_GRPO.md#reward-lenient-eval-vs-strict-training).
- [ ] Update this file and the dataset doc with the actual counts and results.

### Open issues
- **A1 hypothesis (b), LoRA drift bnb-4bit→GPTQ, untested.** The `ftlim_bnb4` stage failed: vLLM 0.33 has no `bitsandbytes` quantization ("Unknown quantization method").
  - Testing it would need HF/unsloth generation on the bnb base (slow) or the adapter on the bf16 base.
  - It is unlikely to be the main cause: ft-only errors are recall/short-CoT errors, not format or garbage drift, and json_valid is 1.0.
- **A100 monitoring (fixed 2026-11-16):** the watchdog now counts only OUR container's GPU processes (`docker top` pids ×
  `nvidia-smi --query-compute-apps` / `pmon`) and writes `alerts.log` (gpu_idle, stage_fail, pipeline_dead, deadline). The resolved config goes to
  `config.log`. FAIL lines carry the root-cause error line, and engine init times out after `ENGINE_INIT_TIMEOUT` (1500 s). Agents wait with
  `remote.sh --machine a100 wait` (returns on the first new DONE/FAIL/ALERT line or TIMEOUT + 1-line status), not with async notifications.
  There is still no auto-retry.
- **A100 / Qwen3.8-27B eval:** engine-init failures were missing gcc for Triton, the flashinfer sampler JIT
  needing nvcc (`VLLM_USE_FLASHINFER_SAMPLER=0`), and `max_num_seqs` > Mamba cache blocks (41 → 40). Engine load
  ~6.6 min. Eval resumes per prompt from `<out>.partial.jsonl`. If a 4-bit base column is needed, use a W4A16
  (Marlin) checkpoint such as `RedHatAI/Qwen3.8-27B-INT4` instead of vLLM bnb.
- **Probe fairness:** the H13 "date" column is label-ambiguous when a distractor date column
  exists. The H15 "previous supplier" vs current one may also be too subtle. Review these
  in the viewer before trusting `frontier_hard` labels.
- **Tables are not hard enough for Sonnet yet:** on an early 24-table probe Sonnet scored
  mean sample-F1 0.982. Its errors were ditto marks copied literally, previous/current
  columns confused, and distractor columns. Llama-70B results on stage 2 are still pending.
- **H19 token budget for GRPO on 8 GB:** whole H19 docs need about 4.5k input + 2.5k output
  tokens. GRPO therefore trains on clean **chunks** (`export --chunk`), while dev/eval stay
  doc-level with chunked inference.
- **Chunk-merge edge cases:** a chunk can mangle an entity name, as when Sonnet wrote the type
  `bank_account` instead of the name. These records are merged only if they share an ID slot;
  schemas without an ID slot can still produce duplicates in that case.
- **Generator model:** Sonnet passed QC on 600/600 samples. Fallbacks if it degrades:
  `copilot-gpt-6.1-terra`, `-sol`, or Opus 5.5 with effort medium.
- **E3.2 (future):** parse tables with the `excel_table_parsing` repo (`xlparse`) as a
  preprocessing step before extraction (plan §4.6).

## Immediate next step: manual review of the pilot testset
**Before scaling the build (3000/lang) or hardening further, the 300-sample pilot
needs a manual pass to check it's actually representative of the target domain** —
i.e. does synthetic text/formatting resemble the private company documents this
model will run on in production, not just generic Wikipedia/news prose?

Use the **[dataset viewer](../viewer/index.html)** for this pass (README §5):
`pip install -r viewer/requirements.txt && python viewer/server.py` and open
`http://127.0.0.1:8000` (dataset-selector dropdown defaults to "testset (pilot)"), or
open `viewer/index.html` directly with no server for the static fallback. Filter by
language/format/hard-case in the sidebar and read each sample with its gold
entities/relations highlighted directly in context (decoys and per-model prediction
overlays are optional, off by default, so the text stays readable while skimming many
samples quickly — toggle individual models on to see ✓/✗/∅ correctness badges and ⚠
decoy-hit badges inline). The **Prompt chain** tab shows the background + chained
extraction prompts used for review/design per schema.

Review checklist (per sample, sampled across languages/formats/schemas):
- [ ] **Format realism** — do the HTML tables, email headers and plain-text layouts
      look like real internal docs (invoices, shipment notices, HR records, asset
      logs), or do they read as generic/encyclopedic?
- [ ] **Entity/relation realism** — are main-entity types, slot values (IDs, dates,
      locations, statuses, orgs) and their typical co-occurrence patterns close to
      what shows up in the user's real documents?
- [ ] **Distractor realism** — are the planted decoys (irrelevant nearby dates/
      locations/orgs, H1–H12 hard cases) the kind of ambiguity that actually occurs
      in real docs, or are they arbitrary/synthetic-looking?
- [ ] **Language fidelity (he/fa)** — natural phrasing, correct date formats
      (Hebrew/Jalali), no translation artifacts or broken diacritics/gershayim.
- [ ] **Noise/boilerplate** — real docs often have disclaimers, signatures, repeated
      headers, OCR noise; check whether the synthetic set under- or over-represents
      this compared to the private corpus.

Output of the review: a short gap list (what's missing or unrealistic), used to
drive the next round of schema/hardcase tuning — not a pass/fail gate.

## Then: how to make the testset harder
The pilot's own "Next" section (plan §10) already lists the first round of
hardening, to be combined with findings from the manual review above:
- More records per document (3–6 instead of ~1–2), so candidate-pair disambiguation
  matters more.
- More decoys per slot (2–3 instead of 1).
- Longer emails/threads (multi-message, nested quoting) and longer HTML tables.
- More H6 (subtitle → list, N lines below) and H11 (near-but-unrelated-date) cases,
  where Llama was weakest (slot F1 0.895 / 0.970 vs Sonnet ≈ 0.99).
- If the manual review surfaces domain-specific patterns (e.g. recurring document
  templates, house-style abbreviations, specific ID formats) not currently covered
  by `re_synth/schemas.py` / `re_synth/hardcases.py`, add them as new schemas/hard
  cases rather than only scaling existing ones.
- Re-run mining after each hardening round; target ≥ 30% hard-tier yield (vs ~14%
  in the pilot) before scaling to the full 3000/lang build.

## Longer-term next steps (unchanged from plan §10)
1. Scale the build to 3000 samples/language once hardening above is applied.
2. Convert Re-DocRED and add the P1 public datasets.
3. Run SFT → GRPO on 4×A100 (README §3).
4. Evaluate on `grpo_test` + the forgetting suite (`eval/forgetting.sh`).

## Open questions (carried from plan §9, still unanswered)
1. Exact main-entity types and slots of interest beyond vehicle/person/org/shipment
   — real example documents (even 10, redacted if needed) would let the manual
   review above be done against actual private-doc patterns instead of guesswork.
2. Output detail: verbatim strings only, or normalized dates/IDs too? Evidence
   spans required?

- 2026-11-17: Llama-3.3-70B (OpenRouter, $0.06) on hard_inss dev: slot-F1 0.887 (P 0.936 / R 0.843), json 1.0, exact 0.33 -> below 27B base-medium 0.911 / ft-s1 0.921 / ft-s2 0.906. Preds `data/hard_inss/preds_weak.jsonl`. Note: dev.jsonl has a duplicate id (hard-inss-438720-05-fa, 38 unique).
- 2026-11-17: Qwen3-4B GRPO (local 3070, AWQ+LoRA, thinking) on hard_inss dev: slot-F1 0.743 (P 0.860 / R 0.655), decoy hit 4.4%, halluc 4.6%, exact 0.05.
