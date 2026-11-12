# Project status

Living summary of where this project stands and what's next. For full detail see the
[research survey](research/entity_relation_extraction.md) and the
[plan](plans/synth_dataset_and_finetune.md) (execution log in plan §10).

## Where we are
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
| M4 | vLLM pre-filter (SFT adapter, **n=4** on a 240-prompt subset, prompts ≤ 2500 tok; 170 kept with reward std>0; then ≤1300-token completions → 111 unique / 191 rows after hard×2) + `train/grpo_unsloth.py`. SFT completion tokens p50 1004 / p95 2239, mean lenient reward 0.65. HF generate ≈ 330 s per 4-prompt step (G=4), so the full run is ~20 steps in 2 h | 🔄 (committed before GRPO) |
| M5 | ~2 h GRPO run, checkpoint eval on pilot + `grpo_dev`, `qwen3-4b-grpo` added at the end of the report (**commit after GRPO**; push the adapter + checkpoints with `hf_sync.py`) | ⏳ |

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

### Next items
- [x] Probe done (M2). 84 `frontier_hard` samples is enough, so no harder constructs for now.
      Candidates for a later round: footnote overrides, in-cell changes (`<del>old</del> new`),
      rowspan-merged cells, cross-table ID joins.
- [ ] Teacher-CoT pass (inspection only); optionally add stage 2 to the viewer manifest
      (`viewer_sources.json`).
- [ ] `export-eval --chunk` → split out `grpo_dev` (~60, stratified, H19-heavy) and `grpo_train`.
- [ ] Pre-filter, write `train/grpo_unsloth.py`, smoke test, pilot run, ~2 h run; evaluate and
      add to the report.
- **Training reward is STRICT** (user decision): exact slot = 1, lenient-only partial = 0.5 TP/FP/FN, main-entity name
      scored as a slot, exact decoy check, +0.1 all-exact bonus; eval stays lenient + `exact_sample`. Table in
      [`datasets/stage_2_RL_GRPO.md`](datasets/stage_2_RL_GRPO.md#reward-lenient-eval-vs-strict-training).
- [ ] Update this file and the dataset doc with the actual counts and results.

### Open issues
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
