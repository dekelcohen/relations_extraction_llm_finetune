# Plan: Synthetic he/fa Entity–Relation Dataset + Fine-tuning a Generic Extractor

Companion research: [../research/entity_relation_extraction.md](../research/entity_relation_extraction.md). Its Tables 1–3 cover techniques, datasets and synthesis methods, and §3 lists hard cases H1–H12.

**Assumptions** (the user was unavailable for clarifying questions; change these here if they are wrong)

| Item | Assumption |
|---|---|
| Base model | **Qwen3-8B** (instruct, thinking off). Qwen3-14B if quality is short. A/B **DictaLM-3.0-12B** for Hebrew. |
| Compute | 1× 80GB GPU (A100/H100) on Linux/WSL2. Frontier and Llama-3.3-70B through `D:\NLP\llm_providers` (OpenRouter etc.). |
| Languages | Hebrew, Persian, English (en for replay and transfer). |
| Output | JSON list of records: main entity + slots; null for absent slots; optional evidence spans. |
| Prompt | Schema + guidelines + 1–3 few-shot examples, supplied by the user at inference time. Must stay generic (new entity types without retraining). |
| Licenses | Prefer CC-BY-SA / ODC-By / CC0 sources. Non-commercial datasets go to eval only. |
| Out of scope | The local `pn_summary` (fa) summarization dataset. The user said to ignore it. |

---

## Table A. Pipeline at a glance

| Phase | Output | Main tools | Exit criterion |
|---|---|---|---|
| P0 Baselines | Zero-shot scores on a 300-sample pilot per language | `llm_providers` (OpenRouter), vLLM | Gap between Qwen3-8B, Llama-3.3-70B and frontier is measured |
| P1 Public data | English/multilingual IE mix (replay + schema diversity) | HF `datasets` | Converted to our JSON format with licenses logged |
| P2 Synthetic he/fa/en | 30k–60k verified samples (text, HTML, email) with H1–H12 tags | Generator package (`re_synth/`), frontier/cheap LLM via `llm_providers`, vLLM | Verifier pass rate ≥ 90%; manual audit error < 3% |
| P2b Adversarial mining | "Hard" subset where Llama-3.3-70B fails and frontier succeeds | `llm_providers` | ≥ 5k hard samples across H1–H12 |
| P3 SFT | LoRA adapter v1 | TRL + PEFT + Liger (Unsloth for iteration) | Valid JSON ≥ 99.5%; record-F1 ≫ zero-shot; forgetting within budget |
| P4 GRPO | LoRA adapter v2 | TRL `GRPOTrainer` + vLLM colocate | +F1 on hard/OOD splits; KL and forgetting within budget |
| P5 Repair (optional) | Merged or distilled model | PEFT merge / TRL GKD | General scores within 1–2 points of base |
| P6 Eval and serve | Report + vLLM multi-LoRA endpoint | vLLM, lm-eval-harness, custom scorer | Per-hard-case F1 dashboard |

## Table B. Public datasets to gather (P1)

| Use | Dataset | Lang | License (verify) | Role |
|---|---|---|---|---|
| Train | IEPile (RE/NER subsets) | en | NC? **[check]** | Schema-instruction diversity, negative schemas |
| Train | Pile-NER-type (UniversalNER) | en | see card | Open entity-type vocabulary |
| Train | B2NERD | en/zh | see card | Diverse NER, pruned |
| Train | GuideX pretraining data | en | see card | Guideline-in-prompt format |
| Train | Re-DocRED + Re-DocRED-CF (`train_mix`) | en | MIT | Doc-level RE + counterfactual robustness |
| Train | MMUIE (weak train split) | en | see card | Long-document, multi-domain UIE |
| Train | Extract-0 | en | see card | Generic document → JSON |
| Train (filtered) | ScrapeGraphAI-100k | 15 langs | see card | Real web schemas. **Re-label** with frontier first. |
| Train | PERLEX train | fa | unspecified | Only Persian RE |
| Train | PEYMA, ArmanPersoNER, MultiCoNER-fa, WikiANN he/fa | fa/he | various | NER-only replay in target languages |
| Eval | NEMO (he), PERLEX test (fa), Re-DocRED test, MMUIE test, SciER test | he/fa/en | various | Real-text checks |
| Replay | Tulu-3 SFT mix subset, FarsInstruct, Hebrew chat/instruct (Dicta) | en/fa/he | various | Anti-forgetting replay |

All are converted by `re_synth/convert/*.py` into the same prompt+JSON format as the synthetic data (§3). A dataset with types but no guidelines gets auto-written guidelines (GuideX-style).

---

## 1. Phase 0: baselines and a pilot set (week 1)

1. Hand-write **10 schemas** of main entity + slots, e.g.:
   - vehicle {plate, color, location, date, status, owner_org}
   - person {id_number, role, location, date, employer_org}
   - shipment {tracking_id, origin, destination, date, status, carrier_org}
   - device, account, event, contract, …
2. Build a **pilot of 300 samples per language** with a quick version of P2 (below), audited manually.
3. Run zero-shot with the identical prompt:
   - Qwen3-8B, Qwen3-14B and DictaLM-3.0 (vLLM).
   - Llama-3.3-70B (OpenRouter, provider pinned).
   - Two frontier models (e.g., GPT-4.1/5-class, Gemini-2.5-Pro, Claude-class).
4. Score with the evaluator (§6). Expected outputs: the per-hard-case gap, plus a check that frontier ≥ 90% F1 on the pilot. If frontier can't reach that, the guidelines or GT are ambiguous: fix them before scaling.

## 2. Phase 2: synthetic pipeline (scrub → inject → render → GT by construction)

### 2.1 Sources (seed articles)
| Source | he | fa | Why |
|---|---|---|---|
| Wikipedia (`wikimedia/wikipedia` 20231101.he/.fa) | ✓ | ✓ | Clean, diverse topics |
| Structured Wikipedia (infobox/tables) | **[check config]** | **[check]** | Real tables for HTML |
| FineWeb-2 `heb_Hebr` / `pes_Arab` | ✓ | ✓ | News, forums, ads, real web genres |
| HPLT 2.0 cleaned | ✓ | ✓ | CC0 |
| HeDC4 / HeSum | ✓ | — | Hebrew web/news |
| naab | — | ✓ | Large Persian corpus |
| English Wikipedia / CC-News | en | en | Parallel versions and replay |

Seed filter: 150–1,500 words, language-ID ≥ 0.9, dedup (MinHash), genre classifier for stratification (news, encyclopedic, business, forum, government notice).

### 2.2 Steps

| Step | What | How | Verification |
|---|---|---|---|
| 1. **Schema sampler** | Choose main-entity type + 2–6 slot types + guideline text | Library of ~40 hand-written schemas + GuideX-style LLM-generated schemas (S6); names paraphrased per sample (GoLLIE) | Schema lint (unique names, definitions present) |
| 2. **Record sampler** | Generate 1–5 main-entity records with slot values (some null) | Python faker-style generators per locale: Israeli/Iranian plates, IDs, Jalali/Hebrew/Gregorian dates, real gazetteer places, fictional org names (counterfactual, H8) | Format validators per slot type |
| 3. **Scrub** | Remove or neutralize existing mentions in the seed that would collide with the schema (other cars, dates, orgs of the target types) | LLM rewrite with instructions "replace/remove entities of types X"; NER pre-tag (GLiNER multi / dictabert-ner) gives candidates | Re-run NER: no unintended target-type entities left, or they are labeled as non-target |
| 4. **Plan distractors** | Choose 0–3 hard-case families (H1–H12) + positions | Rule-based planner: e.g., H2 = target slot 2–3 sentences away with a decoy location between | Plan stored in metadata |
| 5. **Inject** | Rewrite the article so it naturally contains the records + distractors | LLM (cheap frontier, e.g., gemini-2.5-flash) given article + records + plan → returns text with **inline span markers** `⟦slot:id⟧…⟦/⟧` | **Exact-match span check**: every GT value appears in a marker; markers removed after offsets are recorded |
| 6. **Render** | Choose the format | Plain text / HTML with `<table>`, lists under subtitles (H6, H7) / email with long headers and quoted thread (H3, H5) / forms. Jinja templates + LLM for filler | HTML parses; table cells contain GT values |
| 7. **GT build** | JSON records + evidence spans + hard-case tags | Directly from steps 2, 4, 5 (no labeling needed) | JSON schema validation |
| 8. **Judge filter** | Catch unnatural text, contradictions, leftover leaks | (a) Frontier model extracts blind with the same prompt → disagreement with GT = flag. (b) LLM judge for fluency/consistency (1–5). | Keep if judge ≥ 4 and (blind extraction == GT **or** disagreement is explained by the hard case and accepted by a 2nd judge) |
| 9. **Parallelize** | Produce en/he/fa versions of the same records (KnowCoder-X) | Translate the injected doc while keeping markers; re-verify spans | Span check per language |
| 10. **Dedup and split** | Train/dev/test split by seed article and schema family; a held-out-schema test split | MinHash + schema-family split | No seed or schema overlap with the test set |

**Distractor generators (H1–H12)** are parameterized: distance (sentences/lines), number of decoys, decoy role (manufacture date, birthplace, previous owner, forwarding timestamp) and placement. Every sample stores `hard_cases: ["H1","H4"]` so F1 can be reported per case. User examples become unit-test fixtures:
- Mazda/Yehud: the 1/1/1955 invention date is a decoy; 12/2/26 is the pick-up date (H1).
- Hertz subtitle → list three lines below (H6).
- Email header with the title far from the body (H3).
- ent1's date bleeding to ent2 (H4).
- Previous-message dates (H5).

**Locale specifics (H10):**
- Dates in Gregorian `d/m/yy`, Hebrew month names, Hebrew calendar (תשפ"ו), Jalali (۱۴۰۴/۰۸/۱۲) with Persian digits.
- GT stores both the surface string and a normalized ISO date, if the schema asks for normalization.
- Mixed RTL/LTR plates and IDs; Persian ی/ي and ک/ك variants.

### 2.3 Adversarial mining (P2b)
1. Run **Llama-3.3-70B** (pinned OpenRouter provider, T=0) and **frontier** (2 models) on every verified sample.
2. Tier each sample:
   - `easy`: both correct.
   - `hard`: Llama wrong, frontier correct. This is the **target tier**.
   - `ambiguous`: both wrong. Send to a second judge; fix GT or drop.
   - `weird`: Llama right, frontier wrong. Inspect.
3. Analyze which H-cases drive `hard`. Re-weight the distractor planner toward them; iterate 2–3 rounds.
4. Budget estimate: ~60k samples × 3 models × ~2k tokens ≈ 360M tokens. Use `llm_cache` and batch APIs. Start with 5k samples per round.

### 2.4 Target mix (first SFT run, ~60k examples)
| Slice | Share |
|---|---|
| Synthetic he | 25% |
| Synthetic fa | 25% |
| Synthetic en (parallel) | 10% |
| Public en IE (IEPile/Pile-NER/B2NERD/GuideX/Re-DocRED-CF/MMUIE/Extract-0) | 15% |
| Public fa/he NER + PERLEX | 5% |
| General replay (en/he/fa chat/instruct) | 20% |

Within the synthetic slices: ~40% `hard`, 20–30% of slots null (H9), 30% HTML/email formats, and a held-out schema family for zero-shot eval.

### 2.5 Code layout (mirrors `D:\NLP\Tables\excel_table_parsing\xlparse\gen`)
```
re_synth/
  scenarios/      # schema library (YAML) + guideline texts + user hard-case fixtures
  lexicon/        # locale generators: plates, IDs, dates (jalali/hebrew), gazetteers, org names
  source/         # seed loaders (wikipedia, fineweb2, hplt, naab, hedc4) + filters
  scrub.py        # step 3
  plan.py         # step 4 distractor planner (H1–H12)
  inject.py       # step 5 LLM injection with span markers
  layout/         # step 6 renderers: text, html_table, html_list, email, form
  verify.py       # span checks, schema validation, judge filter
  mine.py         # P2b adversarial mining
  convert/        # public datasets -> our format
  build.py        # orchestration, sharding, resume, dedup, splits
  report.py       # stats per lang/format/hard case
llm.py            # thin wrapper over llm_providers (openrouter, cache, usage)
```
- LLM access: `from openrouter import call_openrouter` (via `PYTHONPATH=D:\NLP\llm_providers` or `pip install -e`). Use `response_format` for JSON. Keys come from `.env` (never committed).
- Use `llm_cache` and usage tracking for every call. Log per stage: counts in/out, rejection reasons, cost.

## 3. Prompt and output format (shared by train / RL reward / serving)

```
<system> You extract records about the MAIN ENTITY type defined below. Follow the guidelines exactly. Return JSON only.
<user>
SCHEMA:
  main_entity: vehicle — a specific car/truck/motorcycle ...
  slots:
    plate (id): ... ; color: ... ; location: where the vehicle is currently kept/parked (NOT where it was made/bought) ;
    date: the date of the main event (pick-up/arrival), NOT manufacture or message timestamps ;
    status: ... ; owner_org: organization the vehicle belongs to, may be stated in a heading above a list
GUIDELINES: one record per distinct vehicle; null when a slot is not stated for THAT vehicle; do not copy slots between vehicles; ...
EXAMPLES: <1–3 short examples>
TEXT: <plain text | HTML | email>
```
Output:
```json
{"records":[{"main_entity":"Black Mazda","plate":null,"color":"black","location":"Yehud","date":"12/2/26","status":"awaiting pick-up","owner_org":null,
  "evidence":{"location":"parked in Yehud","date":"pick it tomorrow on 12/2/26"}}]}
```
- Keys follow the schema exactly. Values are verbatim surface strings (plus `*_norm` when requested). `evidence` is optional (on in 50% of training prompts, controlled by an instruction flag).
- An optional `<reasoning>` block (MimicSFT template, ≤ 120 tokens) can be enabled by a flag for RL experiments. Default off at serving.
- Augmentation (GoLLIE): paraphrase slot names and definitions, shuffle slot order, drop some definitions, vary the number of few-shot examples (0–3).

## 4. Training

### 4.1 Stage 1: SFT (LoRA)
- Qwen3-8B, LoRA r=64, α=64–128, rsLoRA, **all linear layers**, dropout 0.05.
- LR 1e-4 cosine, 1–2 epochs, seq len 8k (HTML/email up to 16k), packing off for long docs.
- Loss on assistant tokens only.
- Variants to A/B (in priority order):
  1. Plain SFT.
  2. + **DFT** loss.
  3. + SDFT/SynthSelf-style rewriting of the *free-text* parts only (reasoning/evidence wording), never the JSON values.
  4. RFT: sample the model's own correct outputs (on-policy-ish, "Retaining by Doing").
- Checkpoint gate: valid-JSON ≥ 99.5%, record-F1 on dev, forgetting suite (§6.3) within −2 points of base.

### 4.2 Stage 2: GRPO (RLVR)
- Start from the SFT adapter. LoRA r=16–64, LR 1e-6 to 5e-6, group size 8, temperature 0.8 during rollouts, max completion 1k tokens.
- Use the `dr_grpo` loss, or DAPO-style token-level loss + dynamic sampling (drop groups with all-equal rewards).
- KL β 0.01–0.04 (or 0 with a reference-KL monitor).
- Prompts: `hard` tier + held-out-format samples + 20% easy (stability).
- **Reward** (parser never raises; malformed JSON → 0):
  - `R = 0.6·slotF1 + 0.15·recordMatch + 0.1·evidenceGrounded + 0.1·nullCorrectness − 0.05·lengthPenalty`.
  - Format is a gate: valid JSON + schema keys, else R = 0.
  - **Negative-slot term:** a value predicted where GT is null is penalized (H4/H9).
  - **Distractor term:** predicting the planted decoy value costs an extra −0.2 (we know the decoys).
- Curriculum: by number of records and hard cases (R²GRPO).
- Monitor: reward, slot-F1 per H-case, **KL to base on held-out prompts** (RL's Razor), response length, entropy.

### 4.3 Stage 3: repair (only if forgetting exceeds budget)
1. LoRA scaling 0.5–0.8 (WiSE-FT). Pick the best point on the F1-vs-general curve.
2. TIES/DARE merge of the SFT and GRPO adapters (`peft.add_weighted_adapter`).
3. On-policy distillation (TRL `GKDTrainer`) from the **original base** on general prompts, plus from Qwen3-32B-fine-tuned on RE prompts (same tokenizer).

### 4.4 Optional
- Distill a Qwen3-32B extractor (stages 1–2) → 8B via GKD for better quality at the same serving cost.
- Inference cascade (OSLLM): 8B first; low-confidence records (logprob / self-consistency disagreement) escalate to the 32B or a frontier model.

## 5. Frameworks (efficient runs)

| Need | Primary | Fallback | Notes |
|---|---|---|---|
| Data generation (local) | **vLLM** 0.33 offline batch, xgrammar JSON | SGLang 0.5 | Prefix caching for shared prompts |
| Data generation (API) | `llm_providers` (OpenRouter/Bedrock/Azure/Gemini) + asyncio | bespoke `curator` 0.1.31 | Cache everything; pin providers |
| SFT | **TRL 1.17 `SFTTrainer` + PEFT 0.21 + Liger-Kernel** | LLaMA-Factory 0.9.5 / Axolotl 0.21 | DeepSpeed/FSDP2 if multi-GPU |
| Fast 1-GPU iteration | **Unsloth** 2026.11 (LoRA/QLoRA, GRPO) | — | Quick sweeps; confirm final runs in TRL |
| GRPO / RLVR | **TRL `GRPOTrainer`** + vLLM colocate (`gpu_memory_utilization` 0.3–0.5) | **verl** 0.9 (multi-GPU, scale), OpenRLHF | Dr.GRPO / DAPO options |
| Distillation | TRL `GKDTrainer` (same tokenizer); sequence-level KD for Llama→Qwen | tinker-cookbook recipes, DistillKit | |
| Merging | PEFT `add_weighted_adapter` (ties/dare/svd) | mergekit 0.1.5 (LGPL) | |
| Serving | **vLLM** multi-LoRA (`--enable-lora`) + JSON schema guided decoding | SGLang; llama.cpp/Ollama for laptop smoke tests | Avoid TGI (maintenance mode) |
| Eval | custom scorer + **lm-eval-harness** 0.4.13 (vLLM backend) | evalscope | |

Reusable local code in `D:\NLP\llm-fine-tune`: `sft_trainer.py`, `dpo_trainer.py`, `merge_peft_adapters.py`, `unsloth/unsloth_instruct_finetune.py`. Adapt these rather than rewrite.

**GPU budget (rough, unverified):**
- 8B LoRA SFT on 60k × ~3k tokens ≈ 6–10 h on 1×H100.
- 8B GRPO + LoRA (vLLM colocate) ≈ 1–2 days per 500 steps.
- 32B needs 2–4 GPUs (verl). 70B QLoRA fits on 1×80GB but is slow; only for the teacher.

**Gotchas:**
- Run on Linux/WSL2, not native Windows, for vLLM/TRL.
- Separate pinned venvs for generation (vLLM) and training (TRL).
- **Same chat template** in training, vLLM and reward parsing; Qwen3 thinking off (`enable_thinking=False`).
- Train with the same JSON format that constrained decoding enforces. Constrain keys, not values. Evaluate both constrained and unconstrained.

## 6. Evaluation

### 6.1 Scorer
- Records are aligned to GT by main-entity match (normalized string / plate / ID), using Hungarian matching.
- Metrics:
  - Slot-level P/R/F1 (exact and normalized).
  - Record-exact accuracy.
  - **Null accuracy** (H9).
  - **Decoy rate** (fraction of slots filled with the planted distractor).
  - JSON validity.
- Every metric is reported per language, format (text/HTML/email), hard case, and seen vs **held-out schema**.

### 6.2 Test sets
1. Synthetic test (in-distribution schemas).
2. Synthetic held-out schemas and formats.
3. **Gold real-text set:** 200–300 real he/fa documents (news, notices, emails), labeled by a frontier model and corrected by a human. This catches synthetic-text artifacts.
4. Public: NEMO (he NER), PERLEX test (fa RE), Re-DocRED test, MMUIE test, SciER (sanity check vs the R²GRPO/LA-RL numbers).

### 6.3 Forgetting suite (every checkpoint)
- IFEval, MMLU (or MMLU-Pro subset), GSM8K, Global-MMLU he/fa, MGSM (no he/fa → use translated subsets), plus HeQ / ParsiNLU subsets.
- A JSON-format general task (to catch the SDFT-style JSON failure).
- **Budget:** ≤ 2 points average drop vs base. Larger drops trigger §4.3.

## 7. Milestones

| Week | Deliverable |
|---|---|
| 1 | Schema library v0 (10), scorer, pilot 300×3 langs, P0 baseline report |
| 2–3 | `re_synth` v1: scrub/inject/render/verify for text + HTML + email; 10k samples; manual audit (100 per lang) |
| 3 | Public-data converters; adversarial mining round 1 |
| 4 | SFT v1 (Unsloth sweep → TRL final) + eval + forgetting suite |
| 5 | Mining rounds 2–3; data v2 (40k–60k); SFT v2 |
| 6–7 | GRPO v1 (reward ablations: + decoy term, + evidence term) |
| 8 | Repair/merge if needed; gold real-text eval; vLLM multi-LoRA serving; report and checkpoint |

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Injected text is unnatural, so the model learns artifacts | Inject into real articles (not reverse generation); judge filter; real-text gold eval; style diversity across injectors (2+ LLMs) |
| GT incomplete (leftover real entities of target types) | Scrub step + NER re-check + blind frontier extraction disagreement filter |
| Ambiguous guidelines (e.g., which date counts) | Frontier ≥ 90% on the pilot as the gate; explicit role definitions per slot in the schema |
| Overfitting narrow synthetic distribution | Schema/name paraphrase, held-out schema split, public-data mix, GRPO stage |
| Forgetting | LoRA all-layers, replay, KL monitor, forgetting suite, merge/GKD repair |
| Hebrew/Persian tokenizer inefficiency in Qwen | Measure tokens/word; A/B DictaLM-3.0 for Hebrew |
| API cost of mining | Cache, cheap models for injection, frontier only for verification/mining, batch APIs |
| License contamination | License column per source; NC datasets only for eval |
| Llama-3.3-70B provider variance | Pin provider + quantization in OpenRouter; record in metadata |

## 9. Open questions for the user
1. Exact main-entity types and slots of interest (beyond vehicle/person/org/shipment)? Real example documents (even 10) would sharpen the schemas and the gold eval.
2. Base model constraints: max size for serving, commercial-license needs (affects DictaLM/PersianMind/IEPile use)?
3. Output detail: verbatim strings only, or normalized dates/IDs too? Evidence spans required?
4. GPU availability (1× vs multi-GPU) and API budget for generation/mining.
5. Should one prompt cover multiple main-entity types (multi-schema) or one main type per call?
