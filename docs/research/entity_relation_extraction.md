# Fine-tuning LLMs for Entity & Relation Extraction — Research Survey

Compiled: 2026-11-10. Focus: 2024–2026 work on fine-tuning open LLMs for generic, prompt-driven entity and relation extraction. The prompt carries the schema, guidelines and few-shot examples; the model returns JSON. Typical target: a main entity plus its IDs, dates, locations, status and containing organization, in Hebrew, Persian and English.

**Sources**
- arXiv abstracts verified with `docs/research_tools/paper_research.js`.
- Semantic Scholar citation contexts for 12 core papers. The API was rate-limited (429), so most citation counts are missing.
- OpenAlex and Crossref for non-arXiv papers.
- GitHub and HF APIs, and ACL Anthology / CEUR / bioRxiv pages.
- Four parallel research threads: universal-IE LLMs, RL and synthetic data, forgetting, and he/fa resources.

**Verification tags**
- No tag: checked against the abstract, full text, or a live API.
- **[unverified]**: from background knowledge or a partial read. Re-check before relying on it.
- `n/a`: searched and not found.

**Companion plan:** [../plans/synth_dataset_and_finetune.md](../plans/synth_dataset_and_finetune.md)

---

## Table 1. Training techniques (what, evidence, gain, forgetting)

| Technique | What | Evidence in IE/RE | Gain | Forgetting | Use for us |
|---|---|---|---|---|---|
| **LoRA SFT** (all linear layers, r16–64) | Low-rank adapters | R²GRPO r16; pharmacovigilance Qwen3-8B; B2NER; IEPile; Extract-0 (0.53% params) | Large in-domain gains; JSON validity ~100% | Lower than full fine-tuning (2405.09673) | **Stage 1 default** |
| QLoRA | 4-bit base + LoRA | AutoRE | ≈ LoRA | ≈ LoRA | For 32B–70B on a single GPU |
| Full fine-tuning | All weights | KnowCoder, OneKE, UniversalNER | Highest on narrow tasks | Highest | Avoid |
| **MimicSFT** | SFT with a *templated* reasoning block, no gold chain-of-thought (CoT) | R²GRPO | Better RL start than plain SFT | — | Cheap warm-up before GRPO |
| CoT distillation SFT | Teacher writes reasoning traces | ReasoningNER, RexDrug, LA-RL (diagnostic traces) | +12 F1 vs GPT-4 zero-shot (ReasoningNER) | — | Optional, for hard distractor cases |
| **GRPO / RLVR** | Group-normalized policy gradient with rule rewards | R1-RE, R²GRPO, MR-UIE, ProUIE, LA-RL, Extract-0, NE-R1 | OOD +15 (R1-RE); +24.6 over SFT (R²GRPO); +6.8 SciER RE (LA-RL); +2.5 NER (NE-R1, modest) | **Lowest** (RL's Razor, R1-RE) | **Stage 2** |
| Dr.GRPO / DAPO / GSPO | GRPO fixes: no length/std bias, clip-higher, dynamic sampling, sequence-level ratios | general RL **[unverified IDs]** | Stabilizes short structured outputs | — | Try Dr.GRPO loss + DAPO dynamic sampling |
| DPO | Preference pairs | ADELIE (SFT→DPO) | Modest; no general-ability drop | Low | Option: pairs = (correct vs distractor-fooled) |
| RLHF (reward model) | Learned reward | AOCG-LLM+ (entity precision +10.1) | + | — | Not needed; we have verifiable gold |
| Targeted distillation | SFT on teacher LLM labels over real text | UniversalNER (+7–9 over ChatGPT), Pile-NER, DocIE@XLLM25 | Large | — | Labels for un-injected text spans |
| Reverse generation | Facts first, then text | SynthIE (+57 micro-F1) | Large | — | **Core of our synthetic pipeline** |
| **Self-distillation (SDFT)** | Model rewrites gold targets in its own style | SDFT 2402.13669; SynthSelf 2610.13113 | ≈ SFT | Much lower | Rewrite the "reasoning/explanation" part of targets; keep JSON gold fixed |
| On-policy distillation (GKD / reverse KL) | Student samples scored by teacher log-probs | GKD 2306.13649; Thinking Machines | ≈ RL at ~1/10 compute | Low; restores IFEval | Repair stage; Qwen3-32B → 8B distillation |
| Semi-supervised + LLM augmentation | LLM augments unlabeled text, consistency + pseudo-labels, iterative PEFT of the augmenter | TKDD 2026 low-resource joint ERE | SOTA in low-resource settings | — | Idea for real he/fa text without gold |
| Replay / data mixing | Mix general instruction and chat data | YAYI-UIE (chat data helps), ADELIE | Neutral-to-positive | Lower | 20–50% replay incl. he/fa chat |
| Model merging | WiSE-FT, TIES, DARE | general | Slight target loss | Lower | Final safety net |
| **Teacher-CoT distillation to a classifier + confirmation model** | Teacher LLM writes CoT over training examples to supervise a small classifier plus a separate false-positive filter | Distill-RE (AKBC 2026): +31.5 micro-F1 avg over TACRED/TACREV/Re-TACRED @ 8/16/32-shot | Large, fixes LLM over-prediction bias | n/a (student is a small PLM, not the base chat model) | Two-head idea for our verifier: a classify head + a "is this slot/decoy correct" confirm head |
| **Non-generative decision-model distillation (typed yes/no/choice/score)** | Distill a teacher (closed LLM or our own model) into a LoRA/encoder head that reads logits once per question instead of generating text | OpenJev / BioJev (community, 2026, unreviewed) | Beats the closed API it was trained to imitate, on the task it was trained on; zero-shot is weak | Very low (base LM untouched or encoder-only) | Candidate **cheap verifier**: one forward pass per slot/decoy ("is `<date>` the correct date for `<entity>`?") instead of full re-generation |

---

## Table 2. Datasets

### A. Public English IE/RE training & eval sets

| Dataset | Task | Size | Lang | How it was built (curation) | License | Link | Notes for us |
|---|---|---|---|---|---|---|---|
| **Pile-NER-type** (UniversalNER) | open NER | ~45k passages, ~240k mentions, ~13k types **[unverified]** | en | ChatGPT annotates all entity types in Pile passages; conversation format with negative types | see card | [HF](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) | Largest open-type vocabulary. Good for type-name diversity and the "generic schema" skill. |
| **IEPile** | NER, RE, EE | 0.32B tokens from 33 datasets | en, zh | Cleans 33 existing datasets, then builds schema-based instructions with **hard-negative schemas** and batched schemas | NOASSERTION / CC BY-NC-SA **[unverified]** | [HF](https://huggingface.co/datasets/zjunlp/iepile), [GH](https://github.com/zjunlp/IEPile) | Template for our instruction format and negative sampling. License may be non-commercial. |
| **B2NERD** | open NER | ~50 datasets unified into 400+ types, 6 langs **[unverified]** | multi | Merges datasets, resolves label conflicts into a universal taxonomy, prunes redundant samples | see card | [HF](https://huggingface.co/datasets/Umean/B2NERD) | Shows how to merge label spaces cleanly. |
| **KnowCoder schema data** | NER, RE, EE | >30k types; ~1.5B auto-built examples | en | Wikidata/UIE schemas turned into Python classes; auto-built understanding and following data | NOASSERTION | [Library](https://huggingface.co/datasets/golaxy/KnowCoder-Schema-Library), [Following](https://huggingface.co/datasets/golaxy/KnowCoder-Schema-Following-Data) | Code-style schemas; a large relation-type vocabulary to sample from. |
| **GuideX pretraining data** | zero-shot IE | see paper | en | LLM invents a domain schema, writes guidelines, then labels synthetic text | see card | [HF](https://huggingface.co/datasets/HiTZ/GuideX_pre-training_data) | Guidelines in the prompt, which is our prompt design. |
| **SynthIE** | closed IE (Wikidata triples) | 1.8M | en | Samples Wikidata subgraphs, then an LLM writes text **from** the triples (reverse generation), then filters | MIT (repo) | [HF](https://huggingface.co/datasets/martinjosifoski/SynthIE), [GH](https://github.com/epfl-dlab/SynthIE) | Proves "facts first, then text" gives clean gold. Our pipeline injects the facts into real articles instead. |
| **Extract-0 data** | document IE → JSON | 280,128 | en | Memory-preserving synthetic tasks over document chunks | see card | [HF](https://huggingface.co/datasets/HenriqueGodoy/extract-0) | Generic JSON extraction tasks. |
| **ScrapeGraphAI-100k** | schema-constrained JSON from web pages | 93,695 events, 18k+ schemas, 15 langs | multi | Opt-in telemetry, deduplicated from 9M events; schema-conformance labels only, semantic correctness not checked | see card | [HF](https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k) | Real web prompts and schemas. Labels are not gold. |
| **DocRED** | doc-level RE (96 relations) | 5k annotated + 100k distant | en | Wikipedia + Wikidata distant supervision + crowd annotation | MIT | [GH](https://github.com/thunlp/DocRED) | Many false negatives. Prefer Re-DocRED. |
| **Re-DocRED** | doc-level RE | 3,053 / 500 / 500 docs | en | Re-annotates DocRED to fix false negatives | MIT | [HF](https://huggingface.co/datasets/tonytan48/Re-DocRED) | Main real-text doc-level eval. |
| **Re-DocRED-CF** (CovEReD) | counterfactual DocRE | train 2,870 / dev 466 / test 453; `train_mix` 5,923 | en | Consistently **replaces entities** in Re-DocRED docs to break memorized facts | MIT | [HF](https://huggingface.co/datasets/amodaresi/Re-DocRED-CF), [GH](https://github.com/amodaresi/CovEReD) | **Direct template for our entity-swap and injection step.** |
| **MMUIE** | doc-level UIE (NER + RE) | 34 domains, 723 entity types, 456 relation types | en | KB + LLM automated annotation of long docs; weakly supervised train, **manually verified test** | see card | [HF](https://huggingface.co/datasets/shuyi-zsy/MMUIE) | Newest (EACL 2026) long-document, many-type set. |
| **REBEL** | sentence RE | ~3M sentences, ~900 relations **[unverified]** | en | Wikipedia abstracts + Wikidata + NLI entailment filter | CC-BY-SA-4.0 | [HF](https://huggingface.co/datasets/Babelscape/rebel-dataset) | Noisy. Use for pretraining-scale data or negatives. |
| **SciER** | sci NER + RE | 106 papers, 24k entities, 12k relations | en | Expert annotation of full scientific papers | GPL-3.0 (repo) | [GH](https://github.com/edzq/SciER) | Used by R²GRPO and LA-RL; the RL reference benchmark. |
| **SemEval-2010 T8** | relation classification | 8k / 2.7k | en | Manual | research | — | Used by R1-RE. Possibly contaminated in frontier models. |
| **CoNLL04, NYT, TACRED/Re-TACRED** | joint / sentence RE | small to medium | en | Manual / distant / LDC | mixed | — | Classic benchmarks. TACRED needs an LDC license. |
| **BC5CDR, ADE, DDI** | bio NER + RE | standard | en | Manual | research | — | Used by the pharmacovigilance LoRA paper. |
| **GutBrainIE (CLEF 2025)** | bio NER + RE | Platinum/Gold/Silver/Bronze tiers | en | Expert, in-house and student annotation tiers | CC-BY | [CLEF](https://ceur-ws.org/Vol-4038/paper_24.pdf) | Example of mixing training data by label-quality tier. |

### B. Hebrew / Persian / multilingual resources

| Dataset | Lang | Task | Size | Curation | License | Link | Notes |
|---|---|---|---|---|---|---|---|
| **Hebrew RE** | he | RE | **none exists** | — | — | — | **Gap.** Synthetic data is required. |
| **NEMO corpus** (+BMC) | he | NER (nested, morphology-aware) | ~6k sentences **[unverified]** | Manual annotation of the Hebrew treebank | not stated | [GH](https://github.com/OnlpLab/NEMO-Corpus) | Main Hebrew NER eval. |
| **PERLEX** | fa / en | RE (SemEval-2010 T8 relations) | ~10k sentences | Expert translation of SemEval-2010 T8 | unspecified | [HF](https://huggingface.co/datasets/Aeirya/PERLEX) | **Only Persian RE set.** Sentence-level, coarse relations. BERT-EM-MTB 77.66 F1. |
| **PEYMA** | fa | NER | ~10k sentences | Manual, news | unverified | [HF](https://huggingface.co/datasets/ParsiAI/PEYMA) | Standard Persian NER. |
| **ArmanPersoNER** | fa | NER | ~250k tokens | Manual | unverified | [HF](https://huggingface.co/datasets/hezarai/arman-ner) | |
| **MultiCoNER v2** | 12 langs incl. **fa** (no he) | fine-grained NER, 33 classes | large | Templates + real queries | CC-BY-4.0 | [HF](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) | Persian fine-grained types. |
| **WikiANN** | he, fa (+176) | NER PER/ORG/LOC | large | Silver, from Wikipedia links | unknown | [HF](https://huggingface.co/datasets/unimelb-nlp/wikiann) | Noisy, but covers both languages. |
| **SREDFM / MultiTACRED / UNER** | multi, **no he/fa** | RE / NER | large | Silver / MT / manual | various | [SREDFM](https://huggingface.co/datasets/Babelscape/SREDFM) | Recipes only, not usable data for he/fa. |
| **FarsInstruct** | fa | instructions | — | — | unverified | [HF](https://huggingface.co/datasets/ParsiAI/FarsInstruct) | Persian replay data. |

### C. Source corpora for synthetic injection (he/fa)

| Corpus | Lang | Format | License | Link | Use |
|---|---|---|---|---|---|
| Wikipedia 20231101.he / .fa | he, fa | text | CC-BY-SA-3.0 | [HF wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Clean seed articles. |
| Structured Wikipedia | he/fa **[check configs]** | JSON (infoboxes, tables, sections) | CC-BY-SA-4.0 | [HF](https://huggingface.co/datasets/wikimedia/structured-wikipedia) | Source of tables for HTML samples. |
| FineWeb-2 `heb_Hebr`, `fas_Arab` | he, fa | web text | ODC-By | [HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Diverse genres (news, forums, ads). |
| HPLT 2.0 cleaned | he, fa | web text | CC0 | [HF](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | Most permissive license. |
| HeDC4 | he | web text | none on card | [HF](https://huggingface.co/datasets/HeNLP/HeDC4) | Hebrew web. |
| naab | fa | text | MIT | [HF](https://huggingface.co/datasets/SLPL/naab) | Large Persian corpus. |
| HeSum | he | news summaries | see card | [HF](https://huggingface.co/datasets/biunlp/HeSum) | Hebrew news-like articles. |
| Hebrew news corpus / he-fa HTML-with-tables | — | — | — | **none found** | Build from Wikipedia HTML dumps or Common Crawl. |

## Table 3. How to synthesize / curate IE-RE data (methods from the papers → our pipeline)

| # | Method | Papers | How it works | Gold quality | Evidence of gain | Use in our he/fa pipeline |
|---|---|---|---|---|---|---|
| S1 | **Scrub → inject → GT by construction** (ours) | builds on SynthIE + CovEReD | Sample schema + records → LLM removes original facts of the target types from a real article → LLM rewrites to inject synthetic records → GT known; verify spans | **Complete (closed-world)** | n/a (to be measured) | **Core pipeline** for plain text, HTML, email |
| S2 | Reverse generation (facts → text) | SynthIE | Sample Wikidata subgraph → LLM writes text expressing exactly those triples → filter | High; text can be unnatural | +57 micro-F1 over prior on REBEL-style eval | Fallback when no seed article fits; short docs |
| S3 | Counterfactual entity swap | CovEReD (Re-DocRED-CF) | Consistently replace entities in an annotated doc, so memorized world facts no longer hold | Inherits source gold | Exposes / reduces reliance on parametric knowledge | Swap names/IDs/places in injected records; anti-memorization |
| S4 | LLM labeling of real text (targeted distillation) | UniversalNER (Pile-NER), DocIE@XLLM25 | Teacher LLM annotates all entities/types in real passages; negative types sampled | Silver (teacher errors) | UniNER +7–9 F1 over ChatGPT | Label leftover real entities in he/fa articles so they are not false negatives |
| S5 | KB alignment + LLM extraction | MMUIE, DocRED, REBEL (+NLI filter) | Link mentions to a KB (Wikidata) → candidate triples → LLM/NLI verifies; humans verify test only | Weak train / verified test | DocUIE beats sentence-level UIE on long docs | Cheap silver labels for real he/fa Wikipedia; real-text eval subset |
| S6 | Schema + guideline invention | GuideX | LLM induces domain schema, writes guidelines, then labels text | Silver | up to +7 F1 zero-shot | Auto-generate new main-entity types / slot definitions for prompt diversity |
| S7 | Dataset unification + hard-negative schemas | IEPile, B2NERD, KnowCoder | Merge many datasets, resolve label conflicts, add absent types to the schema, prune for diversity | Inherits source | B2NER +6.8–12 F1 over GPT-4 | English replay mix; negative slots (type asked but absent → null) |
| S8 | Parallel multilingual instances | KnowCoder-X (ParallelNER) | Same instance/schema translated, aligned, then manually checked | High after checks | +20 over SOTA on 29 unseen langs | Produce each record in en + he + fa with the same schema |
| S9 | Synthetic task generation (text-to-structure) | OmniStruct, Extract-0 | LLM writes diverse extraction tasks + JSON targets over document chunks | Silver | Small models rival GPT-4o / beat GPT-4.1 on own bench | Template diversity (field names, nesting) |
| S10 | Self-supervised extraction from raw text | Cuckoo (NTE) | Turn next-token prediction into "extract a span already in context" | Exact (by construction) | Better few-shot IE than prior pretrained IE | Cheap pre-warm on he/fa web text (optional) |
| S11 | Semi-supervised: LLM augment + pseudo-labels | TKDD 2026 low-resource ERE | LLM augments unlabeled text; consistency training + pseudo-labels; iterative PEFT | Silver, iteratively improved | SOTA on 4 low-resource benchmarks | Self-training on real unlabeled he/fa news |
| S12 | Reasoning-trace generation | MimicSFT (R²GRPO), ReasoningNER, RexDrug, LA-RL | Template or teacher-written rationales / error diagnoses before JSON | Format-only (MimicSFT) or teacher | MimicSFT→GRPO +24.6 Rel F1 over SFT | Short rationale for distractor cases; GRPO warm-up |
| S13 | Web/HTML structured data | ScrapeGraphAI-100k, ReaderLM-v2, SCRIBES | Real web pages + schemas; HTML→markdown/JSON training | Schema-valid only | ReaderLM-v2 beats larger LLMs on HTML→MD/JSON | Render injected records into HTML tables, lists under subtitles |
| S14 | Distractor injection (hard negatives) | user hard cases H1–H12; CovEReD; IEPile negatives | Insert near-by wrong dates/locs/orgs, date bleeding, far email headers, subtitle→list | Complete (we control them) | n/a (to be measured) | Every sample gets 0–3 distractor families, tagged for per-case F1 |
| S15 | Adversarial mining (weak-fails / strong-succeeds) | ours (Llama-3.3-70B vs frontier) | Run both on generated samples; keep where open model errs and frontier matches GT | Complete | n/a | Difficulty tiers; up-weight in SFT and RL |

---

## 0. TL;DR for our stack

1. **Base model:** Qwen3-8B (or Qwen3-14B) Instruct with LoRA. Qwen3-8B LoRA cut JSON parse failures from 23.5% to 0.11% and reached 89.4 entity F1 / 62.3 relation F1 in the pharmacovigilance study. A fine-tuned 3B beat a zero-shot 8B there. DictaLM-3.0 (Apache-2.0) is the Hebrew-specialist alternative to A/B test.
   - **Reasoning/thinking is a first-class A/B, not off by default.** User's own experience plus Distill-RE (teacher-CoT-distilled students: +31.5 micro-F1) and ReasoningNER (+12 F1 vs GPT-4 zero-shot) indicate CoT helps RE/RC directly. A/B no-reasoning vs templated-reasoning (MimicSFT) vs native `enable_thinking=True` at Stage 1 SFT, gated on record-F1 **and** the forgetting suite (longer completions can raise KL/cost).
   - **Model-scale ladder** (4×A100 80GB in budget, 320GB total): if Qwen3-8B (either thinking mode) underperforms, escalate **Qwen3.8-27B** (`qwen/qwen3.8-27b`, OpenRouter zero-shot first) → Qwen3-32B (full SFT or LoRA both fit directly, no QLoRA needed) → Gemma-3-27B (different pretraining, useful if the Qwen family plateaus) → Llama-3.3-70B QLoRA as a reference ceiling. Reuse the same data/prompt/scorer at each rung; stop once the gap to frontier closes.
2. **Prompt = schema + guidelines + few-shot, output = JSON.**
   - Guidelines are the key ingredient: GoLLIE ablation, GuideX up to +7 F1, R1-RE's annotation guide.
   - Keep the same format in training, serving and the reward parser.
3. **Data:**
   - Nothing public covers Hebrew RE and Persian RE is tiny, so synthetic "scrub → inject → GT by construction" data is required.
   - Mix in public English IE (IEPile, Pile-NER, B2NERD, GuideX, Re-DocRED + CF) for schema diversity.
   - Add **hard negatives** in two forms: IEPile-style negative schemas and CovEReD-style entity swaps, plus the distractor families in §3.
4. **Training:**
   - **Stage 1 SFT:** short, LoRA r32–64 on all linear layers, 20–50% general/he/fa replay, targets rewritten in the model's own style (SDFT).
   - **Stage 2 GRPO** with a verifiable reward: F1 + span + format + grounding (R²GRPO).
   - SFT→RL gave +24.6 Rel F1 over SFT on SciER (R²GRPO).
   - RL kept OOD +15 points and preserved MATH/IFEval/GPQA (R1-RE).
5. **Forgetting control:**
   - Track KL to the base model; per RL's Razor, KL predicts forgetting.
   - Run a forgetting suite (IFEval, MMLU, GSM8K, Global-MMLU/MGSM he+fa) at each checkpoint.
   - Fallbacks: merge-back (LoRA scale 0.5–0.8, TIES/DARE), then on-policy distillation from the original model.
6. **Adversarial mining:** keep samples where Llama-3.3-70B fails and frontier models succeed (via `llm_providers`/OpenRouter) and up-weight them in SFT and RL.
7. **Frameworks:**
   - Train with TRL (SFT/GRPO/GKD) + PEFT + Liger; Unsloth for 1-GPU iteration; verl if multi-GPU RL is needed.
   - Serve and generate with vLLM (multi-LoRA, xgrammar JSON); evaluate with lm-eval-harness.
   - Details in the plan.
8. **Caveat:** RL gains on RE can be benchmark artifacts. ConsistRE showed a DocRED RL gain vanishes under Re-DocRED. Use **closed-world gold** (our synthetic data has it) for rewards and evals.

---

## 1. Concepts

| Concept | What it means | Why it matters for us |
|---|---|---|
| Universal IE (UIE) | One model for NER, RE and EE, driven by a schema in the prompt | Our "generic extractor" goal. |
| Schema/guideline-conditioned IE | Prompt includes type definitions, edge-case rules and examples (GoLLIE, GuideX, R1-RE) | Lets users add new entity and relation types without retraining. |
| Closed vs open IE | Fixed relation inventory vs free-form relations | Ours is closed per prompt: the schema is given. |
| Doc-level RE | Relations across sentences, coreference, long range | The "car and its location two sentences away" case. |
| n-ary / record extraction | Main entity + many slots (ID, date, location, status, org) | Our output is a record per main entity, not just triples (RexDrug, Extract-0). |
| Hard negatives | Plausible but wrong candidates: types absent from the text, decoy entities | Main lever against false positives. |
| Counterfactual augmentation | Swap entities so world knowledge can't be used | Forces grounding in the text (CovEReD). |
| RLVR | RL with verifiable rewards (rule-based F1, format) | Extraction F1 against gold is the ideal verifiable reward. |
| Catastrophic forgetting | Loss of general and instruction-following skill after fine-tuning | We need the model to stay a usable chat/instruct model. |

## 2. Fine-tuning techniques

See **Table 1** at the top of this document; Table 3 maps data-synthesis methods to our pipeline.

## 2b. Anti-forgetting recipe (synthesized; details in §6e)

1. **LoRA on all linear layers.** Rank 32–64 for SFT, rank 8–64 for RL. Use an LR about 10× the full-FT LR (~1e-4 for SFT, ~1e-6 for GRPO as in R²GRPO). Low rank and attention-only adapters create intruder dimensions that cause more forgetting (2410.21228).
2. **Make targets in-distribution.**
   - Keep JSON gold exact.
   - If the output has a short rationale, let the base model (or a larger same-family model, per SynthSelf) rewrite it, and keep only rewrites whose parsed JSON F1 equals 1.
3. **Replay 20–50% general data:** he/fa/en chat (FarsInstruct, the base model's own answers on Hebrew prompts), IFEval-style instructions, some math.
4. **Short SFT** (1–2 epochs), then **GRPO**. RL is biased toward KL-minimal solutions (RL's Razor, 220 citations). Later work agrees that on-policy data is why RL forgets less ("Denser ≠ Better", 2026; "Retaining by Doing", 2025).
5. **Monitor** KL to base plus a forgetting suite at every checkpoint.
6. **Fallbacks:** if general benchmarks drop more than ~2 points, merge back (LoRA scale 0.5–0.8, or TIES/DARE). Then run on-policy distillation from the original model on general prompts.
7. **Pitfall: SDFT and JSON.** SelfAug (2025) reports SDFT "struggles with format-specific tasks, particularly when structured outputs like JSON are required." So never let the self-rewrite touch the JSON values; rewrite only free text.

## 3. Edge cases / hard cases (from user requirements and literature)

| # | Hard case | Example | Failure mode | Literature hook | How we generate it |
|---|---|---|---|---|---|
| H1 | **Near-by false-positive entity of the right type** | "Black Mazda is parked in Yehud. It was invented in 1/1/1955. The owners will pick it tomorrow on 12/2/26" → date=12/2/26 | Picks the closest date or org | IEPile hard negatives; LA-RL "wrong label" errors | Inject 1–3 same-type decoys at distance ≤ target distance, with a non-schema role (manufacture date, birthplace, prior owner) |
| H2 | **Distractor between entity and its slot** | Car … *Haifa (where the owner works)* … "it is kept in Ashdod" | Nearest-location heuristic | Doc-level RE (Re-DocRED); "lost in the middle" | Place the target slot 2–3 sentences away with a decoy location in between |
| H3 | **Email/form headers far from body** | Subject line holds the plate number; 15 irrelevant header fields, then the body says "the vehicle" | Misses info in the title or header; coreference across a long gap | Doc-level/long-context IE (MMUIE) | Render an email template with long headers; the title carries ID or status; the body uses pronouns |
| H4 | **Attribute bleed between neighbors** | "Truck A arrived on 3/4. Truck B was also seen." → B has no date | Copies A's date to B | LA-RL error taxonomy; ConsistRE constraints | Two same-type main entities; only one has the slot; GT has null for the other |
| H5 | **Irrelevant date tied to the entity** | Quoted-thread dates, "message from 1/2/24" near the entity | Takes the metadata date as the event date | — | Forwarded-message blocks and timestamps near the entity |
| H6 | **Subtitle → list/table a few lines below** | "The following cars belong to Hertz: <3 irrelevant lines> Subaru 45GBF4 / Green Mazda 67U990" | Misses the containing-org relation over the gap | HTML/table IE (ReaderLM-v2, SCRIBES) | Header sentence + filler + list or `<table>`; every row gets org=Hertz |
| H7 | HTML tables with sibling rows | Several cars per table, columns for status and location | Row/column misalignment | ScrapeGraphAI, ReaderLM-v2 | Render injected records as `<table>` with distractor columns |
| H8 | Counterfactual / non-memorizable facts | Fictional plate numbers, real cities in odd combinations | Model "knows" an answer from pretraining | CovEReD | All injected values are sampled, never real |
| H9 | Negative schema types | Schema asks for `status`, but the text has none | Hallucinated value | IEPile negative schemas; pharmacovigilance paper calls for negative supervision | 20–30% of slots absent → GT null |
| H10 | Multilingual / RTL / mixed script | Hebrew text with a Latin plate number and a Gregorian vs Hebrew/Persian calendar date | Normalization errors | KnowCoder-X (cross-lingual) | Locale-specific formats (Jalali dates, Hebrew dates, RTL punctuation) |
| H11 | Coreference / aliases | "the vehicle", "it", "הרכב", "خودرو" | Misses the relation | MMUIE (aliases) | Replace later mentions with pronouns or descriptions |
| H12 | Status negation / change over time | "was stolen, later recovered" | Takes the first status mentioned | — | Inject status sequences; GT = final or as-of-date status per guideline |

## 4. Benchmarks & reported results (headline numbers)

| Paper | Benchmark | Base model | Method | Headline | Did it really improve? |
|---|---|---|---|---|---|
| R1-RE | SemEval-2010 ↔ MDKG (OOD) | Qwen2.5-7B-Instruct | GRPO from instruct | OOD 66.5/65.8 vs SFT 51.3/49.8; ≈ GPT-4o | Yes OOD; in-domain ≈ SFT. Classification only. |
| R²GRPO | SciER (+OOD) | Qwen2.5-7B | MimicSFT → GRPO (F1 + Jaccard + relevance + rule rewards) | Rel 74.38 / OOD 66.74; +24.59 over SFT, +17.97 over GRPO-only | Yes; beats SciBERT supervised. Row-level baselines **[unverified]**. |
| LA-RL | SciER RE, DuEE | n/a | cold-start SFT → 2× GRPO with diagnose/revise | +6.83 F1 SciER RE; ~+20 OOD | Yes vs SFT, same backbone |
| NE-R1 | NER benches | n/a | instruction tuning → RL + retrieval | +2.52 in-domain, +1.18 zero-shot | Modest |
| ConsistRE | DocRED vs Re-DocRED, DWIE | LLM | distillation + consistency RL | RL +0.025 on DocRED **vanishes** on Re-DocRED | No: artifact of incomplete gold |
| Extract-0 | 1,000 held-out doc-IE tasks | 7B | LoRA SFT → GRPO (semantic reward) | mean reward 0.573 vs GPT-4.1 0.457, o3 0.464 | On its own benchmark only |
| UniversalNER | 43 NER datasets | LLaMA 7B/13B | ChatGPT distillation | +7–9 F1 over ChatGPT; +30 over Vicuna | Yes; widely reused baseline |
| KnowCoder | UIE suites | LLaMA2-7B | code schema pretraining + IT | +49.8% few-shot vs LLaMA2; +12.5% zero-shot | Yes; later cited as the code-schema reference |
| GuideX | 7 zero-shot NER | Llama-3.1-8B | synthetic schema + guidelines | up to +7 F1 | Yes zero-shot; ~+2 when added to human data |
| SynthIE | REBEL / Wiki-cIE | Flan-T5 | 1.8M reverse-generated examples | +57 micro-F1 / +79 macro-F1 | Yes; later criticized for being confined to encyclopedic prose (StructText 2025) and domain drift (SNaRe 2025) |
| Pharmacovigilance LoRA | BC5CDR, ADE, DDI | Qwen3-8B, Llama-3.2-3B | multitask LoRA SFT, JSON | 89.42 entity / 62.32 relation F1; parse failures 23.5%→0.11% | Yes vs zero-shot; RE ≪ NER |
| AOCG-LLM+ | ontology construction | LLMs | RLHF extraction module | entity precision 92.5 (+10.1); relation F1 88.3 (+13 vs BERT+GCN) | On its own setup; 0 citations |
| ReaderLM-v2 | HTML → MD/JSON | 1.5B | synthetic draft/refine/critique | +15–20% over GPT-4o | On curated benchmarks |
| GutBrainIE (BIU) | CLEF 2025 | GLiNER + ATLOP | encoder pipeline | binary RE 0.612 (5/11); NER 0.48 (15th); mention RE 0.18 | Shows encoder pipelines are weak on mention-level RE |

Takeaways:
- RL beats SFT mainly **out of domain**; in domain the two are close.
- The SFT→RL order is consistently best.
- The gains depend on **clean, closed gold**.
- Small fine-tuned models (3–8B) match or beat frontier zero-shot on their target schema family, but published OOD evidence is limited to a few papers.

---

## 6. Papers

Format: `) Goal: TLDR`, `) Problem`, `) Tags`, `) FT`, `) Data curation`, `) Prompt/output`, `) Results`, `) Later papers say`, `) Links (Paper / Code (stars, license) / HF model / HF dataset)`, `) Relevance`.
User-priority papers are marked ⭐.

### 6a. RL / RLVR for IE & RE (newest, most relevant to stage 2)

* ⭐ **Beyond path selection: Better LLMs for Scientific IE with MimicSFT and Relevance and Rule-induced (R²)GRPO** (2505.22068, May 2025)
  ) Goal: TLDR - scientific text --> MimicSFT (a templated `<reasoning>` block, then JSON) --> R²GRPO with a composite verifiable reward --> entities + relation triples
  ) Problem: LLMs trail SciBERT-style supervised models on SciIE. The paper also tests the claim that RLVR only "selects paths" instead of adding capacity.
  ) Tags: GRPO, RLVR, LoRA, end-to-end NER+RE, curriculum, pass@k
  ) FT: Qwen2.5-7B-Instruct, LoRA on all linear layers.
    - MimicSFT: r16 / α32, 3 epochs, lr 2e-5, batch 32.
    - R²GRPO: LoRA r64 / α128, lr 1e-6, group-normalized advantage + KL.
    - Reward = w1·F1 + w2·Jaccard span overlap + w3·relevancy (cited evidence must appear in the input, with a length penalty) + w4·rule-pattern reward.
    - Curriculum by entity/triple count. Training examples are selected where SFT fails but the reward is informative.
  ) Data curation: SciER, 106 papers, 24k entities and 12k relations, plus an OOD split. Multi-task SFT over sub-tasks (NER only, RE with gold entities, end-to-end). **No gold chain-of-thought (CoT) is needed:** the reasoning template is fixed ("1. identify entities 2. consider relations 3. formulate").
  ) Prompt/output: `<reasoning>…</reasoning>` + JSON. Best at temperature < 0.6; results reported at T=0.
  ) Results:
    - SciER: NER 88.90 / Rel 74.38. OOD: 81.27 / 66.74.
    - +24.59 Rel over SFT and +17.97 over GRPO-only. Beats SciBERT supervised.
    - Frontier zero-shot models (Gemini-2.0-flash, DeepSeek-V3/R1) are far lower.
    - Best@K: RL has higher Avg@K and Best@1, and both RL and SFT beat the base at every K. This contradicts "RL only narrows paths".
    - The baseline table was scraped without row labels, so exact baseline rows are **[unverified]**.
  ) Later papers say: two citers (S2). One is LA-RL (2607.23420), which builds a 2-stage GRPO on the same SciER benchmark and reports a further +6.83 F1 over SFT.
  ) Links: Paper https://arxiv.org/abs/2505.22068 / Code https://github.com/rlisml/R2GRPO (= ranlislz/R2GRPO, 3★, no license) / HF model: n/a / HF dataset: n/a (SciER: https://github.com/edzq/SciER)
  ) Relevance: **The closest recipe to ours.** Generic JSON for entities and relations, 7B LoRA, SFT then GRPO, reusable reward components (F1 + span + grounding).

* ⭐ **R1-RE: Cross-Domain Relation Extraction with RLVR** (2507.04642, Jul 2025)
  ) Goal: TLDR - sentence with marked entities + human **annotation guide** in the prompt --> GRPO from the instruct model with a format+accuracy reward and free `<think>` --> `<answer>rel(e1,e2)</answer>`
  ) Problem: SFT RE models generalize poorly out of domain. The paper reframes RE as reasoning over annotation guidelines, the way human annotators work.
  ) Tags: RLVR, GRPO, OOD, guidelines, forgetting
  ) FT: Qwen2.5-7B-Instruct (also 3B), GRPO with no SFT warm-up. The reward is a minimal regex format gate, then ±1 accuracy.
  ) Data curation: SemEval-2010 T8 (public) and MDKG (private mental-disease KG). Each comes with a guide that gives definitions and edge cases per relation. Train on one, test OOD on the other. No synthetic data.
  ) Results:
    - In domain ≈ SFT: 88.1 / 91.8 vs SFT 90.4 / 92.4.
    - **OOD 66.5 / 65.8 vs SFT 51.3 / 49.8**, on par with GPT-4o (65.9) and below Claude-3.5-Sonnet (71.1).
    - **Forgetting:** RL kept or raised MATH-500 75.4, IFEval 73.0 and GPQA 29.6, while SFT lowered them (69.8 / 71.9 / 27.2).
    - The annotation guide helps SFT as well (~+10% OOD).
    - Response length grows from 200 to 500–1000 tokens.
  ) Later papers say: cited by Parallel-R1 (2025, 58 citations) among RL-over-SFT works that generalize better. Its main point is that "SFT … suffers from several key limitations" relative to RL.
  ) Links: Paper https://arxiv.org/abs/2507.04642 / Code https://github.com/RunpengDai/R1_RE (2★, no license) / HF model: n/a / HF dataset: n/a (MDKG private)
  ) Relevance: Evidence that GRPO gives OOD robustness without forgetting, and that guidelines belong in the prompt. Limitation: it does relation classification on given entities, not full extraction.

* **LA-RL: Label-Aware Self-Reflection for RL in IE** (2607.23420, Jul 2026)
  ) Goal: TLDR - text --> extract --> diagnose with typed labels (missing span, wrong label, boundary, invalid relation, reversed arguments) --> revise --> two GRPO stages
  ) Problem: free-form self-reflection doesn't line up with structured extraction errors.
  ) Tags: GRPO, self-correction, error taxonomy
  ) FT: cold-start SFT on diagnostic data labeled by an annotator LLM, then 2× GRPO. Rewards: final quality + format validity + first-pass correctness. No process reward model.
  ) Data: SciER (RE), DuEE1.0 (EE), plus NER sets. Diagnostic traces are teacher-generated.
  ) Results: +6.83 avg F1 on SciER RE over SFT, ~+20 F1 OOD RE, +14.8 / +17.5 trigger/argument F1 on DuEE. Stronger constraints help RE, while NER prefers looser ones under shift.
  ) Links: Paper https://arxiv.org/abs/2607.23420 / Code n/a / HF n/a
  ) Relevance: The error taxonomy matches our hard cases. Each hard case (H1/H4 = wrong value, H2 = nearest-location mistake, H6 = missing org) can be a typed diagnostic label.

* **MR-UIE: Multi-Perspective Reasoning with RL for UIE** (2509.09082, Sep 2025)
  ) Goal: TLDR - complex schema prompt --> RL with multi-perspective reasoning --> structured NER/RE/EE
  ) Problem: complex schema descriptions need multi-step reasoning, and ICL or instruction tuning alone falls short.
  ) Tags: RL, UIE, reasoning
  ) FT: RL on an open LLM. Reward, base model and KL settings **[unverified]**; only the abstract was read.
  ) Results: improves accuracy across domains and is SOTA on several datasets. Reasoning in RL "notably enhances generalization in complex IE".
  ) Later papers say: cited as an RL-UIE reference. MR-UIE itself credits GoLLIE's guidelines as "structured priors enabling robust zero-shot generalization".
  ) Links: Paper https://arxiv.org/abs/2509.09082 / Code n/a / HF n/a
  ) Relevance: Shows generic-schema RL works beyond SciER.

* **ProUIE: Macro-to-Micro Progressive Learning for LLM-based UIE** (2604.10633, Apr 2026)
  ) Goal: TLDR - 36 IE datasets --> (1) full-data complete modeling, (2) sampled streamlined alignment with simplified targets, (3) GRPO with **stepwise fine-grained rewards** per structural unit --> NER/RE/EE
  ) Tags: GRPO, curriculum, per-unit reward
  ) Results: beats instruction-tuned baselines on average for NER and RE with a smaller backbone, and gains on a production IE test (numbers **[unverified]**).
  ) Links: Paper https://arxiv.org/abs/2604.10633 / Code n/a / HF n/a
  ) Relevance: Per-entity or per-slot partial rewards suit our record format (each slot = one unit).

* **A Reasoning Paradigm for NER (ReasoningNER)** (2511.11978, Nov 2025)
  ) Goal: TLDR - NER CoT dataset generation --> CoT tuning --> reasoning enhancement with a comprehensive reward --> rationale, then entities
  ) Problem: "cognitive shortcutting", meaning pattern matching without verifiable reasoning, generalizes poorly zero-shot.
  ) Tags: CoT distillation, RL, NER
  ) Results: zero-shot SOTA, **+12.3 F1 over GPT-4**.
  ) Links: Paper https://arxiv.org/abs/2511.11978 / Code https://github.com/HuiResearch/ReasoningIE / HF n/a
  ) Relevance: Optional short rationale before the JSON for the hard distractor cases.

* **NE-R1: Enhancing NER via RL** (2609.02366, Sep 2026)
  ) Goal: TLDR - multi-task instruction tuning --> RL with CoT and on-demand retrieval --> entities
  ) Results: +2.52 F1 in domain, +1.18 zero-shot cross-domain, which is **modest** over strong SFT.
  ) Links: Paper https://arxiv.org/abs/2609.02366 / Code n/a / HF n/a
  ) Relevance: Sets expectations. RL margins over a strong SFT can be small for NER, and the gains are mostly OOD.

* **RexDrug: Reliable Multi-Drug Combination Extraction via Reasoning-Enhanced LLMs** (2603.08166, Mar 2026)
  ) Goal: TLDR - biomedical text --> SFT on multi-agent-generated reasoning traces --> RL with multi-dimensional reward --> **n-ary** drug-combination records
  ) Data: DrugComb, plus DDI13 for generalization. Traces come from multi-agent collaboration and were evaluated by human experts.
  ) Results: SOTA n-ary extraction on DrugComb; generalizes to DDI13.
  ) Links: Paper https://arxiv.org/abs/2603.08166 / Code https://github.com/DUTIR-BioNLP/RexDrug / HF n/a
  ) Relevance: Variable-length n-ary records, like our main entity plus its slots.

* **CONSISTRE: When Does Consistency Pay Off? Relational Constraints for LLM DocRE under Incomplete Annotations** (2607.24312, Jul 2026)
  ) Goal: TLDR - document --> LLM extraction + typed constraints (feedback and deterministic closure) + distillation + consistency-aware RL
  ) Results: **a +0.025 RL gain on DocRED vanishes when rescored on Re-DocRED.** Deterministic closure scores higher at 3.6× lower cost on DWIE.
  ) Links: Paper https://arxiv.org/abs/2607.24312 / Code n/a
  ) Relevance: **Warning.** Rewards and evals on incomplete gold mislead. Our synthetic gold is complete by construction. Also add cheap rule-based constraints (e.g., one location per car at a given time).

* **Extract-0: A Specialized LM for Document IE** (2509.22906, Sep 2025)
  ) Goal: TLDR - document chunks --> 280k synthetic extraction tasks --> LoRA SFT (0.53% of params) --> GRPO with a semantic-similarity reward --> JSON
  ) Results: mean reward 0.573 vs GPT-4.1 0.457 and o3 0.464 on 1,000 held-out tasks (its own benchmark).
  ) Links: Paper https://arxiv.org/abs/2509.22906 / Code n/a / HF dataset https://huggingface.co/datasets/HenriqueGodoy/extract-0
  ) Relevance: A full synthetic → LoRA → GRPO pipeline for schema JSON on a budget (7B).

* ⭐ **AOCG-LLM+: Automatic ontology construction and generation method** (Complex & Intelligent Systems, Jul 2026, DOI 10.1007/s40747-026-02372-7; 0 citations)
  ) Goal: TLDR - domain text --> LLM entity-relation extraction module trained with **RLHF** --> dynamic-radius hierarchical clustering --> conceptual ontology (concept hierarchy)
  ) Problem: manual ontology building doesn't scale and adapts poorly to complex domains.
  ) Tags: RLHF, ERE, ontology induction
  ) Results:
    - Entity precision 92.5%, +10.1 points over a plain LLM.
    - Relation F1 88.3%, +13 over BERT+GCN.
    - +20% entity recall over shallow statistical methods.
    - SOTA clustering metrics. LLM-calibrated evaluation (quantitative plus subjective).
  ) Later papers say: n/a (new).
  ) Links: Paper https://doi.org/10.1007/s40747-026-02372-7 / Code n/a / HF n/a
  ) Relevance: Medium-low.
    - Confirms RL feedback on an extraction module raises precision.
    - Its ontology/clustering stage could **auto-generate relation schemas** for our synthetic sampler.
    - Uses RLHF with a learned reward. We prefer verifiable rewards.

### 6b. Universal / instruction-tuned IE LLMs (stage-1 data and prompt design)

* **UniversalNER: Targeted Distillation from LLMs for Open NER** (2308.03279, Aug 2023, ICLR 2024)
  ) Goal: TLDR - Pile passages --> ChatGPT labels all entity types --> LLaMA-7B/13B conversation-style tuning with negative types --> open NER
  ) Data curation: ChatGPT annotation of Pile passages (Pile-NER), with type-name diversification and negative sampling of absent types. Evaluated on 43 datasets across 9 domains.
  ) Results: +30 F1 over Vicuna/Alpaca, **+7–9 over ChatGPT**, beats InstructUIE without direct supervision.
  ) Later papers say: a standard baseline (B2NER, GLiNER, TdSciNER). HGNet (2026) groups it with GLiNER as span-matching zero-shot NER. TdSciNER (2026) "significantly outperforms UniNER" on scientific NER, so domain specialization still wins.
  ) Links: Paper https://arxiv.org/abs/2308.03279 / Code https://github.com/universal-ner/universal-ner (375★, MIT) / HF model https://huggingface.co/Universal-NER/UniNER-7B-all / HF dataset https://huggingface.co/datasets/Universal-NER/Pile-NER-type
  ) Relevance: Recipe for LLM labels over real text plus negative types.

* **GoLLIE: Annotation Guidelines improve Zero-Shot IE** (2310.03668, Oct 2023, ICLR 2024)
  ) Goal: TLDR - Python dataclass schemas with **guideline docstrings** --> fine-tuned Code-LLaMA 7/13/34B --> instance list
  ) Data curation: public datasets (ACE05, CoNLL03, WikiEvents, RAMS…) re-expressed with guidelines. Training uses guideline paraphrasing, class shuffling and dropout **[unverified]**.
  ) Results: beats prior zero-shot IE. The ablation shows **detailed guidelines are key**.
  ) Later papers say: MR-UIE (2025): guidelines "as structured priors, enabling robust zero-shot generalization to new domains with only 3–5 examples". GLiDRE (2025): zero-shot LLMs, GoLLIE included, are "still limited" on doc-level RE.
  ) Links: Paper https://arxiv.org/abs/2310.03668 / Code https://github.com/hitz-zentroa/GoLLIE (444★, Apache-2.0) / HF model https://huggingface.co/HiTZ/GoLLIE-7B / HF dataset n/a (built from public corpora)
  ) Relevance: Our prompt = guidelines. Copy the paraphrase/shuffle/dropout augmentation.

* **KnowCoder: Coding Structured Knowledge into LLMs for UIE** (2403.07969, Mar 2024, ACL 2024)
  ) Goal: TLDR - schemas as Python classes (>30k types) --> schema-understanding pretraining (~1.5B auto-built examples) --> schema-following instruction tuning --> code output
  ) Results: +49.8% F1 few-shot over LLaMA2, up to +12.5% zero-shot, +21.9% low-resource, +7.5% supervised.
  ) Later papers say: "addresses this limitation by introducing a comprehensive schema representation in code format, integrating taxonomies, constraints" (Event-extraction guidelines paper, 2025, 37 citations). A common baseline alongside GoLLIE and B2NER (SaM 2025).
  ) Links: Paper https://arxiv.org/abs/2403.07969 / Code https://github.com/ICT-GoKnow/KnowCoder (110★, NOASSERTION) / HF model: third-party GGUF only / HF dataset https://huggingface.co/datasets/golaxy/KnowCoder-Schema-Library
  ) Relevance: Schema library to sample relation types from. Code vs JSON format is an A/B option.

* **KnowCoder-X: Boosting Multilingual IE via Code** (2411.04794, Nov 2024)
  ) Goal: TLDR - multilingual schemas as Python classes --> cross-lingual alignment instruction tuning on translated-instance prediction --> code IE in many languages
  ) Data curation: **ParallelNER**, 257k bilingual parallel samples. Built with a robust 3-stage synthesis pipeline plus manual annotation.
  ) Results: **zero training on 29 unseen languages, yet +30.17% over ChatGPT and +20.03% over SOTA**. 64 IE benchmarks in zh/en.
  ) Links: Paper https://arxiv.org/abs/2411.04794 / Code https://github.com/ICT-GoKnow/KnowCoder / HF n/a
  ) Relevance: **Key for he/fa.** Parallel (en ↔ he/fa) instances with a shared schema boost cross-lingual transfer. Generate each synthetic record in en + he + fa.

* **IEPile: Unearthing Large-Scale Schema-Based IE Corpus** (2402.14710, Feb 2024, ACL 2024)
  ) Goal: TLDR - 33 IE datasets cleaned --> schema-based instructions with **hard-negative and batched schemas** --> 0.32B-token bilingual corpus --> LoRA on LLaMA2/Baichuan2
  ) Results: significant zero-shot gains over baselines (numbers **[unverified]**).
  ) Later papers say: used to train OneKE, which industrial KG work applies. MMUIE (2026) notes IEPile-style work is "predominantly … sentence-level", the gap MMUIE fills. Frame-semantic RE (2025): tuning-based methods "are often resource-intensive and less adaptable to rapidly evolving foundation models".
  ) Links: Paper https://arxiv.org/abs/2402.14710 / Code https://github.com/zjunlp/IEPile (214★) / HF model https://huggingface.co/zjunlp/llama2-13b-iepile-lora / HF dataset https://huggingface.co/datasets/zjunlp/iepile
  ) Relevance: Negative-schema sampling recipe (H9).

* **B2NER: Universal Entity Taxonomy across Datasets and Languages for Open NER** (2406.11192, Jun 2024, COLING 2025)
  ) Goal: TLDR - 54 en/zh datasets --> resolve inconsistent definitions into 400+ distinguishable types + diversity-based pruning --> compact B2NERD --> LoRA InternLM2.5-7B / InternLM2-20B
  ) Results: **+6.8–12.0 F1 over GPT-4**. Beats prior methods on 3 OOD benchmarks across 15 datasets and 6 languages.
  ) Later papers say: cited in data-diversity work. "Small-scale, high-quality datasets can outperform larger ones … data diversity playing a crucial role" (2025, 36 citations).
  ) Links: Paper https://arxiv.org/abs/2406.11192 / Code https://github.com/UmeanNever/B2NER / HF model https://huggingface.co/Umean/B2NER-Internlm2.5-7B-LoRA / HF dataset https://huggingface.co/datasets/Umean/B2NERD
  ) Relevance: **Diversity over volume.** Prune our synthetic data for type and semantic diversity.

* **ADELIE: Aligning LLMs on IE** (2405.05008, May 2024, EMNLP 2024)
  ) Goal: TLDR - IEInstruct alignment corpus (closed, open and on-demand IE) --> SFT --> DPO
  ) Results: SOTA among open models on held-out IE, and **general capabilities show no noticeable decline**.
  ) Links: Paper https://arxiv.org/abs/2405.05008 / Code https://github.com/THU-KEG/ADELIE (57★) / HF: third-party GGUF / HF dataset n/a
  ) Relevance: SFT → DPO alternative to GRPO, with general ability preserved.

* **YAYI-UIE: Chat-Enhanced Instruction Tuning for UIE** (2312.15548, Dec 2023)
  ) Goal: TLDR - mix chat data with IE instruction data --> zh/en UIE
  ) Results: SOTA in Chinese, comparable in English. **Mixing chat data helps.**
  ) Links: Paper https://arxiv.org/abs/2312.15548 / Code https://github.com/wenge-research/YAYI-UIE (318★, Apache-2.0) / HF model https://huggingface.co/wenge-research/yayi-uie
  ) Relevance: Supports general-data replay.

* **InstructUIE** (2304.08085, Apr 2023): Flan-T5-11B on IE INSTRUCTIONS (32 datasets). It is the older baseline that UniversalNER beats. Code https://github.com/BeyonderXX/InstructUIE (398★, MIT); HF https://huggingface.co/ZWK/InstructUIE.

* **OneKE: Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System** (2412.20005, Dec 2024)
  ) Goal: TLDR - web/PDF --> multi-agent + configurable knowledge base --> schema-guided extraction. The underlying model is LLaMA fully fine-tuned on IEPile.
  ) Links: Paper https://arxiv.org/abs/2412.20005 / Code https://github.com/zjunlp/OneKE (197★, MIT) / HF model https://huggingface.co/zjunlp/OneKE
  ) Relevance: Reference for system and deployment design.

* **Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest** (2502.11275, Feb 2025)
  ) Goal: TLDR - reframe next-token *prediction* as next-tokens *extraction* (NTE) --> 102.6M examples from LLM pre-training and post-training data --> small extractor
  ) Results: better few-shot IE than prior pretrained IE models.
  ) Links: Paper https://arxiv.org/abs/2502.11275 / Code n/a / HF model https://huggingface.co/KomeijiForce/Cuckoo-C4 / HF dataset https://huggingface.co/datasets/KomeijiForce/Cuckoo_C4_Rainbow
  ) Relevance: Cheap self-supervised data idea that also applies to he/fa web text.

* **OmniStruct: Universal Text-to-Structure Generation across Diverse Schemas** (2511.18335, Nov 2025)
  ) Goal: TLDR - adapt IE, table and function-call datasets into a text-to-structure benchmark --> synthetic task generation for training --> small models
  ) Results: with **no supervised OmniStruct data**, small models fine-tuned on synthetic data **rival GPT-4o**.
  ) Links: Paper https://arxiv.org/abs/2511.18335 / Code n/a / HF n/a
  ) Relevance: Synthetic data alone can reach frontier level on arbitrary schemas.

* **GLiNER / GLiNER-multitask / GLiREL / GLiNER2** (2311.08526, 2406.12925, 2501.03172, 2507.18546)
  ) Goal: TLDR - encoder that matches label prompts to spans (and pairs) --> CPU-fast zero-shot NER and RE
  ) Links: Code https://github.com/urchade/GLiNER (4243★, Apache-2.0); https://github.com/fastino-ai/GLiNER2 (2462★, Apache-2.0); https://github.com/jackboyla/GLiREL (297★) / HF https://huggingface.co/urchade/gliner_multi-v2.1, https://huggingface.co/fastino/gliner2-large-v1, https://huggingface.co/jackboyla/glirel-large-v0
  ) Relevance: Cheap baselines, and pre-taggers for decoy candidates. They don't follow guidelines.

* **GLiNER-2.5 (Fastino)** (no paper yet; successor of GLiNER2)
  ) Goal: TLDR - schema-driven encoder --> NER + classification + structured/relation extraction in one pass, CPU-friendly
  ) Links: HF https://huggingface.co/fastino/gliner2.5-multi-v1, https://huggingface.co/fastino/gliner2.5-base-v1, https://huggingface.co/fastino/gliner2.5-small-v1 (also `GLiNER2.5-Decide` decision variants)
  ) Relevance: Encoder candidate for the experiment matrix (plan §4.6, E1.2). The multi variant matters for he/fa. The `-Decide` models are a possible E2 verifier baseline.

* **GLiNER-Relex (Knowledgator)** (no paper yet)
  ) Goal: TLDR - GLiNER extended to joint entity + relation extraction with label prompts for both
  ) Links: HF https://huggingface.co/knowledgator/gliner-relex-multi-v1.0, https://huggingface.co/knowledgator/gliner-relex-large-v1.0, https://huggingface.co/knowledgator/gliner-relex-base-v1.0
  ) Relevance: The most direct encoder baseline for joint E+R (plan §4.6, E1.3). Fine-tune it on our synthetic data and compare it against the DeBERTa and LLM rows.

* **NuExtract (NuMind 1.x / 2.0 / 3)** (no paper)
  ) Goal: TLDR - text + JSON template --> fine-tuned LM --> filled JSON
  ) Links: Code https://github.com/numindai/nuextract (272★, MIT) / HF https://huggingface.co/numind/NuExtract3
  ) Relevance: Off-the-shelf template-JSON baseline to beat.

* ⭐ **MMUIE: Massive Multi-Domain Universal IE for Long Documents** (Findings of EACL 2026, pp. 6338–6370; Zhang, Chen, Li, Tu, Jing, Jia, Zheng)
  ) Goal: TLDR - long documents from 34 domains --> **automated annotation = knowledge-base alignment + LLM extraction** of fine-grained entities, aliases and triples --> weakly supervised train + manually verified test --> DocUIE fine-tuned model
  ) Problem: universal IE work is sentence-level, and long-document, multi-domain IE has no large benchmark.
  ) Data curation: 723 entity types, 456 relation types. The KB + LLM pipeline produces the training set, and humans verify the test set.
  ) FT: fine-tuned with LLaMA-Factory (repo), backbone **[unverified]**.
  ) Results: sentence-level UIE models and advanced LLMs underperform on long docs. DocUIE generalizes and transfers across domains. Also evaluated on Re-DocRED, DWIE and SciERC.
  ) Later papers say: n/a (new).
  ) Links: Paper https://aclanthology.org/2026.findings-eacl.334/ / Code https://github.com/Shuyi-zsy/Massive-Multi-Domain-UIE (2★) / HF model https://huggingface.co/shuyi-zsy/DocUIE / HF dataset https://huggingface.co/datasets/shuyi-zsy/MMUIE
  ) Relevance: **High.**
    - Long-document training data, needed for H2/H3/H6.
    - KB+LLM annotation is a cheap way to label real he/fa articles.
    - "Weak train + verified test" is a sensible split for our data too.

* **Schema-Grounded Multitask Instruction Fine-tuning for Joint Biomedical NER and RE in Pharmacovigilance** (bioRxiv, Aug 2026, DOI 10.64898/2026.07.30.741807; Rehana, Hur)
  ) Goal: TLDR - BC5CDR + ADE + DDI --> one shared schema-in-prompt JSON interface --> LoRA on Llama-3.2-3B-Instruct and Qwen3-8B --> entities + relation pairs
  ) Data curation: 3 public corpora reformatted into one generation interface. No synthetic data.
  ) Results:
    - Qwen3-8B fine-tuned: **89.42 entity F1 / 62.32 relation F1**.
    - Llama-3B fine-tuned: 87.63 / 58.42, **beating zero-shot 8B**.
    - **JSON parse failures 23.5% → 0.11%.**
    - RE ≪ NER. The authors call for **explicit negative-relation supervision**.
  ) Later papers say: n/a (new).
  ) Links: Paper https://www.biorxiv.org/content/10.64898/2026.07.30.741807v1 / Code n/a / HF n/a
  ) Relevance: **Medium-high, despite the biomedical domain.**
    - Same base family and method as our stage 1 (Qwen3-8B + LoRA + schema JSON).
    - Shows format reliability comes almost for free with SFT.
    - The RE gap supports our distractor/negative focus.

### 6c. Synthetic data, distillation, low-resource (stage-0 data recipes)

* **SynthIE: Exploiting Asymmetry for Synthetic Training Data Generation** (2303.04132, Mar 2023, EMNLP 2023)
  ) Goal: TLDR - sample Wikidata triple subgraphs --> LLM writes text **from** the triples (the "reverse", easy direction) --> 1.8M pairs --> Flan-T5 base/large SFT for closed IE
  ) Data curation: subgraph sampling balanced over relations. Human evaluation rates the text above REBEL's distant labels.
  ) Results: **+57 micro-F1 / +79 macro-F1** over prior SOTA at equal model size.
  ) Later papers say: Wikontic (2025) says closed-IE datasets "are small, noisy and incomplete" (this is the motivation for synthetic data). StructText (2025) notes synthetic evaluation stays "confined to news/encyclopedic prose". SNaRe (2025) says joint label+sentence generation suffers "domain drift". **Our inject-into-real-articles approach addresses that drift.**
  ) Links: Paper https://arxiv.org/abs/2303.04132 / Code https://github.com/epfl-dlab/SynthIE (63★, MIT) / HF model https://huggingface.co/martinjosifoski/SynthIE / HF dataset https://huggingface.co/datasets/martinjosifoski/SynthIE
  ) Relevance: Facts-first gives exact gold. **Basis of S1/S2 in Table 3.**

* **GuideX: Guided Synthetic Data Generation for Zero-Shot IE** (2506.00649, May 2025, Findings of ACL 2025 **[unverified]**)
  ) Goal: TLDR - document --> LLM induces domain schema + guidelines --> labels instances --> Llama-3.1-8B SFT
  ) Results: SOTA on 7 zero-shot NER benchmarks. **Up to +7 F1 with no human data**, ~+2 F1 combined with human data.
  ) Links: Paper https://arxiv.org/abs/2506.00649 / Code **[unverified]** / HF model https://huggingface.co/HiTZ/Llama-3.1-GuideX-8B / HF dataset https://huggingface.co/datasets/HiTZ/GuideX_pre-training_data
  ) Relevance: Auto-generate slot definitions and guidelines for new main-entity types (S6).

* **DocIE@XLLM25: ICL for IE using Fully Synthetic Demonstrations** (2507.05997, Jul 2025, XLLM workshop)
  ) Goal: TLDR - 5k+ Wikipedia abstracts --> reasoning LLM generates ~59k entities and ~30k triples --> retrieval of demonstrations for ICL (no fine-tuning)
  ) Results: details **[unverified]**. The data and pipeline are the main contribution.
  ) Links: Paper https://arxiv.org/abs/2507.05997 / Code n/a / HF n/a
  ) Relevance: Synthetic labels over real Wikipedia, same source as ours. Retrieved demos also feed OSLLM-style inference.

* ⭐ **Enhancing Low-Resource Joint Entity and Relation Extraction Using LLMs within Semi-Supervised Learning** (ACM TKDD, Feb 2026, DOI 10.1145/3799423; Wang, Shen, Qi, Dai; 1 citation)
  ) Goal: TLDR - few labels + unlabeled text --> LLM augments data --> SSL with consistency regularization + pseudo-labels --> **iterative PEFT of the LLM augmenter** with feedback from the joint extractor
  ) Problem: joint ERE models need many labels. LLM augmentation alone adds noise.
  ) Tags: semi-supervised, augmentation, PEFT, pseudo-labels, low-resource
  ) Results: **SOTA on 4 low-resource joint ERE benchmarks** (numbers **[unverified]**).
  ) Later papers say: n/a (1 citation).
  ) Links: Paper https://doi.org/10.1145/3799423 / Code n/a / HF n/a
  ) Relevance: **Medium.** A path to use *real* unlabeled he/fa news: our model pseudo-labels it, keep confident and consistent outputs, retrain (S11). Run it after the synthetic-data model exists.

* Also covered under 6b: **UniversalNER** (targeted distillation), **IEPile/B2NERD** (unification + negatives), **KnowCoder-X ParallelNER** (parallel multilingual), **OmniStruct / Extract-0 / Cuckoo** (synthetic structure tasks).

* ⭐ **Distill-RE: Few-Shot Relation Extraction via Teacher LLM Reasoning Distillation** (AKBC 2026, pages 359–378; Sullivan-Pao, Alemany, Simek, MIT Lincoln Lab)
  ) Problem: LLMs counterintuitively **underperform** small PLMs on few-shot RE because of a systematic over-prediction bias (they guess a relation too often).
  ) Goal: TLDR - teacher LLM writes **chain-of-thought reasoning over the existing few-shot training examples** (no new instances synthesized) --> supervises **two separate students**: (1) a classification model that predicts the relation, (2) a **confirmation model that filters false positives** from (1).
  ) Data curation: no data augmentation — reasoning traces are generated over TACRED/TACREV/Re-TACRED's own 8-/16-/32-shot splits; the gain is from *how* the existing shots are used, not from more data.
  ) Results: **new SOTA for RE student scales 3B–70B**; best config beats the strongest baseline by **+31.5 micro-F1** averaged across TACRED/TACREV/Re-TACRED at 8/16/32-shot. Without synthesizing new instances it still comes within 1.8 F1 of the best data-augmentation baseline.
  ) Links: Paper https://aclanthology.org/2026.akbc-1.25/ (PDF https://aclanthology.org/2026.akbc-1.25.pdf) / Code n/a / HF n/a
  ) Relevance: **High.** The classify+confirm two-model split is a direct template for a cheap **decoy-rejection verifier** in our pipeline (§6g); the "reason over existing shots, don't synthesize" framing is a cheaper alternative/complement to our full synthetic-injection approach for the *classification* side specifically.

### 6d. Document-level, HTML / semi-structured, retrieval, robustness

* ⭐ **OSLLM: A Retrieve-Reason-Refine Framework for Multi-Domain RE with LLMs** (ICME 2025; Jie Zhou et al.; no arXiv; 1 citation)
  ) Goal: TLDR - (1) **Retrieve** demonstrations by relation-aware embedding --> (2) **Reason** with the LLM --> (3) **Refine** via template self-optimization + answer self-evaluation. **Boundary samples** where small fine-tuned RE models are unsure are routed to the LLM.
  ) Problem: multi-domain RE. Small supervised models are fast but brittle; LLMs are general but costly and imprecise.
  ) Tags: retrieval ICL, self-refinement, cascade (small model → LLM)
  ) FT: no LLM fine-tuning (inference framework) **[unverified]**.
  ) Results: improves multi-domain RE over baselines (numbers **[unverified]**, no full text).
  ) Links: Paper: IEEE ICME 2025 proceedings (DOI **[unverified]**) / Code n/a / HF n/a
  ) Relevance: **Inference-time add-on.** (a) Retrieve few-shot examples per prompt from our synthetic pool. (b) Cascade: fine-tuned 8B first, escalate low-confidence records to a larger model. (c) Self-evaluation as a cheap verifier in the data pipeline.

* **CovEReD: Consistent Document-Level RE via Counterfactuals** (2407.06699, Jul 2024, Findings of EMNLP 2024)
  ) Goal: TLDR - Re-DocRED --> **consistent entity replacement** (5 variants) --> counterfactual docs --> train on the factual + counterfactual mix
  ) Results: factual-only models break on counterfactual docs. The mix improves consistency.
  ) Links: Paper https://arxiv.org/abs/2407.06699 / Code https://github.com/amodaresi/CovEReD / HF dataset https://huggingface.co/datasets/amodaresi/Re-DocRED-CF (MIT)
  ) Relevance: **Direct template for S3**. Swapped names/IDs/places block world-knowledge shortcuts.

* **Re-DocRED: Revisiting DocRED, the false-negative problem** (2205.12696, EMNLP 2022)
  ) Goal: TLDR - re-annotate DocRED's missing relations --> 3,053/500/500 docs
  ) Later papers say: ConsistRE (2607.24312) finds method rankings and an RL gain **flip** between DocRED and Re-DocRED scoring.
  ) Links: Paper https://arxiv.org/abs/2205.12696 / Code https://github.com/tonytan48/Re-DocRED (69★, MIT) / HF dataset https://huggingface.co/datasets/tonytan48/Re-DocRED
  ) Relevance: Real-text English doc-level eval.

* **AutoRE: Document-Level RE with LLMs** (2403.14888, Mar 2024)
  ) Goal: TLDR - document --> **Relation → Head → Facts (RHF)** decomposition --> QLoRA open LLM --> triples
  ) Results: SOTA on Re-DocRED per the authors **[unverified numbers]**.
  ) Links: Paper https://arxiv.org/abs/2403.14888 / Code **[unverified]** / HF (third-party) https://huggingface.co/francisco-the-man/re-docred-mistral-7b-autore
  ) Relevance: Our "main entity → slots" output follows the same head-first decomposition.

* **ReaderLM-v2: Small LM for HTML → Markdown and JSON** (2503.01151, Mar 2025)
  ) Goal: TLDR - raw HTML (up to 512K tokens) --> 1.5B model trained on draft/refine/critique synthetic data --> Markdown or schema JSON
  ) Results: **+15–20% over GPT-4o** on the authors' benchmarks, especially >100K-token pages.
  ) Links: Paper https://arxiv.org/abs/2503.01151 / HF model https://huggingface.co/jinaai/ReaderLM-v2 (CC-BY-NC-4.0 **[unverified]**)
  ) Relevance: Pre-processor option for HTML inputs, and a synthesis recipe for HTML tables (S13).

* **SCRIBES: Web-Scale Script-Based Semi-Structured Extraction with RL** (2510.01832, Oct 2025)
  ) Goal: TLDR - pages from one site --> LLM writes reusable extraction scripts --> RL reward = cross-page layout similarity
  ) Results: +13% script quality, +4% downstream QA for GPT-4o.
  ) Links: Paper https://arxiv.org/abs/2510.01832 / Code n/a
  ) Relevance: Tables/lists/infoboxes. Low priority (script generation, not direct extraction).

* **ScrapeGraphAI-100k** (2602.15189, Feb 2026)
  ) Goal: TLDR - 93,695 real (page Markdown, prompt, JSON schema, response) events, 18k+ schemas, 15 languages
  ) Caveat: labels mark schema conformance only; **semantic correctness is not validated**.
  ) Links: Paper https://arxiv.org/abs/2602.15189 / HF dataset https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k
  ) Relevance: Real-world schema and prompt diversity. Re-label with a frontier model before training.

* ⭐ **From Named Entities to Relations: End-to-End Biomedical IE** (CLEF 2025 BioASQ GutBrainIE; Keinan, Cohen, Tsarfaty, BIU ONLP; 4 citations)
  ) Goal: TLDR - GLiNER-Large-Bio NER --> ATLOP doc-level RE (roberta-large) --> pipeline end-to-end
  ) Data curation: trains on mixed Platinum / Gold / Silver annotation tiers (expert → student quality).
  ) Results: binary RE 0.612 (5th/11), ternary RE 0.591 (6th/12), NER 0.48 (15th), **mention-level RE 0.18 (11th)**. RE scores are capped by NER errors.
  ) Links: Paper https://ceur-ws.org/Vol-4038/paper_24.pdf / Code n/a / HF n/a
  ) Relevance: **Medium-low (no LLM fine-tuning).** Lessons: pipeline error propagation (NER → RE) argues for joint generative extraction, and **quality tiers** are a useful data-mix knob (verified synthetic > silver real).

### 6e. Forgetting and post-training (how not to break the base model)

* **LoRA Learns Less and Forgets Less** (2405.09673, May 2024, TMLR)
  ) TLDR: Llama-2 7B/13B on code/math, LoRA (r16–256) vs full fine-tuning --> LoRA gains less but **forgets less**. Full fine-tuning learns perturbations of rank 10–100× typical LoRA ranks.
  ) Later papers say: MemSFT (2026): PEFT **still** forgets, with an inverse-linear relation between target gain and retention. Cited by "Intruder Threshold" (2026).
  ) Links: Paper https://arxiv.org/abs/2405.09673 / Code https://github.com/danbider/lora-tradeoffs (22★)
  ) Relevance: LoRA is necessary but not sufficient. Measure forgetting anyway.

* **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (2410.21228, Oct 2024)
  ) TLDR: SVD shows LoRA adds **intruder dimensions** (singular vectors orthogonal to the pretrained ones). More intruders → more forgetting. Higher rank + rank-stabilized scaling reduces them.
  ) Links: Paper https://arxiv.org/abs/2410.21228 / Code https://github.com/reeceshuttle/intruder-dimensions (14★, MIT)
  ) Relevance: Use r ≥ 32 on all linear layers (+rsLoRA). Run the intruder check as a diagnostic.

* **LoRA Without Regret** (Thinking Machines blog, Sep 2025)
  ) TLDR: LoRA on **all** layers (incl. MLP) matches full fine-tuning for small/medium SFT and for RL. Even rank-1 suffices for RL, since policy gradient carries few bits per episode. Optimal LoRA LR ≈ 10× full-FT LR.
  ) Links: https://thinkingmachines.ai/blog/lora/ / Code https://github.com/thinking-machines-lab/tinker-cookbook (4245★, Apache-2.0)
  ) Relevance: Use LoRA for both stages. A small GRPO rank (8–32) is fine.

* **SFT Memorizes, RL Generalizes** (2501.17161, Jan 2025, ICML 2025)
  ) TLDR: on rule and visual variants, outcome-reward RL generalizes OOD while SFT memorizes. **SFT is still needed** to stabilize the output format before RL.
  ) Later papers say: "When Models Edit Too Much" (2026): SFT "overfits narrow synthetic distributions" while RL preserves transferable competence.
  ) Links: Paper https://arxiv.org/abs/2501.17161 / Code https://github.com/LeslieTrue/SFTvsRL (337★) / HF dataset https://huggingface.co/datasets/tianzhechu/SFTvsRL_Data
  ) Relevance: **Short SFT for format → GRPO for robustness**, especially important because our synthetic data is narrow.

* **RL's Razor: Why Online RL Forgets Less** (2509.04259, Sep 2025, ICLR 2026; ~220 citations)
  ) TLDR: forgetting is predicted by **forward KL(fine-tuned ‖ base) on the new task's distribution**. On-policy RL is implicitly biased toward KL-minimal solutions, so it forgets less than SFT at equal task accuracy.
  ) Later papers say: "Denser ≠ Better" (2026) confirms RFT forgets less than SFT and attributes it to on-policy data. "On-Policy Self-Distillation without Any Supervision" (2026) builds on it.
  ) Links: Paper https://arxiv.org/abs/2509.04259 / Code: no official repo (community https://github.com/SakanaAI/rl-razor-mnist)
  ) Relevance: **Log KL to base on held-out task prompts every N steps** as the forgetting early-warning metric.

* **Retaining by Doing: On-Policy Data Mitigates Forgetting** (2510.18874, Oct 2025)
  ) TLDR: Llama/Qwen. RL forgets less than SFT because of **on-policy data**. Approximately on-policy data (samples refreshed each epoch) captures most of the benefit.
  ) Links: Paper https://arxiv.org/abs/2510.18874 / Code https://github.com/princeton-pli/retaining-by-doing (47★)
  ) Relevance: Cheap middle ground: **rejection-sampling SFT** on our own model's correct outputs (RFT) before GRPO.

* **GKD: On-Policy Distillation of LMs** (2306.13649, ICLR 2024) + **On-Policy Distillation** (Thinking Machines blog, Oct 2025)
  ) TLDR: the student generates and the teacher scores each token (reverse KL / GJSD). Dense signal, roughly RL-level quality at ~1/10 the compute. Also **restores** instruction following after domain SFT by distilling from the original model.
  ) Links: Paper https://arxiv.org/abs/2306.13649 / TRL `GKDTrainer` https://github.com/huggingface/trl / cookbook https://github.com/thinking-machines-lab/tinker-cookbook
  ) Relevance: Repair stage, or Qwen3-32B → 8B distillation (same tokenizer required).

* **SDFT: Self-Distillation Bridges Distribution Gap** (2402.13669, ACL 2024)
  ) TLDR: the model **rewrites gold responses in its own words** and trains on the rewrites. Matches SFT on the task while preserving general ability and safety.
  ) Later papers say: SelfAug (2025): SDFT "struggles with format-specific tasks … JSON" and SelfAug beats it. OntoTune (2025): self-training "bridges the distribution gap" and mitigates forgetting.
  ) Links: Paper https://arxiv.org/abs/2402.13669 / Code https://github.com/sail-sg/sdft (167★)
  ) Relevance: **Apply only to free-text parts** (rationale, explanations). **Never rewrite the gold JSON.**

* **Self-Distillation Enables Continual Learning** (Shenfeld et al., Jan 2026, arXiv ID **[unverified]**)
  ) TLDR: the model *with demonstrations in context* acts as teacher for itself *without context* (on-policy). New skills with little forgetting.
  ) Links: Code https://github.com/idanshen/Self-Distillation (733★)
  ) Relevance: Natural fit, since our gold records can be the in-context "privileged" info. Watch **Privileged Context as Drift** (2610.07842): gold-only teacher hints can cause drift.

* **SynthSelf** (2610.13113, Oct 2026): a slightly larger same-family model rewrites the new-task data, plus rejection sampling. Beats 6 off- and on-policy distillation baselines. Use Qwen3-14B to rewrite targets for Qwen3-8B. Code n/a.

* **Spurious Forgetting in Continual Learning** (2501.13453, ICLR 2025): many "forgetting" drops are **task-alignment/format loss**, not lost knowledge, and are recoverable with a little replay. Code https://github.com/zzz47zzz/spurious-forgetting (64★).

* **DFT: On the Generalization of SFT, RL Perspective with Reward Rectification** (2508.05629, ICLR 2026): scales each token's loss by its (stop-gradient) probability. One-line change that generalizes better than SFT. Code https://github.com/yongliang-wu/DFT (928★). A cheap A/B test in stage 1.

* **Model merging: WiSE-FT / TIES / DARE** (2109.01903, 2306.01708, 2311.03099): interpolate or sparsify task vectors to restore base behavior. With LoRA, scaling the adapter to 0.5–0.8 is a one-line WiSE-FT. PEFT `add_weighted_adapter` (ties/dare/svd) merges several adapters. Code: wise-ft (769★), ties-merging (224★), MergeLM (877★).

### 6f. Hebrew / Persian models and resources

* **DictaLM 2.0** (2407.07080, Jul 2024): Mistral-7B with continued pretraining on ~200B he+en tokens plus an extended Hebrew tokenizer → instruct model. Introduces Hebrew LLM eval tasks. HF https://huggingface.co/dicta-il/dictalm2.0-instruct.
* **DictaLM-3.0** (Dicta, 2025–2026; 1.7B / 12B / 24B-Thinking, Apache-2.0 **[check license per card]**): newest Hebrew-specialist family. **A/B base candidate vs Qwen3-8B for Hebrew.** HF https://huggingface.co/dicta-il.
* **NEMO** (2007.15620, TACL 2021): Hebrew nested, morphology-aware NER corpus and models. **Main real-text Hebrew NER eval.** Code https://github.com/OnlpLab/NEMO-Corpus. Also `dicta-il/dictabert-ner` as a strong encoder baseline.
* **PERLEX** (2005.06588): expert Persian translation of SemEval-2010 T8 (~10k sentences). The **only Persian RE benchmark**; BERT-EM-MTB 77.66 F1. HF https://huggingface.co/datasets/Aeirya/PERLEX.
* **ParsiNLU** (2012.06154): Persian NLU suite. Useful for the fa forgetting/replay suite. Persian LLMs: **Dorna** (PartAI, Llama-3-8B based) and **PersianMind** (non-commercial license).
* **RED^FM / SREDFM** (2306.09802) and **MultiTACRED** (2305.04582): multilingual RE (silver Wikipedia/Wikidata and machine-translated TACRED) with **no Hebrew or Persian**. Their recipes (Wikidata alignment + NLI/critic filtering; MT + projection) can be applied to he/fa Wikipedia for a silver real-text eval set.
* Cross-lingual note: KnowCoder-X (6b) shows parallel en↔X instances with shared schemas transfer strongly to unseen languages. **Generate parallel en/he/fa versions of each synthetic record.**

### 6g. Zero-shot prompting and non-generative decision-model verifiers (inference-time / cheap confirmation)

* ⭐ **Zero-Shot Relation Classification with LLMs: Binary Prompting and Relevance Elicitation (BRE)** (AKBC 2026, pages 343–358; Sullivan-Pao, Alemany, Simek, MIT Lincoln Lab)
  ) Goal: TLDR - for each entity pair, ask the LLM a **true/false ("binary relevance") question per candidate relation** from a closed schema instead of one open-ended "what is the relation?" question --> score each "true" answer for relevance --> keep the single best-scoring relation (or none).
  ) Problem: open-ended zero-shot relation classification over-predicts (the LLM picks *some* relation even when none applies); this reduces false positives by decomposing the decision into calibrated binary checks.
  ) FT: **no fine-tuning** — pure zero-shot prompting framework.
  ) Results: **new SOTA zero-shot RC accuracy on the TACRED family** (TACRED/TACREV/Re-TACRED) **[exact numbers unverified, abstract only]**.
  ) Links: Paper https://aclanthology.org/2026.akbc-1.24/ (PDF https://aclanthology.org/2026.akbc-1.24.pdf) / Code n/a / HF n/a
  ) Relevance: **High, same team/companion paper to Distill-RE (§6c).** Binary true/false-per-candidate is exactly the decoy-rejection pattern we need (e.g. "is `12/2/26` the pickup date for the Mazda?" asked once per candidate date near the main entity), and it needs **no training** — usable immediately as a prompt-engineering baseline for our verifier before we build a trained one.

* **"Jev" decision models (TypeSafe AI, proprietary) and the OpenJev open-source family** (community project, not a single peer-reviewed paper; 2026)
  ) What it is: a **non-autoregressive "System-1" decision model** paradigm — instead of generating text, the model answers *typed* questions about a fixed input state (`Choice` = pick one of N labels, `Score` = place on an ordered rubric, `Noul` = calibrated P(true) for a yes/no gate) in a **single forward pass**, reading the answer off logits/label heads rather than parsing generated text. Asking 20 typed questions costs about the same as asking one.
  ) Architecture (OpenJev, `S1LV3RJ1NX/openjev`): default is a **rank-16 LoRA adapter on Qwen3-1.7B** (won an ablation against a ModernBERT encoder head on 7/7 held-out benchmarks for in-domain accuracy, e.g. 0.605 vs 0.343 on Banking77); the ModernBERT encoder alternative trades accuracy for 2.8x lower p95 latency (20ms vs 56ms) and ~6x smaller footprint (0.6GB vs 3.4GB). **Fine-tune-first tool**: a few hundred labeled examples beat the closed commercial API it mimics on that task; zero-shot it is weaker.
  ) Biomedical RE adaptation: **`Gabriel382/BioJev`** — "biomedical decision-model benchmark and adaptation framework built on OpenJev, with reproducible multi-dataset evaluation, biomedical NLI, **relation extraction, calibration**, and cross-dataset generalization experiments." **`lighteternal/biodecision-v2-4b`** (HF model) benchmarks the same pattern on drug–protein / drug–drug **relation decisions** against TypeSafe's closed Jev 1.13.
  ) FT: LoRA SFT distillation of a small backbone into typed classification/scoring heads (see Table 1 row "Non-generative decision-model distillation").
  ) Results: beats the closed API it was fine-tuned to imitate on its training task; **no RE-specific benchmark numbers found [unverified]** — BioJev is a benchmark/framework, not a single reported result.
  ) Links: OpenJev code https://github.com/S1LV3RJ1NX/openjev (16★, Apache-2.0) / HF https://huggingface.co/s1lv3rj1nx/openjev-general-lora / BioJev code https://github.com/Gabriel382/BioJev (MIT) / biodecision model https://huggingface.co/lighteternal/biodecision-v2-4b / curated alternatives list https://github.com/notsointresting/awesome-jev-family
  ) Relevance: **Medium-high, as a candidate verifier architecture, not a training recipe for the base extractor.** Instead of (or in addition to) BRE's prompted binary questions, a small LoRA "decision head" distilled from our own teacher could answer "is `<candidate>` the correct `<slot>` for `<main entity>`?" in one forward pass per decoy — much cheaper than re-generating/re-prompting the full 8B extractor per candidate. Treat as a **stage-2+ efficiency optimization** for the decoy-rejection verifier (§3), not a blocker for the initial pipeline.

---

## 7. Gaps, caveats and unverified items

* **No Hebrew RE dataset exists**, and Persian RE is limited to PERLEX (sentence-level, translated). Real-text he/fa doc-level evaluation must be built: a small manually checked set of ~200–300 docs per language is recommended (see plan).
* **No public he/fa HTML-with-tables extraction set.** We must render our own (S13).
* Numbers marked **[unverified]** came from partial reads or memory. Verify before citing externally, especially: MR-UIE, ProUIE, IEPile, AutoRE, OSLLM, TKDD-SSL, DocIE@XLLM25, and the GuideX venue.
* "What later papers say" comes from Semantic Scholar citation contexts for 12 papers. Many 2026 papers (MMUIE, AOCG-LLM+, pharmacovigilance, LA-RL, USP 2610.34343) have **no** citing papers yet.
* RL-vs-SFT forgetting evidence comes mostly from math/QA/tool tasks. For RE only R1-RE measured it (MATH-500/IFEval/GPQA). **We must measure forgetting ourselves.**
* RL gains on RE can be **evaluation artifacts** under incomplete gold (ConsistRE). Rewards and evals should use complete synthetic gold plus a verified real-text subset.
* The user's additional search hits (Nov 2026) were added with relevance ratings. Items rated Medium-low (AOCG-LLM+, BIU GutBrainIE) inform data and pipeline choices, not the training recipe.
* **BRE and Distill-RE (AKBC 2026) have no code/HF releases found** despite GitHub/HF search — abstract-only verification via ACL Anthology; exact result tables are **[unverified]**, and they have no citing papers yet (too new). Re-check for a camera-ready code release closer to the AKBC workshop date.
* **"Jev" is not a citable paper** — it is a proprietary product (TypeSafe AI) plus an open community clone family (OpenJev/BioJev). Treated here as an architecture pattern for §6g/Table 1, not as peer-reviewed evidence; no RE-specific benchmark numbers were found for the biomedical adaptation.
* **Document-length scope: first experiments (and likely beyond) target 1–4 pages, ~2K–8K tokens** (plain text / HTML-with-tables / email), not 30K+ (see plan Assumptions). This de-prioritizes the very-long-document papers (MMUIE, DocUIE, ReaderLM-v2's 512K-token claims, §6d) for v1 — they remain relevant background/benchmarks, but are not the initial target regime.
* **Distill-RE/BRE do not do entity extraction.** TACRED/TACREV/Re-TACRED are relation-*classification* benchmarks: subject/object spans are pre-marked in the input sentence; the student/teacher models only pick a relation label (Distill-RE) or answer true/false per candidate label (BRE) for an **already-given pair**. Their techniques transfer to us only as a **pair-classification/verification stage**, not as a replacement for entity extraction (see §8).

---

## 8. Recommended inference architecture: extract → propose candidate pairs → verify

This section reframes the single-pass "schema in, JSON records out" design (§3 of the plan) as three decoupled stages, each matched to a different lever from the papers above. Motivation: Distill-RE and BRE show that **pair-level relation decisions are far more reliable when they're a closed binary/choice check over a short candidate list** than when a single model has to both find candidates and decide correctness end-to-end (over-prediction bias, §6c/§6g). Decoupling also makes each stage separately trainable and debuggable.

1. **Stage A — Entity/mention extraction (LLM, open).** The fine-tuned extractor reads the raw text and emits all entity and attribute-value mentions (main-entity mentions, candidate dates, locations, IDs, org names, statuses) with spans, **without** committing to which value belongs to which main entity. This is a simpler sub-task than full record extraction (closer to NER/UIE span-tagging, e.g. UniversalNER/GoLLIE-style, §6b) and is expected to have higher recall and be easier to keep accurate under fine-tuning.
2. **Stage B — Candidate pair proposal (recall-oriented, over-generate).** For each main-entity mention, build a short list of candidate (slot, value) pairs from **two complementary sources**:
   - (a) **Model-proposed candidates**: the same LLM (or a second pass) directly guesses which values attach to which entity. Today this is what our single-pass pipeline does end-to-end, and per the user it is *not very accurate* on its own — keep it as one signal, not the only one.
   - (b) **Heuristic/structural candidates**: all values within a proximity window (same sentence, ±N sentences) **and** all values sharing the same structural context as the main entity — same table row, same list item, same paragraph, same email field block, same HTML row/cell/section. This directly targets the H-case distractor families in §3 (near-but-wrong decoys, subtitle-refers-to-a-table-N-lines-below, email-header-far-but-relevant fields) by guaranteeing the *correct* value is always in the candidate set, even when stage A/B(a) would miss or misrank it.
   - The union of (a) and (b) is intentionally over-generated (includes decoys by design) — precision is Stage C's job, not Stage B's.
3. **Stage C — Verifier (fine-tuned, precision-oriented, closed decision).** For each candidate (main entity, slot, value) pair, a trained verifier answers a **closed yes/no or choice question**: "is `<value>` the correct `<slot>` for `<main entity>`?" (or, when several candidates compete for one slot, "which of these candidates is the correct `<slot>`, if any?"). This is structurally identical to:
   - **BRE's** binary relevance elicitation (§6g) — true/false per candidate, prompted, no training needed as a baseline;
   - **Distill-RE's** confirmation model (§6c) — a trained second model that filters false positives from an over-eager first pass;
   - **Jev/OpenJev's** typed decision head (§6g) — a small LoRA/encoder model that answers the same question in one forward pass per candidate instead of re-prompting a full generative model, which matters here because Stage C may be called **once per candidate pair**, i.e. many times per document.
   Recommended order of implementation: (1) BRE-style prompting with the existing fine-tuned model as a zero-extra-training baseline → (2) a small trained classify+confirm verifier (Distill-RE-style, fine-tuned on our synthetic gold **and** the planted decoys as hard negatives — we already know which candidate is a decoy, so this is a free labeled dataset) → (3) if latency/cost demands it, distill the verifier into a Jev-style single-forward-pass decision model.
4. **Training data implication (free lunch from the existing pipeline):** because our synthetic injection (plan §2) already knows the true (entity, slot, value) and every planted decoy, **every generated document already yields positive and negative training pairs for Stage C with no extra annotation** — a candidate pair is positive if it matches the injected ground truth and negative if it's a decoy, a same-row/paragraph distractor, or an unrelated nearby value. This is a direct, low-cost way to train the verifier alongside the main extractor.
5. **Where this fits the training plan:** Stage A reuses the Stage-1 SFT / Stage-2 GRPO extractor (plan §4) with a narrower output (mentions, not records). Stage C is a new, smaller model/head (plan §4, new "stage 1c"), trained on the same synthetic corpus relabeled as pairwise positive/negative examples. Stage B needs no training — it is a deterministic proximity/structure rule plus the existing extractor's own guesses.
6. **Caveat:** this adds inference latency/calls (Stage C runs once per candidate, not once per document) — mitigate with batching, a cheap Stage-C model (Jev-style), or capping candidates per slot (e.g. top-k by proximity) before verification.
