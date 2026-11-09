# relations_extraction_llm_finetune

Fine-tune an open LLM (default: Qwen3-8B + LoRA) as a **generic, prompt-driven entity & relation extractor** (main entity + IDs, dates, locations, status, containing orgs) for Hebrew, Persian and English, using a synthetic "scrub → inject → GT-by-construction" dataset with hard distractors.

## Docs
- [Research survey](docs/research/entity_relation_extraction.md): top tables (training techniques, datasets, data-synthesis methods), hard cases H1–H12, ~45 paper summaries (2023–2026) with links, and anti-forgetting notes.
- [Plan: synthetic dataset + fine-tuning](docs/plans/synth_dataset_and_finetune.md): pipeline, public data, distractor generation, adversarial mining (Llama-3.3-70B vs frontier), SFT → GRPO, frameworks (TRL/Unsloth/vLLM/verl), evaluation, milestones.
- [Research tools](docs/research_tools/README.md): `paper_research.js` (arXiv/S2 helper).
