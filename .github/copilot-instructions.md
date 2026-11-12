# Copilot instructions for this repository

## Model selection for agent work

Pick the model/reasoning-effort by task type:

- **Planning, complex logic, and troubleshooting** (e.g. designing the training pipeline, debugging GRPO/reward issues, reasoning through forgetting trade-offs, architecture decisions like §8's extract→candidate→verify split): use **Claude Opus, high reasoning effort**.
- **Implementation plans** (turning an agreed plan into concrete file/code changes, writing scripts, routine edits): use **Claude Sonnet 5.5, medium reasoning effort**.
- **Synthetic data generation work** (`re_synth/` pipeline, injection/distractor generation, judge/verifier prompts): use either **Sonnet** or **GPT Terra** — this is not yet settled; **try both and validate** (compare injected-sample quality, verifier pass rate, manual audit error rate) before standardizing on one.

**Cost rule:** once complex planning is complete (design agreed, script/plan written), hand training + eval execution to a **Claude Sonnet 5.5, medium reasoning** subagent instead of running it on the planning tier.

When unsure which bucket a task falls into, default to the planning/troubleshooting tier (Opus high) rather than under-provisioning reasoning for a task that turns out to be complex.

### Model provider: Copilot only for Claude / GPT models

**Claude Sonnet, Claude Opus, and OpenAI GPT 6.1 (and other versions) models — Astra, Sol, Terra, Luna — must be called only through the Copilot provider, `copilot-<model id>`** (e.g. `copilot-claude-sonnet-5.5`, `copilot-claude-opus-5.5`, `copilot-gpt-6.1-terra`; routed by `re_synth/llm.py` to the Copilot CLI). Do not call them via OpenRouter or other providers. Use OpenRouter only for models the Copilot provider does not serve (e.g. Llama).

## Milestones: commit + push to GitHub and HF

At the end of **every stable task / milestone** (e.g. after synthetic generation, before and after a training run, after an eval/report update):

1. Update `docs/status.md` (milestone table, open issues, next items).
2. `git add -A && git commit` (with the Copilot co-author trailer) and `git push` — code, docs, and data under `data/` that fits GitHub (< 100 MB per file).
3. Push artifacts too big for GitHub (adapters, checkpoints, large datasets) to private HF Hub repos with `python scripts/hf_sync.py push <dir> --repo Shani123/<name>`; this updates `hf_artifacts.json`, which must be committed. Another machine resumes with `git pull && python scripts/hf_sync.py pull`.

Never commit `.env` (holds `OPENROUTER_API_KEY`, `HF_TOKEN`).
