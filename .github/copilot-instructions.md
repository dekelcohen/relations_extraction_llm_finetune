# Copilot instructions for this repository

## Model selection for agent work

Pick the model/reasoning-effort by task type:

- **Planning, complex logic, and troubleshooting** (e.g. designing the training pipeline, debugging GRPO/reward issues, reasoning through forgetting trade-offs, architecture decisions like §8's extract→candidate→verify split): use **Claude Opus, high reasoning effort**.
- **Implementation plans** (turning an agreed plan into concrete file/code changes, writing scripts, routine edits): use **Claude Sonnet 5.5, medium reasoning effort**.
- **Synthetic data generation work** (`re_synth/` pipeline, injection/distractor generation, judge/verifier prompts): use either **Sonnet** or **GPT Terra** — this is not yet settled; **try both and validate** (compare injected-sample quality, verifier pass rate, manual audit error rate) before standardizing on one.

When unsure which bucket a task falls into, default to the planning/troubleshooting tier (Opus high) rather than under-provisioning reasoning for a task that turns out to be complex.
