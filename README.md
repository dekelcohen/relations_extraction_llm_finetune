# relations_extraction_llm_finetune

Fine-tune an open LLM (default: Qwen3-8B + LoRA) as a **generic, prompt-driven entity & relation extractor** (main entity + IDs, dates, locations, status, containing orgs) for Hebrew, Persian and English, using a synthetic "scrub → inject → GT-by-construction" dataset with hard distractors.

## Docs
- [Research survey](docs/research/entity_relation_extraction.md): top tables (training techniques, datasets, data-synthesis methods), hard cases H1–H12, ~45 paper summaries (2023–2026) with links, and anti-forgetting notes.
- [Plan: synthetic dataset + fine-tuning](docs/plans/synth_dataset_and_finetune.md): pipeline, public data, distractor generation, adversarial mining (Llama-3.3-70B vs frontier), SFT → GRPO, frameworks (TRL/Unsloth/vLLM/verl), evaluation, milestones.
- [Research tools](docs/research_tools/README.md): `paper_research.js` (arXiv/S2 helper).
- [Status](docs/status.md): where things stand, the pilot-testset manual-review checklist, and the plan for making the testset harder.

## Repo layout
| Path | What |
|---|---|
| `re_synth/` | Synthetic data pipeline: `lexicon` (he/fa/en values, Jalali/Hebrew dates), `schemas` (8 schemas), `hardcases` (H1–H12 plans + decoys), `source` (Wikipedia/FineWeb-2 seeds via HF datasets-server), `inject` (LLM scrub+inject with `[[r0.slot]]` markers + verification), `layout` (text/HTML/email, H6 lists, H7 tables), `score` (normalized matching, metrics, GRPO reward), `mine` (weak vs strong tiers), `export` (SFT/GRPO jsonl), `convert/redocred` |
| `train/` | `sft.py` (TRL SFT + LoRA), `grpo.py` (TRL GRPO + vLLM), `rewards.py`, `merge.py` (merge / WiSE-scale LoRA), `sft_unsloth.py` (optional 1-GPU) |
| `eval/` | `predict.py` (vLLM/HF batch inference + scoring by lang/format/hard-case/tier), `forgetting.sh` (lm-eval general benchmarks) |
| `configs/` | `sft_qwen3_8b.yaml`, `grpo_qwen3_8b.yaml`, `accelerate/zero2.yaml`, `accelerate/zero3.yaml` |
| `scripts/` | `gen_data.sh`, `run_sft_4xa100.sh`, `run_grpo_4xa100.sh`, `run_eval.sh` |
| `data/pilot/` | Pilot: seeds, ~300 generated samples, baseline predictions/scores, tiers, exported SFT/GRPO files |

## 1. Install (GPU server, 4×A100)
```bash
git clone https://github.com/dekelcohen/relations_extraction_llm_finetune && cd relations_extraction_llm_finetune
conda env create -f envs/environment.yml          # python 3.11 + torch 2.13 / vllm 0.32 / trl 1.17.1 / transformers 5.16-5.19
conda activate re-train
pip install flash-attn==2.8.3.post2 --no-build-isolation   # optional; else set attn_implementation: sdpa
python -m pytest -q tests                          # offline unit tests (scorer, dates, markers)
huggingface-cli login                              # Qwen3 weights are public; needed for gated models / pushing
```
Without conda: `python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-train.txt && pip install -e .`
Unsloth (optional, separate env): `conda create -n re-unsloth python=3.11 -y && conda activate re-unsloth && pip install -r envs/requirements-unsloth.txt`.

Version notes: `vllm==0.32.0` pins `torch==2.13.0` and `transformers<5.20`; `trl==1.17.1` accepts `vllm<0.33`. If you change one, re-check the other two.

## 2. Data (any machine with internet, no GPU)
Needs `OPENROUTER_API_KEY` (env var, or `RE_ENV_FILE=/path/to/.env`) and [`llm_providers`](../llm_providers) on `PYTHONPATH`.
```bash
export PYTHONPATH=/path/to/llm_providers RE_ENV_FILE=/path/to/.env
N=1000 bash scripts/gen_data.sh      # seeds -> build -> predict weak/strong -> score -> mine tiers -> export
# step by step: python -m re_synth {seeds,build,predict,score,mine,export,show} --help
```
Outputs `data/synth/export/{sft,grpo}_{train,val,test}.jsonl` (chat messages; the chat template is applied at train time, Qwen3 with `enable_thinking=False`). LLM calls are cached in `cache/llm` (re-runs are free and resumable).
Public data: `python -m re_synth convert-redocred --inp Re-DocRED/data/train_revised.json --rel-info rel_info.json --out data/public/redocred.jsonl`, then pass it to `export --samples` together with the synthetic files.

## 3. Train (4×A100)
```bash
# Stage 1: LoRA SFT (ZeRO-2, 4 GPUs, eff. batch 64) + merge  -> outputs/sft-qwen3-8b-merged
bash scripts/run_sft_4xa100.sh
#   variants: BASE=Qwen/Qwen3.5-9B OUT=outputs/sft-q35 bash scripts/run_sft_4xa100.sh
#             ... --loss_type dft --use_liger_kernel false        (DFT loss)
#             ... --replay_dataset allenai/tulu-3-sft-mixture --replay_ratio 0.15   (anti-forgetting replay)
# Stage 2: GRPO with vLLM server on GPU 3, training on GPUs 0-2  -> outputs/grpo-qwen3-8b-merged
bash scripts/run_grpo_4xa100.sh
#   focus on mined hard samples: bash scripts/run_grpo_4xa100.sh --tiers hard,easy
```
Single GPU debug: `python train/sft.py --config configs/sft_qwen3_8b.yaml --max_train_samples 50 --max_steps 5`.
Larger models (Qwen3-32B / Qwen3.8-27B): use `configs/accelerate/zero3.yaml`, `per_device_train_batch_size: 1`.

## 4. Evaluate
```bash
bash scripts/run_eval.sh Qwen/Qwen3-8B                      # base model baseline
bash scripts/run_eval.sh outputs/sft-qwen3-8b-merged
bash scripts/run_eval.sh outputs/grpo-qwen3-8b-merged data/synth/export/grpo_test.jsonl --guided_json
bash eval/forgetting.sh Qwen/Qwen3-8B && bash eval/forgetting.sh outputs/grpo-qwen3-8b-merged   # general-skill regression
python train/merge.py --base Qwen/Qwen3-8B --adapter outputs/sft-qwen3-8b --out outputs/sft-wise08 --scale 0.8  # WiSE trade-off
```
Metrics: slot P/R/F1, record F1, null accuracy, decoy-hit rate (picked a distractor), hallucination rate (value not in text), JSON validity, exact-sample rate, grouped by language / format / hard case / tier.

## Status
Pilot (300 synthetic he/fa/en samples, Llama-3.3-70B vs Claude Sonnet baselines, tiers, export, CPU smoke tests of all training scripts): see [plan §10](docs/plans/synth_dataset_and_finetune.md#10-execution-status-pilot-run-2026-11-10). Pilot data is in `data/pilot/`. Current blocking step: manual review of the pilot testset for realism vs. private company docs, then hardening — see [docs/status.md](docs/status.md).
