# Reproducing every training and eval run

The exact training and eval files (row counts and `sha256` prefixes as committed), and the command that produced each adapter and each
metrics file. Results live in [status.md](status.md); the data builders are documented in each `data/*/README.md`.

## 0. Setup
- **Data / API calls (no GPU):** `export PYTHONPATH=<path>/llm_providers RE_ENV_FILE=.env`. `.env` holds `OPENROUTER_API_KEY` and `HF_TOKEN` and is never committed.
  - `copilot-*` models (Sonnet 5.5, Opus, GPT) go through the Copilot CLI.
  - All other models (Llama-3.3-70B) go through OpenRouter.
- **GPU runs:** use `scripts/pipeline` with `configs/machines/{local3070,a100}.env` (docker container, envs, VRAM settings) and `configs/runs/*.env` (model, data, hyper-parameters).
  - Start a run: `bash scripts/pipeline/remote.sh --machine M start --run R [--stages "..."]`. The machine `local3070` runs locally (`sudo -n docker`).
  - The resolved config is written to `outputs/pipeline/M/config.log`.
- **Artifacts:** adapters are on the private HF Hub (`hf_artifacts.json`). `python scripts/hf_sync.py pull` restores them.

## 1. Training files

| file | rows | sha256[:12] | content / builder |
|---|---|---|---|
| `data/synth_hefa/sft_cot_train.jsonl` | 552 (he 278 / fa 274) | a7d7aa227266 | **Stage-1 SFT.** Synthetic he/fa samples (`gen_data.sh` with `LANGS=he,fa`) plus Sonnet 5.5 teacher CoT, kept only when verified against gold (`python -m re_synth teacher-cot --samples data/synth_hefa/samples_{he,fa}.jsonl --model copilot-claude-sonnet-5.5 --out data/synth_hefa/cot_{he,fa}`). CoT mean 651 chars. |
| `data/stage_2_RL_GRPO/grpo_train.jsonl` | 726 chunk rows (532 docs) | 1f6402ae9b35 | **GRPO prompt pool.** Stage-2 synthetic docs (600 he/fa, H13–H19, long docs), chunked. Split by `scripts/split_grpo.py`. |
| `data/stage_2_RL_GRPO/grpo_train_filtered.jsonl` | 191 (111 unique; hard/frontier ×2) | d6eb885cc8ee | **GRPO train.** `train/prefilter.py --data data/stage_2_RL_GRPO/grpo_train.jsonl --lora outputs/sft-cot-qwen3-4b --out_dir data/stage_2_RL_GRPO --subset 240` keeps prompts with reward std > 0 (n=4). `train/refilter.py --dir data/stage_2_RL_GRPO --max_len 1300` then keeps completions ≤ 1300 tokens. |
| `data/hard_inss/train.jsonl` | 147 (he 78 / fa 69) | ab4ea6f90d2d | INSS-hosted hard samples (`scripts/scrape_inss.py`, `build_hard_inss.py`, `build_hard_inss2.py`); see [README](../data/hard_inss/README.md). Used only through the sft_stage2 sets below. |
| `data/sft_stage2/train.jsonl` | 860 | 56e8742209f2 | Stage-2 Sonnet-CoT mix: 481 stage-2 CoT + 61 INSS teacher-CoT + 138 stage-1 replay (`scripts/build_sft_stage2.py`). **Built but not trained on:** replaced by the self-distill set because its CoT is too short. |
| `data/sft_stage2/train_selfdistill.jsonl` | 994 chunk rows: stage2_synth 798 / INSS 65 / replay 131 | 46efef3ae1cd | **Stage-2 SFT (27B and 4B).** The same 680 docs as `train.jsonl` (`scripts/selfdistill_prep.py` → `selfdistill_src.jsonl`) re-solved by **27B base INT4 at reasoning_effort=medium** (pipeline stage `selfdistill_gen`, run `qwen3.8-27b-stage2`, A100). Only docs with slot-F1 = 1 are kept (619/680, incl. a seed-1 retry on `selfdistill_src2.jsonl`), assembled by `scripts/build_selfdistill.py`. |

**Leakage guard:** every training set excludes ids, and texts with ≥0.5 8-gram overlap, from pilot, grpo_dev, test_long and hard_inss/dev (`build_sft_stage2.py`).

## 2. Eval files

| set | file | rows | sha256[:12] |
|---|---|---|---|
| pilot | `data/pilot/export/pilot_all.jsonl` | 298 | 6242b84be01f |
| grpo_dev | `data/stage_2_RL_GRPO/grpo_dev.jsonl` | 60 docs | 19f42516efa8 |
| test_long | `data/test_long/test_long.jsonl` | 80 docs | 672ec9ee4a57 |
| hard_inss | `data/hard_inss/dev.jsonl` | 39 rows (38 unique; `hard-inss-438720-05-fa` appears twice) | fc0af6473163 |

- Docs longer than `chunk_min` are predicted per chunk and merged (`re_synth/chunking.py`, default settings).
- Scoring uses `re_synth.score` (slot-F1). Each preds file `X.preds.jsonl` gets a sibling `X.preds.jsonl.metrics.json`.

## 3. Training runs

| run | base | init | data | hyper-params | command | adapter (HF) |
|---|---|---|---|---|---|---|
| 4B SFT stage 1 | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | none | `synth_hefa/sft_cot_train` | r32, 2 ep, lr 1e-4, bs1×ga16, max_len 4096, thinking | `python train/sft_unsloth.py --model unsloth/Qwen3-4B-unsloth-bnb-4bit --load_in_4bit --enable_thinking --train data/synth_hefa/sft_cot_train.jsonl --out outputs/sft-cot-qwen3-4b --max_len 4096 --bs 1 --ga 16 --epochs 2 --r 32` (3070; also `remote.sh --machine local3070 start --run qwen3-4b --stages sft`) | `outputs/sft-cot-qwen3-4b` |
| 4B GRPO | same | 4B SFT stage 1 | `grpo_train_filtered` | 20 steps, 2 prompts × G=4, lr 1e-5, β=0, dr_grpo, reward `strict_extraction` (1.0) + format (0.1), max completion 1024 | `bash scripts/grpo_retry.sh --max_steps 20 --prompts_per_step 2` (resumes after OOM; the other defaults in `train/grpo_unsloth.py` match). Best checkpoint by grpo_dev = final (`scripts/eval_grpo.sh`). | `Shani123/re-qwen3-4b-grpo` |
| 27B SFT stage 1 | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | none | `synth_hefa/sft_cot_train` | r32, 2 ep, lr 1e-4, bs1×ga16, max_len auto (= longest row) | `remote.sh --machine a100 start --run qwen3.8-27b --stages "setup_unsloth download_sft ; sft_smoke sft"` | `Shani123/re-qwen3.8-27b-sft-cot` |
| 27B SFT stage 2 | same | 27B stage 1 (weights only) | `sft_stage2/train_selfdistill` | r32, 2 ep (116 steps), lr 5e-5, warmup 0.05, bs1×ga16, max_len 8704, chat kw `reasoning_effort=medium` | `remote.sh --machine a100 start --run qwen3.8-27b-stage2 --stages "sft_smoke sft"` | `Shani123/re-qwen3.8-27b-sft-cot-stage2` |
| 4B SFT stage 2 (apples-to-apples with 27B stage 2) | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | 4B stage 1 | `sft_stage2/train_selfdistill` | as 27B stage 2, but max_len auto = 9728 and no `reasoning_effort` (the 4B template has none) | `remote.sh --machine local3070 start --run qwen3-4b-stage2` (stages `sft_smoke sft ft_eval`) | `outputs/sft-cot-qwen3-4b-stage2` |

Stage-2 runs load the stage-1 adapter weights with a fresh optimizer (`SFT_INIT_ADAPTER`). Known bug: a `--sft-epochs` CLI override is ignored when the run config sets `SFT_EPOCHS`.

## 4. Eval runs (output under `outputs/eval/`)

| model | engine settings | command |
|---|---|---|
| 4B base / SFT / GRPO | vLLM `Qwen/Qwen3-4B-AWQ` (+ LoRA), thinking, T 0.6, top_p 0.95, top_k 20, max_tokens 5120, max_model_len 8192, enforce_eager | `python eval/predict.py --enable_thinking --temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 5120 --max_model_len 8192 --gpu_mem 0.9 --enforce_eager --model Qwen/Qwen3-4B-AWQ --max_lora_rank 32 --lora <adapter> --data <set file> --out outputs/eval/<name>.<set>.preds.jsonl` (`scripts/eval_grpo.sh`). Names: `sft.*` / `qwen3-4b-ft-think` (SFT), `grpo-final.grpo_dev`, `grpo.test_long`, `grpo.hard_inss`, `qwen3-4b-grpo-think` (GRPO pilot). Via the pipeline: `--run qwen3-4b` / `qwen3-4b-stage2` stage `ft_eval` writes `<FT_NAME>.<set>`. |
| 27B base FP8 | `Qwen/Qwen3.8-27B-FP8`, thinking, max_tokens 5120 | `remote.sh --machine a100 start --run qwen3.8-27b --stages base_eval` with `EVAL_BASES=fp8` |
| 27B base / ft INT4 | `RedHatAI/Qwen3.8-27B-INT4`, `TRITON_ATTN`, bf16 KV, max_num_seqs 48, max_tokens 12288, max_model_len 16384, + LoRA for ft | `remote.sh --machine a100 start --run qwen3.8-27b[-stage2] --stages "base_eval ft_eval"`. The `-medium` names add `--chat_kw '{"reasoning_effort":"medium"}'` (the template default is xhigh). `eval/retry_hits.py` re-runs outputs that hit max_tokens at medium effort. |
| Sonnet 5.5 (strong) | Copilot CLI, chunked | `python -m re_synth predict --samples <samples> --model copilot-claude-sonnet-5.5 --out data/<set>/preds_strong.jsonl` |
| Llama-3.3-70B (weak) | OpenRouter, chunked | `python -m re_synth predict --samples <samples> --model meta-llama/llama-3.3-70b-instruct --out data/<set>/preds_weak.jsonl`. For hard_inss, `<samples>` = the dev ids taken from `data/hard_inss/pool.jsonl` (sample format). |

- Score API/CLI preds with `python -m re_synth score --samples <samples> --preds <preds> --out <metrics.json>`. Use `re_synth.mine.score_preds` + `re_synth.score.aggregate` for row-format sets (hard_inss: `outputs/eval/llama-3.3-70b.hard_inss.metrics.json`).
- Report: `bash scripts/build_report.sh` (comparison, error analysis, strong-gap) → `outputs/report/pilot_compare.html`.

## 5. New machine: 4×A100-80GB server (`configs/machines/a100x4.env`)
Host volume `/mnt/dekel` is mounted in the container as `/data` (repos in `/data/repos`, envs in `/data/envs`, HF cache in `/data/hf_cache`).
`docker/docker-compose.yml` is started by the pipeline with `gpus: all`, `network_mode: host` and `ipc: host`; `ipc: host` replaces the old `--shm-size=16g`, which was for torch DataLoader/NCCL shared memory.
The pipeline is single-GPU: `GPU_IDS=0` in the config. Env versions are pinned to the A100 runs: vLLM 0.33.0, Unsloth 2026.11.4, transformers 5.17.0, trl 1.15.0, peft 0.21.3, bitsandbytes 0.50.2.

```bash
# on the server host
mkdir -p /mnt/dekel/repos /mnt/dekel/envs /mnt/dekel/hf_cache && cd /mnt/dekel/repos
git clone https://github.com/dekelcohen/relations_extraction_llm_finetune.git
git clone https://github.com/dekelcohen/llm_providers.git        # only for API data/predict steps (re_synth)
cd relations_extraction_llm_finetune
printf 'HF_TOKEN=...\nOPENROUTER_API_KEY=...\n' > .env          # never commit
pip install -q huggingface_hub python-dotenv; set -a; . ./.env; set +a; python3 scripts/hf_sync.py pull        # adapters + large artifacts from HF (hf_artifacts.json)
R=scripts/pipeline/remote.sh; M="--machine a100x4"
bash $R $M start --run qwen3.8-27b --stages "setup_vllm setup_unsloth"   # build the image, create the pinned envs (once)
bash $R $M watchdog --deadline-h 8                                        # GPU-idle alerts + hard stop
bash $R $M wait                                                           # blocks until the next DONE/FAIL/ALERT
bash $R $M status
```

Reproduce each run with `bash $R $M start --run <R> --stages "<S>"` (§3 and §4 give the hyper-parameters):

| what | `--run` | `--stages` |
|---|---|---|
| 4B base eval + stage-1 SFT + ft eval | `qwen3-4b` | `download_eval download_sft ; base_eval sft_smoke sft ft_eval` |
| 4B stage-2 SFT (self-distill) + eval on 4 sets | `qwen3-4b-stage2` | `sft_smoke sft ft_eval` (needs `outputs/sft-cot-qwen3-4b`) |
| 27B base INT4 eval + stage-1 SFT + ft eval | `qwen3.8-27b` | `download_eval download_sft ; base_eval sft_smoke sft ft_eval` |
| 27B stage-2 data (self-distill traces) | `qwen3.8-27b-stage2` | `selfdistill_gen`, then `python scripts/build_selfdistill.py` |
| 27B stage-2 SFT + eval | `qwen3.8-27b-stage2` | `sft_smoke sft ft_eval` |
| 4B GRPO | none (in the container) | `bash scripts/grpo_retry.sh --max_steps 20 --prompts_per_step 2`, then `bash scripts/eval_grpo.sh` |

- **Fresh training:** `hf_sync pull` restores the finished adapters, and `sft` skips any adapter that already exists. Use a new `--sft-out` (or delete the adapter dir) to retrain.
- **Results:** `outputs/eval/*.metrics.json`, logs under `outputs/pipeline/a100x4/`. Afterwards: `bash $R $M watchdog-stop`, `docker stop relx-dev-1`, and check `nvidia-smi` is clear.
- **80 GB:** the 27B runs fit easily, and `MAX_NUM_SEQS=64` (vs 40/48 on the 40 GB A100) only changes throughput, not results.
