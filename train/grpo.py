"""GRPO (RLVR) on top of the SFT model, reward = re_synth scorer (train/rewards.py).

Recommended 4xA100 (vLLM server on GPU 3, training on GPUs 0-2):
  CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model outputs/sft-merged --max_model_len 8192 &
  CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes 3 \
      train/grpo.py --config configs/grpo_qwen3_8b.yaml
Colocate alternative (all 4 GPUs train + generate): add --vllm_mode colocate --vllm_gpu_memory_utilization 0.3
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config  # noqa: E402

from data import read_jsonl, render_prompt  # noqa: E402
from rewards import extraction_reward, format_reward, slot_f1_reward  # noqa: E402

log = logging.getLogger("grpo")


@dataclass
class ScriptArgs:
    train_files: str = field(default="data/pilot/export/grpo_train.jsonl")
    eval_files: str | None = field(default=None)
    enable_thinking: bool = False
    tiers: str | None = field(default=None, metadata={"help": "comma list of meta.tier to keep, e.g. hard,easy"})
    max_train_samples: int | None = None
    max_eval_samples: int | None = 200
    max_prompt_tokens: int | None = field(default=7168, metadata={"help": "drop rows whose rendered prompt is longer"})


def _rows(paths, tok, args, limit):
    rows = read_jsonl(paths)
    if args.tiers:
        keep = set(args.tiers.split(","))
        rows = [r for r in rows if r.get("meta", {}).get("tier", "unknown") in keep]
    out = []
    for r in rows:
        p = render_prompt(tok, r["prompt"], args.enable_thinking)
        if args.max_prompt_tokens and len(tok(p, add_special_tokens=False)["input_ids"]) > args.max_prompt_tokens:
            continue
        out.append({"prompt": p, "gt": r["gt"], "decoys": r["decoys"], "text": r["text"], "slot_types": r["slot_types"]})
    if len(out) < len(rows):
        log.warning("dropped %d rows longer than %s prompt tokens", len(rows) - len(out), args.max_prompt_tokens)
    return Dataset.from_list(out[:limit])


def main():
    parser = TrlParser((ScriptArgs, GRPOConfig, ModelConfig))
    args, cfg, model_args = parser.parse_args_and_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = _rows(args.train_files, tok, args, args.max_train_samples)
    evald = _rows(args.eval_files, tok, args, args.max_eval_samples) if args.eval_files else None
    log.info("GRPO train rows=%d eval rows=%s", len(train), None if evald is None else len(evald))

    if model_args.dtype not in (None, "auto"):
        cfg.model_init_kwargs = {**(cfg.model_init_kwargs or {}), "dtype": model_args.dtype}
    if model_args.attn_implementation:
        cfg.model_init_kwargs = {**(cfg.model_init_kwargs or {}), "attn_implementation": model_args.attn_implementation}

    trainer = GRPOTrainer(model=model_args.model_name_or_path, args=cfg,
                          reward_funcs=[extraction_reward, format_reward, slot_f1_reward],
                          train_dataset=train, eval_dataset=evald, processing_class=tok,
                          peft_config=get_peft_config(model_args))
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    log.info("saved -> %s", cfg.output_dir)


if __name__ == "__main__":
    main()
