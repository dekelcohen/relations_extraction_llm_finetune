"""SFT (LoRA / full) for prompt-driven entity-relation extraction with TRL.

Single GPU:   python train/sft.py --config configs/sft_qwen3_8b.yaml
4xA100:       accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes 4 \
                  train/sft.py --config configs/sft_qwen3_8b.yaml
Any SFTConfig / ModelConfig field can be overridden on the CLI (e.g. --learning_rate 5e-5 --loss_type dft).
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_peft_config  # noqa: E402

from data import load_replay, mix, read_jsonl, to_prompt_completion  # noqa: E402

log = logging.getLogger("sft")


@dataclass
class ScriptArgs:
    train_files: str = field(default="data/pilot/export/sft_train.jsonl", metadata={"help": "comma-separated jsonl"})
    eval_files: str | None = field(default="data/pilot/export/sft_val.jsonl")
    enable_thinking: bool = field(default=False, metadata={"help": "Qwen3 template: keep the <think> block or not"})
    replay_dataset: str | None = field(default=None, metadata={"help": "general SFT data mixed in (anti-forgetting)"})
    replay_ratio: float = field(default=0.0, metadata={"help": "replay rows = ratio * task rows"})
    max_train_samples: int | None = None
    max_eval_samples: int | None = 300


def main():
    parser = TrlParser((ScriptArgs, SFTConfig, ModelConfig))
    args, cfg, model_args = parser.parse_args_and_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    task = read_jsonl(args.train_files)[: args.max_train_samples]
    replay = load_replay(args.replay_dataset, int(len(task) * args.replay_ratio), cfg.seed) \
        if args.replay_dataset and args.replay_ratio > 0 else []
    train = to_prompt_completion(mix(task, replay, cfg.seed), tok, args.enable_thinking)
    log.info("train rows: task=%d replay=%d -> %d", len(task), len(replay), len(train))
    log.info("example prompt tail:\n%s\nexample completion:\n%s", train[0]["prompt"][-400:], train[0]["completion"][:400])
    evald = None
    if args.eval_files and os.path.exists(args.eval_files.split(",")[0]):
        evald = Dataset.from_list(to_prompt_completion(read_jsonl(args.eval_files)[: args.max_eval_samples], tok,
                                                       args.enable_thinking))

    model_kwargs = dict(dtype=model_args.dtype, attn_implementation=model_args.attn_implementation,
                        trust_remote_code=True)
    cfg.model_init_kwargs = {**(cfg.model_init_kwargs or {}), **{k: v for k, v in model_kwargs.items() if v is not None}}

    trainer = SFTTrainer(model=model_args.model_name_or_path, args=cfg, train_dataset=Dataset.from_list(train),
                         eval_dataset=evald, processing_class=tok, peft_config=get_peft_config(model_args))
    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    log.info("saved -> %s", cfg.output_dir)


if __name__ == "__main__":
    main()
