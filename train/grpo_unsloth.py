"""Single-GPU (8 GB) GRPO on top of the SFT adapter with Unsloth + TRL GRPOTrainer (HF generate, no vLLM).

python train/grpo_unsloth.py --train data/stage_2_RL_GRPO/grpo_train_filtered.jsonl --out outputs/grpo-qwen3-4b \
    --max_steps 60 --max_completion 1024
The SFT adapter is loaded as the trainable policy; with beta=0 there is no reference model.
Hyperparameter rationale: docs/research/grpo_hyperparameters.md, docs/datasets/stage_2_RL_GRPO.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/sft-cot-qwen3-4b")
    ap.add_argument("--train", default="data/stage_2_RL_GRPO/grpo_train_filtered.jsonl")
    ap.add_argument("--out", default="outputs/grpo-qwen3-4b")
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--G", type=int, default=4, help="num_generations")
    ap.add_argument("--bs", type=int, default=1, help="per-device train batch (completions per micro-step)")
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--max_completion", type=int, default=1024)
    ap.add_argument("--max_prompt", type=int, default=2500)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--format_weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    from unsloth import FastLanguageModel  # before trl/transformers
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from data import read_jsonl, render_prompt
    from progress import ProgressCallback
    from rewards import extraction_reward, format_reward, slot_f1_reward, strict_extraction_reward

    model, tok = FastLanguageModel.from_pretrained(model_name=a.model, load_in_4bit=True, fast_inference=False,
                                                   max_seq_length=a.max_prompt + a.max_completion)
    rows = []
    for r in read_jsonl(a.train):
        p = render_prompt(tok, r["prompt"], True)
        if len(tok(p, add_special_tokens=False)["input_ids"]) <= a.max_prompt:
            rows.append({"prompt": p, "gt": r["gt"], "decoys": r["decoys"], "text": r["text"],
                         "slot_types": r["slot_types"]})
    ds = Dataset.from_list(rows).shuffle(seed=a.seed)
    # generate one prompt's G completions at a time (8 GB), accumulate over prompts_per_step prompts per update
    ga = a.prompts_per_step * a.G // a.bs
    print(f"train prompts={len(ds)} steps={a.max_steps} completions/step={a.bs * ga} (G={a.G})")

    cfg = GRPOConfig(
        output_dir=a.out, use_vllm=False, per_device_train_batch_size=a.bs, gradient_accumulation_steps=ga,
        num_generations=a.G, steps_per_generation=a.G // a.bs, max_completion_length=a.max_completion, temperature=1.0, top_p=0.95,
        beta=a.beta, loss_type="dr_grpo", scale_rewards="batch", mask_truncated_completions=False,
        reward_weights=[1.0, a.format_weight, 0.0, 0.0], learning_rate=a.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=max(1, round(0.05 * a.max_steps)), max_steps=a.max_steps, bf16=True,
        gradient_checkpointing=True, logging_steps=1, save_steps=a.save_steps, save_total_limit=20,
        report_to="none", seed=a.seed, optim="adamw_8bit", max_grad_norm=1.0)
    trainer = GRPOTrainer(model=model, processing_class=tok, args=cfg,
                          reward_funcs=[strict_extraction_reward, format_reward, extraction_reward, slot_f1_reward], train_dataset=ds,
                          callbacks=[ProgressCallback(os.path.join(a.out, "progress"), title="GRPO " + os.path.basename(a.out))])
    trainer.train(resume_from_checkpoint=a.resume)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
