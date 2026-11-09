"""Optional single-GPU LoRA SFT with Unsloth (faster / less memory). Separate env: envs/requirements-unsloth.txt.
Not exercised in CI/smoke tests (needs a CUDA GPU). Produces a PEFT adapter compatible with train/merge.py.

python train/sft_unsloth.py --model unsloth/Qwen3-8B --train data/synth/export/sft_train.jsonl --out outputs/sft-unsloth
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-8B")
    ap.add_argument("--train", default="data/synth/export/sft_train.jsonl")
    ap.add_argument("--out", default="outputs/sft-unsloth")
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true", help="QLoRA")
    a = ap.parse_args()

    from unsloth import FastLanguageModel  # must be imported before trl/transformers
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from data import read_jsonl, to_prompt_completion

    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=a.load_in_4bit)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r, lora_dropout=0.0, use_rslora=True, use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ds = Dataset.from_list(to_prompt_completion(read_jsonl(a.train), tok))
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.ga, max_length=a.max_len, completion_only_loss=True, bf16=True,
                    logging_steps=5, save_steps=200, lr_scheduler_type="cosine", warmup_ratio=0.03, report_to="none")
    SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
