"""Merge a LoRA adapter into its base (optionally WiSE-style scaled, i.e. interpolate base<->finetuned).

python train/merge.py --base Qwen/Qwen3-8B --adapter outputs/sft-qwen3-8b --out outputs/sft-merged [--scale 0.8]
--scale < 1 shrinks the LoRA delta (W = W0 + s*BA): cheap weight-space ensembling that trades a little task
accuracy for less forgetting; sweep 0.6-1.0 with eval/forgetting.sh.
"""
from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("merge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=getattr(torch, a.dtype), device_map="cpu")
    model = PeftModel.from_pretrained(model, a.adapter)
    if a.scale != 1.0:
        n = 0
        for mod in model.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                for k in mod.scaling:
                    mod.scaling[k] *= a.scale
                    n += 1
        log.info("scaled %d LoRA modules by %.3f", n, a.scale)
    model = model.merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(a.adapter)
    except Exception:
        tok = AutoTokenizer.from_pretrained(a.base)
    tok.save_pretrained(a.out)
    log.info("merged -> %s", a.out)


if __name__ == "__main__":
    main()
