"""Batch inference on an exported test file (grpo_test.jsonl format) + scoring.

vLLM (GPU):   python eval/predict.py --model outputs/sft-merged --data data/synth/export/grpo_test.jsonl \
                  --out outputs/eval/sft.preds.jsonl [--lora outputs/grpo-lora] [--tp 1] [--guided_json]
HF (CPU/dbg): python eval/predict.py --backend hf --model <path> --data ... --limit 5
Then scores: writes <out>.metrics.json and prints a markdown table grouped by lang/format/hard-case/tier.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))

from data import read_jsonl, render_prompt  # noqa: E402
from re_synth.score import aggregate, parse_prediction, score_sample  # noqa: E402

log = logging.getLogger("predict")
OUT_SCHEMA = {"type": "object", "properties": {"records": {"type": "array", "items": {"type": "object"}}},
              "required": ["records"]}


def gen_vllm(a, prompts):
    from vllm import LLM, SamplingParams
    kw = {}
    if a.guided_json:
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=OUT_SCHEMA)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=OUT_SCHEMA)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, **kw)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, enable_lora=bool(a.lora),
              max_lora_rank=a.max_lora_rank, gpu_memory_utilization=a.gpu_mem)
    lreq = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lreq = LoRARequest("adapter", 1, a.lora)
    outs = llm.generate(prompts, sp, lora_request=lreq)
    return [o.outputs[0].text for o in outs]


def gen_hf(a, prompts, tok):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora)
    res = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)
        out = model.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False)
        res.append(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return res


def score_rows(rows, preds):
    per, groups = {}, {}
    for r in rows:
        recs, ok = parse_prediction(preds.get(r["id"]))
        m = score_sample(recs, ok, json.loads(r["gt"]), json.loads(r["decoys"]), r["text"], json.loads(r["slot_types"]))
        per[r["id"]] = m
        meta = r.get("meta", {})
        keys = ["all"] + [f"lang={meta.get('lang')}", f"format={meta.get('format')}", f"tier={meta.get('tier')}"] + \
               [f"hc={h}" for h in meta.get("hard_cases", [])]
        for k in keys:
            groups.setdefault(k, []).append(m)
    return per, {k: aggregate(v) for k, v in sorted(groups.items(), key=lambda x: (x[0] != "all", x[0]))}


def to_md(rep):
    cols = ["n", "slot_p", "slot_r", "slot_f1", "rec_f1", "null_acc", "decoy_hit_rate", "halluc_rate", "json_valid",
            "exact_sample"]
    lines = ["| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {g} | " + " | ".join("" if m.get(c) is None else str(m[c]) for c in cols) + " |" for g, m in rep.items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=12288)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--guided_json", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.lora or a.model)
    rows = read_jsonl(a.data)[: a.limit]
    prompts = [render_prompt(tok, r["prompt"], a.enable_thinking) for r in rows]
    log.info("generating %d prompts with %s (%s)", len(prompts), a.model, a.backend)
    texts = gen_vllm(a, prompts) if a.backend == "vllm" else gen_hf(a, prompts, tok)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({"id": r["id"], "model": a.model, "lora": a.lora, "raw": t}, ensure_ascii=False) + "\n")
    per, rep = score_rows(rows, {r["id"]: t for r, t in zip(rows, texts)})
    with open(a.out + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump({"report": rep, "per_sample": per}, f, indent=1)
    print(to_md(rep))


if __name__ == "__main__":
    main()
