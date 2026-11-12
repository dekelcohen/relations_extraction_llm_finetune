"""Batch inference on an exported test file (grpo_test.jsonl format) + scoring.

vLLM (GPU):   python eval/predict.py --model outputs/sft-merged --data data/synth/export/grpo_test.jsonl \
                  --out outputs/eval/sft.preds.jsonl [--lora outputs/grpo-lora] [--tp 1] [--guided_json]
HF (CPU/dbg): python eval/predict.py --backend hf --model <path> --data ... --limit 5
Then scores: writes <out>.metrics.json and prints a markdown table grouped by lang/format/hard-case/tier.
Long documents (> --chunk_min chars) are predicted per overlapping chunk and merged (re_synth/chunking.py);
--no_chunk disables this. Output rows of chunked docs also keep the per-chunk raw outputs ("chunks").
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
from re_synth import chunking  # noqa: E402
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
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens, seed=a.seed,
                        **kw)
    extra = {k: v for k, v in {"quantization": a.quantization, "kv_cache_dtype": a.kv_cache_dtype,
                               "max_num_seqs": a.max_num_seqs, "enforce_eager": a.enforce_eager}.items() if v}
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, enable_lora=bool(a.lora),
              max_lora_rank=a.max_lora_rank, gpu_memory_utilization=a.gpu_mem, **extra)
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


def chunk_prompt(r: dict, chunk: str | None) -> list[dict]:
    """The row's prompt with the document replaced by one chunk of it."""
    if chunk is None:
        return r["prompt"]
    doc = "<<<\n" + r["text"] + "\n>>>"
    msgs = [dict(m) for m in r["prompt"]]
    assert doc in msgs[-1]["content"], f"{r['id']}: document not found in prompt"
    msgs[-1]["content"] = msgs[-1]["content"].replace(doc, "<<<\n" + chunk + "\n>>>")
    return msgs


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
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantization", default=None, help="e.g. bitsandbytes (pre-quantized bnb-4bit checkpoints)")
    ap.add_argument("--kv_cache_dtype", default=None, help="e.g. fp8 to fit longer contexts on small GPUs")
    ap.add_argument("--max_num_seqs", type=int, default=None)
    ap.add_argument("--enforce_eager", action="store_true", help="skip CUDA graphs (saves VRAM on small GPUs)")
    ap.add_argument("--guided_json", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_chunk", action="store_true", help="never split long documents")
    ap.add_argument("--chunk_chars", type=int, default=chunking.CHUNK_CHARS)
    ap.add_argument("--chunk_overlap", type=int, default=chunking.CHUNK_OVERLAP)
    ap.add_argument("--chunk_min", type=int, default=chunking.CHUNK_MIN)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.lora or a.model)
    rows = read_jsonl(a.data)[: a.limit]
    prompts, owner = [], []          # one prompt per document, or one per chunk of a long document
    for i, r in enumerate(rows):
        parts = [] if a.no_chunk else chunking.split(r["text"], a.chunk_chars, a.chunk_overlap, a.chunk_min)
        for c in parts or [None]:
            prompts.append(render_prompt(tok, chunk_prompt(r, c), a.enable_thinking))
            owner.append(i)
    log.info("generating %d prompts for %d docs with %s (%s)", len(prompts), len(rows), a.model, a.backend)
    outs = gen_vllm(a, prompts) if a.backend == "vllm" else gen_hf(a, prompts, tok)
    per_doc = [[] for _ in rows]
    for i, t in zip(owner, outs):
        per_doc[i].append(t)
    texts = [o[0] if len(o) == 1 else chunking.merge_raws(o, json.loads(r["slot_types"]), r["text"]) for r, o in zip(rows, per_doc)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, t, o in zip(rows, texts, per_doc):
            row = {"id": r["id"], "model": a.model, "lora": a.lora, "raw": t} | ({"chunks": o} if len(o) > 1 else {})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    per, rep = score_rows(rows, {r["id"]: t for r, t in zip(rows, texts)})
    with open(a.out + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump({"report": rep, "per_sample": per}, f, indent=1)
    print(to_md(rep))


if __name__ == "__main__":
    main()
