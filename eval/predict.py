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
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))

from data import read_jsonl, render_prompt  # noqa: E402
from re_synth import chunking  # noqa: E402
from re_synth.score import aggregate, parse_prediction, score_sample  # noqa: E402

log = logging.getLogger("predict")
OUT_SCHEMA = {"type": "object", "properties": {"records": {"type": "array", "items": {"type": "object"}}},
              "required": ["records"]}


_LLM = {}  # engine reused across datasets in one process


def _pipe_log(msg):
    if os.environ.get("PIPE_LOG"):
        with open(os.environ["PIPE_LOG"], "a") as f:
            f.write(time.strftime("[%Y-%m-%d %H:%M:%S]") + f" STAGE eval-set INFO {msg}\n")


def _init_watchdog(sec):
    """Fail fast if engine init hangs: kill the engine-core children and exit(3) with a greppable error line."""
    import threading

    def boom():
        print(f"TimeoutError: vLLM engine init exceeded {sec:.0f}s (ENGINE_INIT_TIMEOUT)", file=sys.stderr, flush=True)
        try:
            import psutil
            for c in psutil.Process().children(recursive=True):
                c.kill()
        except Exception:  # noqa: BLE001
            pass
        os._exit(3)
    t = threading.Timer(sec, boom)
    t.daemon = True
    t.start()
    return t


def gen_vllm(a, prompts, cache_path=None):
    """Generate with one shared engine. With cache_path: stream requests through the engine, append every finished
    prompt to cache_path (resume: prompts already there are skipped) and log throughput/ETA periodically."""
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
                        n=getattr(a, "n", 1), **kw)
    extra = {k: v for k, v in {"quantization": a.quantization, "kv_cache_dtype": a.kv_cache_dtype,
                               "max_num_seqs": a.max_num_seqs, "enforce_eager": a.enforce_eager}.items() if v}
    extra |= json.loads(getattr(a, "engine_kw", None) or "{}")
    if "llm" not in _LLM:
        t0 = time.time()
        wd = _init_watchdog(float(os.environ.get("ENGINE_INIT_TIMEOUT", 1500)))
        _LLM["llm"] = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, enable_lora=bool(a.lora),
                          max_lora_rank=a.max_lora_rank, gpu_memory_utilization=a.gpu_mem, **extra)
        wd.cancel()
        _pipe_log(f"engine loaded in {time.time() - t0:.0f}s ({a.model}{' +lora' if a.lora else ''}; {extra})")
    llm = _LLM["llm"]
    lreq = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lreq = LoRARequest("adapter", 1, a.lora)
    if getattr(a, "n", 1) > 1:  # prefilter: all n samples per prompt
        return [[c.text for c in o.outputs] for o in llm.generate(prompts, sp, lora_request=lreq)]
    if not cache_path:
        return [o.outputs[0].text for o in llm.generate(prompts, sp, lora_request=lreq)]
    import hashlib
    key = lambda p: hashlib.sha1(f"{a.model}|{a.lora}|{a.max_tokens}|{p}".encode()).hexdigest()  # noqa: E731
    keys = [key(p) for p in prompts]
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            try:
                d = json.loads(line)
                done[d["k"]] = d["text"]
            except (json.JSONDecodeError, KeyError):  # truncated last line after a kill
                pass
    todo = [i for i, k in enumerate(keys) if k not in done]
    name = os.path.basename(cache_path).replace(".partial.jsonl", "")
    if len(todo) < len(prompts):
        _pipe_log(f"{name} resume: {len(prompts) - len(todo)}/{len(prompts)} prompts cached")
    eng = llm.llm_engine
    for i in todo:
        eng.add_request(f"{name}-{i}", prompts[i], sp, lora_request=lreq)
    t0 = last = time.time(); ntok = nfin = 0; every = float(os.environ.get("PROGRESS_EVERY", 300))
    with open(cache_path, "a", encoding="utf-8") as f:
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not o.finished:
                    continue
                i = int(o.request_id.rsplit("-", 1)[1]); c = o.outputs[0]
                done[keys[i]] = c.text; nfin += 1; ntok += len(c.token_ids)
                f.write(json.dumps({"k": keys[i], "text": c.text, "ntok": len(c.token_ids),
                                    "finish": c.finish_reason}, ensure_ascii=False) + "\n"); f.flush()
            if time.time() - last > every and nfin:
                last = time.time(); dt = last - t0; left = len(todo) - nfin
                msg = (f"{name} progress {len(prompts) - left}/{len(prompts)} prompts, {nfin * 60 / dt:.1f} prompts/min, "
                       f"{ntok / dt:.0f} gen tok/s (finished reqs), avg {ntok / nfin:.0f} tok/prompt, "
                       f"ETA {left * dt / nfin / 60:.0f} min, running={eng.get_num_unfinished_requests()}")
                log.info(msg); _pipe_log(msg)
    return [done[k] for k in keys]


def gen_hf(a, prompts, tok):
    import torch
    if not prompts:
        return []
    model = getattr(a, "_hf_model", None)
    if model is None:
        start = time.monotonic()
        if a.backend == "unsloth":
            from unsloth import FastLanguageModel
            model, _ = FastLanguageModel.from_pretrained(
                a.lora or a.model, max_seq_length=a.max_model_len,
                load_in_4bit=a.quantization == "bitsandbytes", fast_inference=False)
            FastLanguageModel.for_inference(model)
            if a.lora and not any(p.detach().count_nonzero().item()
                                  for name, p in model.named_parameters() if "lora_B" in name):
                raise ValueError(f"No nonzero LoRA B weights loaded from {a.lora}")
        else:
            from transformers import AutoModelForCausalLM
            kw = {"dtype": torch.bfloat16}
            if a.quantization == "bitsandbytes":
                from transformers import BitsAndBytesConfig
                kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)}
            model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", **kw)
            if a.lora:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, a.lora)
                if not any(p.detach().count_nonzero().item()
                           for name, p in model.named_parameters() if "lora_B" in name):
                    raise ValueError(f"No nonzero LoRA B weights loaded from {a.lora}; check the model architecture")
        a._hf_model = model.eval()
        devices = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})
        _pipe_log(f"HF model loaded in {time.monotonic() - start:.0f}s; device map: {devices}")
    device = model.get_input_embeddings().weight.device
    batch_size = max(1, a.hf_batch_size)
    tok.padding_side = "left"
    if batch_size > 1 and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    res = []
    start = last = time.monotonic()
    every = float(os.environ.get("PROGRESS_EVERY", 300))
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        ids = tok(batch, return_tensors="pt", add_special_tokens=False, padding=len(batch) > 1).to(device)
        out = model.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False)
        res.extend(tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True))
        now = time.monotonic()
        if i == 0 or now - last >= every or len(res) == len(prompts):
            elapsed = max(now - start, 1e-6)
            msg = f"HF progress {len(res)}/{len(prompts)} prompts, {len(res) * 60 / elapsed:.1f} prompts/min"
            log.info(msg)
            _pipe_log(msg)
            last = now
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


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True, help="repeatable: several datasets share one engine load")
    ap.add_argument("--out", action="append", required=True, help="one per --data")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--backend", choices=["vllm", "hf", "unsloth"], default="vllm")
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
    ap.add_argument("--hf_batch_size", type=int, default=1)
    ap.add_argument("--enforce_eager", action="store_true", help="skip CUDA graphs (saves VRAM on small GPUs)")
    ap.add_argument("--engine_kw", default=None, help='extra vLLM LLM(...) kwargs as JSON, e.g. {"language_model_only": true}')
    ap.add_argument("--chat_kw", default=None, help='extra chat-template kwargs as JSON, e.g. {"reasoning_effort": "medium"}')
    ap.add_argument("--guided_json", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_chunk", action="store_true", help="never split long documents")
    ap.add_argument("--chunk_chars", type=int, default=chunking.CHUNK_CHARS)
    ap.add_argument("--chunk_overlap", type=int, default=chunking.CHUNK_OVERLAP)
    ap.add_argument("--chunk_min", type=int, default=chunking.CHUNK_MIN)
    return ap


def main():
    a = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.backend == "unsloth":
        import unsloth  # noqa: F401 - apply patches before importing Transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.lora or a.model)
    assert len(a.data) == len(a.out), "--data/--out must pair up"
    for data, out in zip(a.data, a.out):
        run_set(a, tok, data, out)


def run_set(a, tok, data, out_path):
    t0 = time.time()
    rows = read_jsonl(data)[: a.limit]
    prompts, owner = [], []          # one prompt per document, or one per chunk of a long document
    for i, r in enumerate(rows):
        parts = [] if a.no_chunk else chunking.split(r["text"], a.chunk_chars, a.chunk_overlap, a.chunk_min)
        for c in parts or [None]:
            prompts.append(render_prompt(tok, chunk_prompt(r, c), a.enable_thinking, **json.loads(a.chat_kw or "{}")))
            owner.append(i)
    log.info("generating %d prompts for %d docs with %s (%s)", len(prompts), len(rows), a.model, a.backend)
    cache = out_path + ".partial.jsonl"
    outs = gen_vllm(a, prompts, cache) if a.backend == "vllm" else gen_hf(a, prompts, tok)
    per_doc = [[] for _ in rows]
    for i, t in zip(owner, outs):
        per_doc[i].append(t)
    texts = [o[0] if len(o) == 1 else chunking.merge_raws(o, json.loads(r["slot_types"]), r["text"]) for r, o in zip(rows, per_doc)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, t, o in zip(rows, texts, per_doc):
            row = {"id": r["id"], "model": a.model, "lora": a.lora, "raw": t} | ({"chunks": o} if len(o) > 1 else {})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    per, rep = score_rows(rows, {r["id"]: t for r, t in zip(rows, texts)})
    with open(out_path + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump({"report": rep, "per_sample": per}, f, indent=1)
    print(to_md(rep))
    ntrunc = ntok = 0
    if os.path.exists(cache):
        cl = [json.loads(l) for l in open(cache, encoding="utf-8") if l.strip().endswith("}")]
        ntrunc = sum(d.get("finish") == "length" for d in cl)
        ntok = sum(d.get("ntok", 0) for d in cl) / max(len(cl), 1)
        os.remove(cache)
    allr = rep["all"]  # one line per finished dataset in the pipeline log
    _pipe_log(f"{os.path.basename(out_path)} DONE slot_f1={allr.get('slot_f1')} exact={allr.get('exact_sample')} "
              f"json_valid={allr.get('json_valid')} hit_max_tokens={ntrunc}/{len(prompts)} avg_gen_tok={ntok:.0f} n={allr.get('n')} prompts={len(prompts)} "
              f"wall={time.time() - t0:.0f}s ({len(prompts) * 60 / (time.time() - t0):.1f} prompts/min)")


if __name__ == "__main__":
    main()
