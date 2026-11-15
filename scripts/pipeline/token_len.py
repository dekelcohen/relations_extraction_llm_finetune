"""Token-length stats of an SFT jsonl rendered with the base model's chat template (thinking on). Prints p99/max/max_len."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "train"))
from transformers import AutoTokenizer
from data import read_jsonl, to_prompt_completion

tok = AutoTokenizer.from_pretrained(sys.argv[1])
L = sorted(len(tok(p["prompt"] + p["completion"]).input_ids) for p in to_prompt_completion(read_jsonl(sys.argv[2]), tok, True))
print(f"n={len(L)} p50={L[len(L)//2]} p99={L[int(len(L)*.99)]} max={L[-1]}")
print(f"max_len {math.ceil((L[-1] + 16) / 512) * 512}")
