import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import QwenForCausalLM, KVCache

name, dev, dt = "Qwen/Qwen2.5-0.5B", "cuda", torch.float16
cfg = AutoConfig.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(dev).eval()
mine = QwenForCausalLM(cfg).to(dt).to(dev).eval()
mine.load_state_dict(hf.state_dict(), strict=False)

text = "The KV cache stores keys and values so that each new token only needs one forward step. " * 3
ids = tok(text, return_tensors="pt").input_ids.to(dev)
n = ids.shape[1]
k = n // 2

with torch.no_grad():
    full = mine(ids).float()
    ref = hf(ids).logits.float()
    cache = KVCache(24, 1, 2, n, 64, dt, dev)
    parts = [mine(ids[:, :k], cache=cache, start_pos=0)]
    for i in range(k, n):
        parts.append(mine(ids[:, i:i + 1], cache=cache, start_pos=cache.length))
    cached = torch.cat(parts, dim=1).float()


def report(label, a, b):
    d = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"{label:22s} max diff {d:.4f}   argmax agree {agree:.3f}")


print("tokens:", n)
report("cached vs uncached", cached, full)
report("uncached vs HF", full, ref)
report("cached vs HF", cached, ref)