import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import QwenForCausalLM, greedy_generate

name = "Qwen/Qwen2.5-0.5B"
cfg = AutoConfig.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

mine = QwenForCausalLM(cfg).eval()
missing, unexpected = mine.load_state_dict(hf.state_dict(), strict=False)
print("missing:", missing)
print("unexpected:", unexpected)

ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    a, b = hf(ids).logits, mine(ids)
print("max abs diff:", (a - b).abs().max().item())
print("same argmax:", torch.equal(a.argmax(-1), b.argmax(-1)))

ref = hf.generate(ids, attention_mask=torch.ones_like(ids),
                  max_new_tokens=20, min_new_tokens=20, do_sample=False)
out = greedy_generate(mine, ids, 20)
print("generation matches:", torch.equal(ref, out))
print(tok.decode(out[0]))