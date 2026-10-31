import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from model import Attention

attn = Attention().eval()

# 1. GQA: both versions must agree
kv = torch.randn(2, 2, 8, 64)
print(torch.equal(attn.expand_gqa_vector(kv), attn.expand_gqa_loop(kv)))

# 2. Causality: changing the last token must not change earlier outputs
x = torch.randn(1, 8, 896)
x2 = x.clone(); x2[:, -1] += 1.0
print(torch.allclose(attn(x)[:, :-1], attn(x2)[:, :-1], atol=1e-6))

# 3. fp16 runs without a dtype error
a16 = Attention().half().cuda().eval()
y = a16(torch.randn(1, 8, 896, dtype=torch.float16, device="cuda"))
print(y.dtype)