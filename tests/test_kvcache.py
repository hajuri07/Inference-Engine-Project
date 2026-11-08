import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from model import KVCache

def make_cache():
    return KVCache(num_layers=2, batch=1, num_kv_heads=2, max_len=16,
                   head_dim=64, dtype=torch.float32, device="cpu")

def test_first_write_returns_same_values():
    c = make_cache()
    k1 = torch.randn(1, 2, 3, 64)
    v1 = torch.randn(1, 2, 3, 64)
    k, v = c.update(0, k1, v1)
    assert torch.equal(k, k1)    # what I put in is what I get back
    assert torch.equal(v, v1)