
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from types import SimpleNamespace
from model import QwenForCausalLM, KVCache

cfg = SimpleNamespace(
    hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=100, num_hidden_layers=3,
    rms_norm_eps=1e-6, rope_parameters={"rope_theta": 10000.0},
    tie_word_embeddings=True,
)


def build():
    torch.manual_seed(0)
    model = QwenForCausalLM(cfg).eval()
    cache = KVCache(num_layers=3, batch=1, num_kv_heads=2, max_len=32,
                    head_dim=16, dtype=torch.float32, device="cpu")
    return model, cache


@torch.no_grad()
def test_prefill_then_decode_matches_uncached():
    model, cache = build()
    ids = torch.randint(0, 100, (1, 10))
    full = model(ids)
    out = model(ids[:, :6], cache=cache, start_pos=0)
    assert torch.allclose(out, full[:, :6], atol=1e-4)
    for i in range(6, 10):
        out = model(ids[:, i:i + 1], cache=cache, start_pos=cache.length)
        assert torch.allclose(out[:, 0], full[:, i], atol=1e-4), f"mismatch at position {i}"
    assert cache.length == 10


@torch.no_grad()
def test_chunked_prefill_matches_uncached():
    model, cache = build()
    ids = torch.randint(0, 100, (1, 9))
    full = model(ids)
    a = model(ids[:, :4], cache=cache, start_pos=0)
    b = model(ids[:, 4:9], cache=cache, start_pos=4)
    assert torch.allclose(torch.cat([a, b], dim=1), full, atol=1e-4)


@torch.no_grad()
def test_last_only_matches_full_last_row():
    model, _ = build()
    ids = torch.randint(0, 100, (1, 7))
    full = model(ids)
    last = model(ids, logits_last_only=True)
    assert last.shape == (1, 1, 100)
    assert torch.allclose(last[:, 0], full[:, -1], atol=1e-5)
