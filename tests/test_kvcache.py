import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from model import KVCache


def make_cache():
    return KVCache(num_layers=2, batch=1, num_kv_heads=2, max_len=16,
                   head_dim=64, dtype=torch.float32, device="cpu")


def rand_kv(T):
    return torch.randn(1, 2, T, 64), torch.randn(1, 2, T, 64)


def test_first_write_returns_same_values():
    c = make_cache()
    k1, v1 = rand_kv(3)
    k, v = c.update(0, k1, v1)
    assert torch.equal(k, k1)
    assert torch.equal(v, v1)


def test_two_writes_keep_old_and_append_new():
    c = make_cache()
    k1, v1 = rand_kv(3)
    k2, v2 = rand_kv(1)
    c.update(0, k1, v1)
    c.advance(3)
    k, v = c.update(0, k2, v2)
    assert k.shape == (1, 2, 4, 64)
    assert torch.equal(k[:, :, 0:3], k1)   # old data kept
    assert torch.equal(k[:, :, 3:4], k2)   # new data after it
    assert torch.equal(v[:, :, 0:3], v1)
    assert torch.equal(v[:, :, 3:4], v2)


def test_layers_do_not_mix():
    c = make_cache()
    k0, v0 = rand_kv(3)
    k1, v1 = rand_kv(3)
    c.update(0, k0, v0)
    c.update(1, k1, v1)
    assert torch.equal(c.k_cache[0][:, :, :3], k0)
    assert torch.equal(c.v_cache[0][:, :, :3], v0)
    assert torch.equal(c.k_cache[1][:, :, :3], k1)
    assert torch.equal(c.v_cache[1][:, :, :3], v1)


def test_no_advance_means_overwrite():
    c = make_cache()
    ka, va = rand_kv(3)
    kb, vb = rand_kv(3)
    c.update(0, ka, va)
    k, v = c.update(0, kb, vb)   # no advance in between
    assert torch.equal(k, kb)
    assert torch.equal(v, vb)


def test_update_overflow_raises():
    c = make_cache()
    k, v = rand_kv(17)           # max_len is 16
    with pytest.raises(ValueError):
        c.update(0, k, v)


def test_advance_overflow_raises():
    c = make_cache()
    with pytest.raises(ValueError):
        c.advance(17)


def test_update_overflow_after_advance_raises():
    c = make_cache()
    c.advance(15)
    k, v = rand_kv(2)            # 15 + 2 = 17 > 16
    with pytest.raises(ValueError):
        c.update(0, k, v)


def test_reset():
    c = make_cache()
    c.advance(3)
    c.reset()
    assert c.length == 0
    k1, v1 = rand_kv(3)
    k, v = c.update(0, k1, v1)
    assert k.shape == (1, 2, 3, 64)   # wrote at position 0 again
    assert torch.equal(k, k1)