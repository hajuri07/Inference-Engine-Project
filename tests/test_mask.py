import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import make_mask


def test_start_zero_equals_old_mask():
    T = 6
    old = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.equal(make_mask(T, 0, "cpu"), old)


def test_single_token_blocks_nothing():
    m = make_mask(1, 7, "cpu")
    assert m.shape == (1, 8)
    assert not m.any()


def test_example_from_notes():
    m = make_mask(2, 3, "cpu")
    expected = torch.tensor([[False, False, False, False, True],
                             [False, False, False, False, False]])
    assert torch.equal(m, expected)