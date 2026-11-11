import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from model import QwenForCausalLM, generate_cached


def test_cached_greedy_matches_hf():
    name = "Qwen/Qwen2.5-0.5B"
    cfg = AutoConfig.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    mine = QwenForCausalLM(cfg).eval()
    missing, unexpected = mine.load_state_dict(hf.state_dict(), strict=False)
    assert missing == [] and unexpected == []

    eos = hf.generation_config.eos_token_id
    ids = tok("what is the capital of france", return_tensors="pt").input_ids
    ref = hf.generate(ids, attention_mask=torch.ones_like(ids),
                      max_new_tokens=20, do_sample=False)
    out = generate_cached(mine, ids, 20, eos_id=eos)
    assert torch.equal(ref, out), (ref.tolist(), out.tolist())