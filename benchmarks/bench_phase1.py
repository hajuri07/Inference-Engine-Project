import sys, pathlib, time, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from model import QwenForCausalLM, generate_cached

name, dev, dtype, N_NEW = "Qwen/Qwen2.5-0.5B", "cuda", torch.float16, 128
cfg = AutoConfig.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
mine = QwenForCausalLM(cfg).to(dtype).to(dev).eval()
mine.load_state_dict(hf.state_dict(), strict=False)


@torch.no_grad()
def naive_last(model, ids, n_new):
    for _ in range(n_new):
        nxt = model(ids, logits_last_only=True)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


engines = {
    "HF cache on": lambda ids, n: hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=True),
    "HF cache off": lambda ids, n: hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=False),
    "mine naive (last-only)": lambda ids, n: naive_last(mine, ids, n),
    "mine cached": lambda ids, n: generate_cached(mine, ids, n, eos_id=None),
}


def timed(fn, ids, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        fn(ids, n)
    torch.cuda.synchronize()
    return time.perf_counter() - t


print("| prompt | engine | tok/s | TTFT ms | extra peak MB |")
print("|---|---|---|---|---|")
for plen in (7, 482, 2000):
    torch.manual_seed(0)
    ids = torch.randint(1000, 20000, (1, plen), device=dev)
    for ename, fn in engines.items():
        timed(fn, ids, 4)                                   # warmup
        times = [timed(fn, ids, N_NEW) for _ in range(3)]
        ttft = statistics.median([timed(fn, ids, 1) for _ in range(3)])
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        timed(fn, ids, N_NEW)
        peak = (torch.cuda.max_memory_allocated() - base) / 1e6
        print(f"| {plen} | {ename} | {N_NEW / statistics.median(times):.1f} | {ttft * 1000:.0f} | {peak:.0f} |")