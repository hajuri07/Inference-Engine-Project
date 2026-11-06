import sys, pathlib, time, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model import QwenForCausalLM, greedy_generate

name = "Qwen/Qwen2.5-0.5B"
cfg = AutoConfig.from_pretrained(name)
tok = AutoTokenizer.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
mine = QwenForCausalLM(cfg)
mine.load_state_dict(hf.state_dict(), strict=False)
del hf
mine = mine.half().cuda().eval()

def bench(prompt, new=128, runs=3):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    greedy_generate(mine, ids, new)                      # warmup
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        greedy_generate(mine, ids, new)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    print(f"prompt={ids.shape[1]} | {new / statistics.median(times):.1f} tok/s | "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")

bench("Explain how a transformer works.")
bench("Explain how a transformer works. " * 80)