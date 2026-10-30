import time, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
print("transformers version:", transformers.__version__)

def bench(prompt, use_cache, runs=3, new_tokens=128):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    n_in = inputs["input_ids"].shape[1]
    gen = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
               do_sample=False, use_cache=use_cache)
    model.generate(**inputs, **gen)                      # warmup
    times = []
    for _ in range(runs):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n_new = out.shape[1] - n_in
    med = sorted(times)[len(times) // 2]
    print(f"prompt={n_in} tokens | cache={use_cache} | new={n_new} | "
          f"{n_new/med:.1f} tok/s | peak {torch.cuda.max_memory_allocated()/1e9:.3f} GB")

short = "Explain how a transformer works."
long = "Explain how a transformer works. " * 80

for p in (short, long):
    for c in (True, False):
        bench(p, c)