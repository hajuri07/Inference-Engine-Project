# What the script did

## Generation performance

| Prompt length | Cache | Throughput (tokens/s) | Peak memory (GB) |
| ---: | :---: | ---: | ---: |
| 7 tokens | On | 67.9 | 1.008 |
| 7 tokens | Off | 64.6 | 1.011 |
| 482 tokens | On | 67.8 | 1.024 |
| 482 tokens | Off | 25.3 | 1.024 |

### Test conditions

- **GPU:** RTX 4050 Laptop
- **Precision:** FP16
- **Generated tokens:** 128
- **Transformers:** 5.21.1
- **Measurement:** Median of 3 runs; laptop plugged in

## Benchmark procedure
## Naive V0

1. Loaded Hugging Face’s Qwen, copied its weights into **your** model, then deleted the Hugging Face copy to free memory.
2. Moved your model to the GPU in FP16, the same format as the earlier benchmark.
3. Generated 128 tokens with `greedy_generate`, your Phase 0 loop with **no cache**. It reprocesses the whole text for every new token.
4. Ran it 3 times and took the median, after one warmup run that isn’t timed. The `synchronize()` calls make the timer wait for the GPU to finish.
5. Did all of this for two prompts, one short (7 tokens) and one long (482 tokens).

## What the numbers mean

| Prompt length | Hugging Face, cache off (tokens/s) | Your naive loop (tokens/s) |
| ---: | ---: | ---: |
| 7 tokens | 64.6 | 44.3 |
| 482 tokens | 25.3 | 15.9 |
