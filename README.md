# LLM inference engine from scratch (Qwen2.5-0.5B)

A small LLM inference engine written from scratch in PyTorch for **Qwen2.5-0.5B**, built to understand how inference actually works: the model itself, the KV cache, batching, quantization, and a custom kernel.

This is a learning project. It is not a replacement for Hugging Face Transformers or vLLM, and it is slower than HF in several places (see the results). The point is a correct, tested, measured engine that I can explain line by line.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Own model that matches Hugging Face | Done |
| 1 | KV cache | Done |
| 2 | Batching and continuous batching | Planned |
| 3 | int8 quantization | Planned |
| 4 | One custom kernel (Triton or CUDA) | Planned |
| 5 | FastAPI streaming server, final benchmarks, write-up | Planned |

Rules I keep for every phase: correctness tests, a baseline, and the write-up are never cut.

## Results so far (Phase 1)

Setup: Qwen2.5-0.5B, fp16, RTX 4050 6GB laptop, WSL2, batch size 1, 128 new tokens, median of 3. Full method and caveats are in [benchmarks/README.md](benchmarks/README.md).

Decode speed in tokens per second (higher is better):

| Prompt tokens | HF with cache | HF without cache | Mine, no cache | Mine, with cache |
|---|---|---|---|---|
| 7 | 64.4 | 66.6 | 53.6 | 54.2 |
| 482 | 65.7 | 27.7 | 21.2 | 52.5 |
| 2000 | 61.4 | 8.0 | 2.1 | 46.8 |

What this shows:

- **The cache works.** My cached decode stays at about 47 to 54 tok/s from 7 to 2000 prompt tokens. My own naive loop falls from 54 to 2.1 tok/s over the same range, so the cache is about 22x faster at 2000 tokens.
- **Part of that 22x is my naive loop being wasteful.** HF's own cache gain at 2000 tokens is 7.7x (8.0 to 61.4 tok/s). Both numbers are reported on purpose.
- **Decode speed is about 76 to 84% of HF** in plain PyTorch.
- **Prefill is the weak spot.** Time to first token at 2000 tokens is 343 ms for mine vs 126 ms for HF (about 2.7x slower). My likely explanation is that attention builds the full (14, T, S) score matrix. I have not profiled this yet, so it is a hypothesis.
- Run-to-run noise on this laptop is roughly 5 to 20%, so treat small differences as noise.

## Correctness

Every piece has a test that was written to prove it, not just to pass.

| Test file | What it proves |
|---|---|
| `tests/test_logits.py` | Own model vs Hugging Face in fp32: max logit diff about 2.6e-5, same argmax, greedy output matches HF up to and including the end token |
| `tests/test_kvcache.py` (8 tests) | Cache writes the right slots, keeps old data, layers do not overwrite each other, overflow raises, reset works |
| `tests/test_mask.py` (3 tests) | Causal mask for cached attention: `start_pos=0` equals the old mask, one new token blocks nothing, hand-checked example |
| `tests/test_cache_equivalence.py` (3 tests) | On a tiny random model: prefill then one-token decode gives the same logits as the uncached run, chunked prefill matches, last-position-only logits match |
| `tests/test_cached_vs_hf.py` | On the real Qwen weights (fp32, CPU): cached greedy output equals HF `generate`, token for token |
| `tests/check_fp16_gpu.py` | fp16 on GPU (55 tokens): cached vs uncached max logit diff 0.078, about the same as the uncached vs HF noise floor of 0.089; argmax agreement 1.000. This is a small check, not a full proof. |

A lesson from the generation test: HF's `min_new_tokens` blocks the end token, while my loop does not. Compare both sides under the same stopping rule, or the outputs differ for reasons unrelated to the model.

## What is implemented

All in `model.py`:

- `RMSNorm`, `SwiGlU`, `ROPE` (with a `start_pos` offset), grouped-query `Attention` (14 query heads, 2 KV heads), `DecoderLayer`, `QwenBody`, `QwenForCausalLM`
- `KVCache`: one K and one V tensor per layer, shape `(batch, 2, max_len, 64)`. `update()` writes but does not advance, `advance()` is called once per forward pass after all 24 layers, `reset()` empties it.
- `make_mask(T, start_pos, device)`: causal mask of shape `(T, start_pos + T)` for cached attention
- `greedy_generate`: the naive loop (no cache), kept as a baseline
- `generate_cached`: prefill the prompt once, then decode one token at a time, stopping at the end token

Design decisions worth knowing:

- **K is stored after RoPE and before the 2-to-14 head expansion.** Storing after RoPE means old keys never need re-rotating. Storing before expansion keeps the cache at **12,288 bytes per token** (24 layers x 2 for K and V x 2 KV heads x 64 dims x 2 bytes in fp16). Storing expanded heads would be 7x bigger.
- **Last-position-only logits** (`logits_last_only=True`) skip the large `lm_head` multiply for prompt positions that are never used.
- The cache is allocated in the same dtype and device as the model, so memory numbers are real.

## Known limitations

- Batch size 1 only (Phase 2 changes this).
- Greedy decoding only, no sampling.
- Attention is plain PyTorch ops: it builds the full score matrix, which makes prefill slow and memory-hungry for long prompts.
- Each layer rebuilds the RoPE cos and sin tables on every call. Small cost, not optimized yet.
- Tested on one model (Qwen2.5-0.5B) and one machine.

## How to run

Environment used: Python 3.14, transformers 5.21.1, PyTorch (fill in your version), Ubuntu on WSL2.

```
python -m venv ~/envs/engine
source ~/envs/engine/bin/activate
pip install torch transformers pytest
```

Run the fast tests (CPU, a few seconds):

```
python -m pytest tests/test_kvcache.py tests/test_mask.py tests/test_cache_equivalence.py -v
```

Run the real-model checks (downloads Qwen2.5-0.5B, needs about 4 GB of RAM):

```
python tests/test_logits.py
python -m pytest tests/test_cached_vs_hf.py -v
```

Run the fp16 GPU check and the benchmark (needs a CUDA GPU):

```
python tests/check_fp16_gpu.py
python benchmarks/bench_phase1.py
```

## Repository layout

```
model.py                     the model, KV cache, mask, generation loops
tests/                       correctness tests and checks
benchmarks/bench_phase1.py   Phase 1 benchmark script
benchmarks/results.md        saved results
benchmarks/README.md         benchmark method, tables, caveats
```

## Roadmap

- **Phase 2:** batching with padding and per-sequence lengths, then continuous batching (a scheduler that swaps finished sequences for new ones). This needs a redesign of the cache's shared `length`.
- **Phase 3:** int8 weight quantization, with accuracy and memory measured.
- **Phase 4:** one custom kernel in Triton or CUDA. A fused attention is a candidate, since prefill attention is the current weak spot, but I will profile first.
- **Phase 5:** FastAPI streaming server, final benchmarks, and a full write-up.

## Acknowledgements

Model architecture and weights: Qwen2.5-0.5B by the Qwen team. Reference implementation used for comparison: Hugging Face Transformers.