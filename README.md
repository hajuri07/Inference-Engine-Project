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