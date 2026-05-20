# Raw Benchmark Results

**Hardware:** Apple M4 Max | **Inference:** vMLX | **Date:** 2026-05-20

Tests: Short generation / Medium generation / Long generation / Long prompt (prefill test)

---

## Granite 4.1 Family

### granite-4.1-8b-mxfp4

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 370 | 58.5 | 41 | 0.7 |
| Medium generation | 85 | 71.1 | 271 | 3.7 |
| Long generation | 129 | 69.8 | 333 | 7.5 |
| Long prompt (prefill) | 764 | 61.4 | 723 | 1.8 |
| **Average** | **337** | **65.2** | **342** | **13.7** |

### granite-4.1-30b-mxfp4

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 350 | 23.3 | 43 | 1.2 |
| Medium generation | 242 | 26.1 | 95 | 10.1 |
| Long generation | 389 | 25.8 | 111 | 20.2 |
| Long prompt (prefill) | 3026 | 23.5 | 182 | 5.4 |
| **Average** | **1002** | **24.7** | **108** | **36.9** |

### granite-4.1-30b-nvfp4

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 686 | 22.8 | 22 | 1.6 |
| Medium generation | 253 | 23.3 | 91 | 11.2 |
| Long generation | 405 | 24.0 | 106 | 21.8 |
| Long prompt (prefill) | 3865 | 21.9 | 180 | 5.8 |
| **Average** | **1102** | **23.0** | **100** | **40.5** |

### granite-4.1-30b-4bit

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 1029 | 20.2 | 15 | 1.9 |
| Medium generation | 254 | 22.0 | 91 | 11.9 |
| Long generation | 401 | 22.0 | 107 | 23.7 |
| Long prompt (prefill) | 3089 | 20.1 | 179 | 5.9 |
| **Average** | **1193** | **21.1** | **98** | **43.4** |

#### granite-4.1-30b variant comparison

| Variant | Avg TTFT | Avg TPS | PP t/s (prefill) | Total |
|---|---|---|---|---|
| mxfp4 | **1002ms** | **24.7** | **182** | **36.9s** |
| nvfp4 | 1102ms | 23.0 | 180 | 40.5s |
| 4bit | 1193ms | 21.1 | 179 | 43.4s |

---

## Gemma 4 Family

### gemma-4-26b-a4b-it-mxfp4 (MoE, ~4B active)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 1423 | 45.0 | 14 | 1.4 |
| Medium generation | 2548 | 100.5 | 11 | 2.5 |
| Long generation | 5075 | 100.9 | 9 | 5.1 |
| Long prompt (prefill) | 1898 | 67.4 | 290 | 1.9 |
| **Average** | **2736** | **78.4** | **81** | **10.9** |

### gemma-4-31b-it-mxfp4 (Dense 31B)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 3051 | 21.0 | 7 | 3.1 |
| Medium generation | 10651 | 24.0 | 3 | 10.7 |
| Long generation | 21746 | 23.5 | 2 | 21.7 |
| Long prompt (prefill) | 9392 | 13.6 | 59 | 9.4 |
| **Average** | **11210** | **20.5** | **17** | **44.8** |

---

## Qwen 3 Family

### Qwen3.6-27B-mxfp4 (Dense 27B)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 2814 | 22.7 | 4 | 2.8 |
| Medium generation | 8937 | 28.6 | 2 | 8.9 |
| Long generation | 17470 | 29.3 | 2 | 17.5 |
| Long prompt (prefill) | 6924 | 18.5 | 79 | 6.9 |
| **Average** | **9036** | **24.8** | **22** | **36.1** |

### Qwen3.6-35B-A3B-4bit (MoE, ~3B active)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 1374 | 46.6 | 9 | 1.4 |
| Medium generation | 2499 | 102.4 | 8 | 2.5 |
| Long generation | 4984 | 102.7 | 8 | 5.0 |
| Long prompt (prefill) | 1794 | 71.3 | 307 | 1.8 |
| **Average** | **2663** | **80.8** | **83** | **10.7** |

### Qwen3.6-35B-A3B-UD-MXFP4_K_XL (MoE, ~3B active)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 4155 | 15.4 | 3 | 4.2 |
| Medium generation | 330* | 77.5 | 6 | 3.3 |
| Long generation | 6506 | 78.7 | 6 | 6.5 |
| Long prompt (prefill) | 2235 | 57.3 | 246 | 2.2 |
| **Average** | **4050** | **57.2** | **65** | **16.2** |

*330ms medium TTFT looks anomalous — possible 3300ms misread from screenshot.

### Qwen3.6-35B-A3B-OptiQ-4bit (MoE, ~3B active)

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 2793 | 22.9 | 6 | 2.8 |
| Medium generation | 3625 | 70.6 | 7 | 3.6 |
| Long generation | 7156 | 71.5 | 6 | 7.2 |
| Long prompt (prefill) | 2618 | 48.9 | 212 | 2.6 |
| **Average** | **4048** | **53.5** | **58** | **16.2** |

### Qwen3.6-35B-A3B-MXFP4-MTP — JANGQ-AI

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 3386 | 18.9 | 4 | 3.4 |
| Medium generation | 9136 | 28.0 | 2 | 9.1 |
| Long generation | 17456 | 29.3 | 2 | 17.5 |
| Long prompt (prefill) | 5494 | 23.3 | 100 | 5.5 |
| **Average** | **8868** | **24.9** | **27** | **35.5** |

### Qwen3.6-35B-A3B-MXFP4-MTP — OsaurusAI

| Test | TTFT (ms) | TPS | PP t/s | Time (s) |
|---|---|---|---|---|
| Short generation | 3162 | 20.2 | 4 | 3.2 |
| Medium generation | 8545 | 30.0 | 2 | 8.5 |
| Long generation | 16741 | 30.6 | 2 | 16.7 |
| Long prompt (prefill) | 5621 | 22.8 | 98 | 5.6 |
| **Average** | **8517** | **25.9** | **27** | **34.1** |

---

## Full Comparison — Averages

| Model | Arch | Quant | Avg TTFT | Avg TPS | PP t/s (prefill) | Total (s) |
|---|---|---|---|---|---|---|
| granite-4.1-8b-mxfp4 | Dense 8B | MXFP4 | 337 | 65.2 | 723 | 13.7 |
| granite-4.1-30b-mxfp4 | Dense 30B | MXFP4 | 1002 | 24.7 | 182 | 36.9 |
| granite-4.1-30b-nvfp4 | Dense 30B | NVFP4 | 1102 | 23.0 | 180 | 40.5 |
| granite-4.1-30b-4bit | Dense 30B | INT4 | 1193 | 21.1 | 179 | 43.4 |
| gemma-4-26b-a4b-mxfp4 | MoE ~4B | MXFP4 | 2736 | 78.4 | 290 | 10.9 |
| gemma-4-31b-mxfp4 | Dense 31B | MXFP4 | 11210 | 20.5 | 59 | 44.8 |
| Qwen3.6-27B-mxfp4 | Dense 27B | MXFP4 | 9036 | 24.8 | 79 | 36.1 |
| Qwen3.6-35B-A3B-4bit | MoE ~3B | INT4 | 2663 | 80.8 | 307 | 10.7 |
| Qwen3.6-35B-A3B-UD-MXFP4_K_XL | MoE ~3B | Dyn MXFP4 | 4050 | 57.2 | 246 | 16.2 |
| Qwen3.6-35B-A3B-OptiQ-4bit | MoE ~3B | OptiQ INT4 | 4048 | 53.5 | 212 | 16.2 |
| Qwen3.6-35B-A3B-MXFP4-MTP (JANGQ) | MoE ~3B | MXFP4+MTP | 8868 | 24.9 | 100 | 35.5 |
| Qwen3.6-35B-A3B-MXFP4-MTP (Osaurus) | MoE ~3B | MXFP4+MTP | 8517 | 25.9 | 98 | 34.1 |
