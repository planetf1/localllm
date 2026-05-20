# Local LLM Benchmarks

**Hardware:** Apple M4 Max  
**Inference:** vMLX  
**Date:** 2026-05-20  
**Benchmark tool:** Built-in vMLX bench (Short / Medium / Long generation + Long prompt prefill test)

## Metrics

| Metric | Description | Lower/Higher better |
|---|---|---|
| TTFT | Time to First Token (ms) | Lower |
| TPS | Generation tokens/second | Higher |
| PP t/s | Prompt processing tokens/second | Higher |
| Time | Total wall time (s) | Lower |

## Models Tested

| # | Model | Family | Arch | Quant |
|---|---|---|---|---|
| 1 | granite-4.1-8b-mxfp4 | Granite 4.1 | Dense 8B | MXFP4 |
| 2 | granite-4.1-30b-mxfp4 | Granite 4.1 | Dense 30B | MXFP4 |
| 3 | granite-4.1-30b-nvfp4 | Granite 4.1 | Dense 30B | NVFP4 |
| 4 | granite-4.1-30b-4bit | Granite 4.1 | Dense 30B | INT4 |
| 5 | gemma-4-26b-a4b-it-mxfp4 | Gemma 4 | MoE ~4B active | MXFP4 |
| 6 | gemma-4-31b-it-mxfp4 | Gemma 4 | Dense 31B | MXFP4 |
| 7 | Qwen3.6-27B-mxfp4 | Qwen 3 | Dense 27B | MXFP4 |
| 8 | Qwen3.6-35B-A3B-4bit | Qwen 3 | MoE ~3B active | INT4 |
| 9 | Qwen3.6-35B-A3B-UD-MXFP4_K_XL | Qwen 3 | MoE ~3B active | Dynamic MXFP4 |
| 10 | Qwen3.6-35B-A3B-OptiQ-4bit | Qwen 3 | MoE ~3B active | OptiQ INT4 |
| 11 | Qwen3.6-35B-A3B-MXFP4-MTP (JANGQ-AI) | Qwen 3 | MoE ~3B active | MXFP4+MTP |
| 12 | Qwen3.6-35B-A3B-MXFP4-MTP (OsaurusAI) | Qwen 3 | MoE ~3B active | MXFP4+MTP |

## Quick Reference — Average Scores

| Model | Avg TTFT (ms) | Avg TPS | PP t/s (prefill) | Agentic Grade |
|---|---|---|---|---|
| granite-4.1-8b-mxfp4 | 337 | 65.2 | 723 | A |
| Qwen3.6-35B-A3B-4bit | 2663 | 80.8 | 307 | A- |
| gemma-4-26b-a4b-mxfp4 | 2736 | 78.4 | 290 | B+ |
| granite-4.1-30b-mxfp4 | 1002 | 24.7 | 182 | B+ |
| Qwen3.6-35B-A3B-UD-MXFP4_K_XL | 4050 | 57.2 | 246 | B |
| Qwen3.6-35B-A3B-OptiQ-4bit | 4048 | 53.5 | 212 | B- |
| granite-4.1-30b-nvfp4 | 1102 | 23.0 | 180 | C+ |
| granite-4.1-30b-4bit | 1193 | 21.1 | 179 | C+ |
| Qwen3.6-35B-A3B-MXFP4-MTP (Osaurus) | 8517 | 25.9 | 98 | C |
| Qwen3.6-35B-A3B-MXFP4-MTP (JANGQ) | 8868 | 24.9 | 100 | C |
| Qwen3.6-27B-mxfp4 | 9036 | 24.8 | 79 | D |
| gemma-4-31b-it-mxfp4 | 11210 | 20.5 | 59 | D |

## Files

- [`benchmarks/raw.md`](benchmarks/raw.md) — Full per-test numbers for all 12 models
- [`benchmarks/analysis.md`](benchmarks/analysis.md) — Quantization, architecture, and long-context analysis
- [`benchmarks/recommendations.md`](benchmarks/recommendations.md) — Use-case recommendations (opencode, agentic loops, context tiers)
