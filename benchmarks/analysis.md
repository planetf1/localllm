# Benchmark Analysis

**Hardware:** Apple M4 Max | **Inference:** vMLX | **Date:** 2026-05-20

---

## 1. Architecture: Dense vs MoE

The single biggest performance driver is **MoE (Mixture of Experts)** vs dense architecture.

| Comparison | Dense | MoE | Difference |
|---|---|---|---|
| gemma-4 26B vs 31B | 31B: 20.5 TPS | 26B (4B active): 78.4 TPS | **3.8× faster** |
| Qwen 27B vs 35B-A3B | 27B: 24.8 TPS | 35B (3B active): 80.8 TPS | **3.3× faster** |

**Why:** MoE routes each token through only a subset of FFN experts (~3–4B params active vs 27–31B). Attention layers are still dense, but FFN dominates compute at generation time. The result: a "35B" MoE model generates tokens at the speed of a ~3B dense model while retaining the quality of a much larger parameter count.

**KV cache caveat:** KV cache scales with attention layer count and sequence length — not expert count. A 35B MoE and a 35B dense model have similar KV cache sizes if they have similar depth. However, MoE models tend to be *shallower* (fewer layers) than equivalent-quality dense models, which is a secondary advantage at long context.

---

## 2. Quantization Format Analysis

### granite-4.1-30b: Why mxfp4 > nvfp4 > 4bit

Granite 4.1 was explicitly designed and released with MXFP4 as the primary quantization target. IBM's quantization pipeline optimises weight calibration for MXFP4. MLX also has mature MXFP4 Metal kernels for this model family.

| Variant | Avg TPS | Short TTFT | PP t/s (prefill) |
|---|---|---|---|
| mxfp4 | 24.7 | **350ms** | 182 |
| nvfp4 | 23.0 | 686ms | 180 |
| 4bit | 21.1 | 1029ms | 179 |

- TPS differences are modest (17% between mxfp4 and 4bit)
- **Short TTFT is the biggest gap** — mxfp4 is 3× faster than 4bit for short prompts
- PP t/s (prefill) is essentially identical across all three at 179–182: at long context they converge completely

### Qwen3.6-35B-A3B: Why 4bit > mxfp4 variants

The inverse pattern. Standard INT4 outperforms MXFP4 formats for Qwen MoE:

| Variant | Avg TPS | Avg TTFT | PP t/s (prefill) |
|---|---|---|---|
| **4bit** | **80.8** | **2663ms** | **307** |
| UD-MXFP4_K_XL | 57.2 | 4050ms | 246 |
| OptiQ-4bit | 53.5 | 4048ms | 212 |
| MXFP4-MTP (x2) | ~25 | ~8700ms | ~99 |

**Root cause — kernel maturity:** MLX's optimised Metal kernels for its native `q4_K`/`q4_0` grouped INT4 format are highly tuned and have been the primary local inference format for Apple Silicon. MXFP4's block-level microscaling requires different dequantization logic that falls back to less-optimised paths on M-series GPU.

**Root cause — model origin:** Qwen3.6-35B-A3B quants were produced by the community, not the model authors. The standard 4bit conversion was done with mature MLX tooling. The MXFP4 variants are newer community conversions with less calibration work.

**Contrast with Granite:** IBM ships their models *already in MXFP4* with their own calibration. The mlx-community just wraps these. The kernel path is well-tested.

### General principle

> **Whichever format the model's primary release pipeline used tends to win on MLX** — that is where the quantization calibration effort and MLX kernel optimisation effort both landed.

- Granite 4.1 → IBM released in MXFP4 → mxfp4 wins
- Qwen3.6-35B-A3B → community-quantised, mature tooling is INT4 → 4bit wins
- When evaluating a new model: check what format the original authors released or recommended before assuming MXFP4 is superior.

### OptiQ vs standard 4bit

OptiQ uses an optimisation pass to minimise weight reconstruction error (similar to GPTQ), giving marginally better perplexity at the same bit width. However:

- TPS penalty: −34% vs standard 4bit (53.5 vs 80.8)
- TTFT penalty: +52% worse
- Quality gain: marginal at 35B scale

**Verdict:** OptiQ trades significant speed for marginal quality. Not worth it for agentic workloads. Better justification exists for smaller models (8B) where 4-bit quality loss is more impactful.

### MTP (Multi-Token Prediction) variants

Both JANGQ and Osaurus MXFP4-MTP quants severely underperform (~25 TPS, TTFT 8–9s avg):

- MTP is a speculative decoding strategy designed for *batched server workloads* — it predicts N+1 tokens ahead and verifies them in parallel across requests. Single-user local inference gets no benefit.
- The MTP overhead (additional forward passes for speculative tokens, verification logic) adds pure cost with no payoff in a single-request setup.
- Combined with immature MXFP4 kernels: double penalty.

---

## 3. Long Context Scaling

### The KV cache picture

KV cache size formula: `2 × layers × kv_heads × head_dim × seq_len × bytes`

Estimated KV cache at 128k context (FP16):

| Model | Est. KV @ 128k | + Model weights | Total RAM |
|---|---|---|---|
| granite-4.1-8b | ~8 GB | ~5 GB | ~13 GB |
| Qwen3.6-35B-A3B | ~7 GB* | ~18 GB | ~25 GB |
| granite-4.1-30b | ~11 GB | ~16 GB | ~27 GB |
| gemma-4-26b-a4b | ~7 GB* | ~14 GB | ~21 GB |

*MoE models use GQA with fewer KV heads + shallower stacks → smaller KV cache than equivalent dense models.

All fit comfortably on M4 Max (128GB).

### KV cache quantization

MLX supports `--kv-cache-bits` (INT8/INT4). At INT8:

| Context | KV cache (FP16) | KV cache (INT8) | KV cache (INT4) |
|---|---|---|---|
| 64k | ~3.5 GB | ~1.75 GB | ~0.9 GB |
| 128k | ~7 GB | ~3.5 GB | ~1.75 GB |
| 256k | ~14 GB | ~7 GB | ~3.5 GB |

256k becomes viable for MoE models with INT8 KV cache (~22 GB total on M4 Max).

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-cache-bits 8 \
  --context-size 131072
```

### Why MoE scales better at long context

1. Shallower attention stacks → smaller KV cache baseline
2. Lower active parameter count → FFN compute stays fast even as KV cache I/O grows
3. At very long context (256k+), KV cache bandwidth dominates — MoE's FFN savings diminish, but KV cache size advantage persists

### Agentic loop context model

In vMLX server mode, KV cache is persisted between turns within a session. You do **not** re-prefill the full context on every turn — only new tokens are processed. This changes the economics:

- **Initial load:** slow (full context prefill, one-time cost)
- **Per-turn:** fast (only new tokens since last turn, typically 1–3k)
- **The real bottleneck:** total session duration × per-turn overhead, not a single prefill

Estimated initial load time at 128k (using prefill PP t/s as proxy):

| Model | PP t/s (prefill) | Initial 128k load |
|---|---|---|
| granite-4.1-8b | 723 | ~3 min |
| Qwen3.6-35B-A3B-4bit | 307 | ~7 min |
| gemma-4-26b-a4b | 290 | ~7.4 min |
| granite-4.1-30b-mxfp4 | 182 | ~12 min |
| Qwen dense / gemma-31b | <80 | 25+ min |

Note: these PP t/s values were measured at short prompts. Real degradation at 128k means actual times will be longer — treat as lower bounds.

---

## 4. Downloaded But Not Benchmarked

The following variants appeared in the vMLX download queue during this session but were not benchmarked:

| Model | Notes |
|---|---|
| Qwen3.6-35B-A3B-MLX-MXFP4 | Standard MXFP4 from mlx-community. Expected to perform similar to UD-MXFP4_K_XL (~55–60 TPS). Lower priority given 4bit already superior. |
| Qwen3.6-35B-A3B-RotorQuant-MLX-MXFP4 | RotorQuant applies rotation-based weight transformation before quantization to reduce outliers. May improve quality over standard MXFP4 at similar speed. Worth testing if MXFP4 quality is needed. |

---

## 5. Agentic Scoring

Agentic loops (opencode, tool-call agents) weight metrics differently from raw generation:

- **40% TTFT** — each tool-call decision pays this cost
- **35% PP t/s** — context grows with every tool call; processing speed compounds
- **25% generation TPS** — output quality feel, but not the bottleneck

| Model | TTFT score | PP t/s score | TPS score | Agentic Grade |
|---|---|---|---|---|
| granite-4.1-8b-mxfp4 | ★★★★★ | ★★★★★ | ★★★★ | **A** |
| Qwen3.6-35B-A3B-4bit | ★★★ | ★★★★ | ★★★★★ | **A-** |
| gemma-4-26b-a4b-mxfp4 | ★★★ | ★★★★ | ★★★★★ | **B+** |
| granite-4.1-30b-mxfp4 | ★★★★ | ★★★ | ★★ | **B+** |
| Qwen3.6-35B-A3B-UD-MXFP4_K_XL | ★★ | ★★★ | ★★★ | **B** |
| Qwen3.6-35B-A3B-OptiQ-4bit | ★★ | ★★★ | ★★★ | **B-** |
| granite-4.1-30b-nvfp4 | ★★★ | ★★★ | ★★ | **C+** |
| granite-4.1-30b-4bit | ★★★ | ★★★ | ★★ | **C+** |
| MXFP4-MTP variants | ★ | ★★ | ★★ | **C** |
| Qwen3.6-27B-mxfp4 | ★ | ★★ | ★★ | **D** |
| gemma-4-31b-mxfp4 | ★ | ★ | ★★ | **D** |

**granite-4.1-30b grade rationale:** Earns B+ despite low TPS because (a) Granite 4.1 is IBM's explicitly tool-use/function-calling optimised family — architectural quality bonus; (b) short TTFT of 350ms means tool-call decisions feel fast; (c) PP t/s of 182 handles context accumulation reasonably well.
