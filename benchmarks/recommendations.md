# Recommendations

**Hardware:** Apple M4 Max | **Inference:** vMLX | **Date:** 2026-05-20

---

## TL;DR

| Use case | Recommended model |
|---|---|
| Agentic / opencode, 32k ctx | **Qwen3.6-35B-A3B-4bit** |
| Agentic / opencode, 64k+ ctx | **Qwen3.6-35B-A3B-4bit** + `--kv-cache-bits 8` |
| Rapid short-cycle tool loops | **granite-4.1-8b-mxfp4** |
| Quality-focused single session | **granite-4.1-30b-mxfp4** |
| 256k context | API (Claude Sonnet/Haiku) — local not viable |

---

## Context Window Tiers

### 32k — All viable candidates work

**Primary:** `Qwen3.6-35B-A3B-4bit`  
Best TPS (80.8), best prefill PP t/s (307), MoE keeps generation fast throughout the session. Quality ceiling is highest of the tested models.

**Alternative:** `granite-4.1-8b-mxfp4`  
If you need sub-second TTFT on every turn or are doing rapid tool-call loops with short prompts. 337ms avg TTFT vs 2663ms on Qwen. Trades quality for responsiveness.

**Avoid:** Dense 27B+ models (Qwen3.6-27B, gemma-4-31b). PP t/s of 17–22 means context processing is painfully slow even at 32k. The MoE alternatives at similar or larger total params run 3–4× faster.

### 64k — KV cache quantization starts paying off

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-cache-bits 8 \
  --context-size 65536
```

With INT8 KV cache: ~22 GB total RAM (model + KV). INT8 quality loss is negligible for most tasks.

granite-4.1-8b becomes compelling here — its 723 PP t/s means it processes accumulated context 2.4× faster than Qwen-35B-A3B. If your agent is reading large files repeatedly, the prefill speed difference is felt.

### 128k — Realistic with MoE + KV quant

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-cache-bits 8 \
  --context-size 131072
```

Estimated RAM: ~25 GB (18 GB weights + 7 GB KV at INT8). Comfortable on M4 Max (128 GB).

Initial context load is slow (~7+ min at 128k) but paid once per session. Per-turn overhead is low if KV cache stays warm. Do not restart the server mid-session.

granite-4.1-8b is the only model where 128k initial load is under 3 minutes. Use it when session startup time matters or for rapid-iteration workflows.

**Models to avoid at 128k:** gemma-4-31b (59 PP t/s → 36+ min initial load), Qwen3.6-27B (79 PP t/s → 27 min), any MXFP4-MTP variant.

### 256k — Local inference not recommended

Even with INT4 KV cache, the initial load time at 256k is prohibitive for most models on M4 Max:

| Model | INT4 KV @ 256k | + Weights | Initial load est. |
|---|---|---|---|
| granite-4.1-8b | ~3.7 GB | ~5 GB | ~6 min |
| Qwen3.6-35B-A3B-4bit | ~3.5 GB | ~18 GB | ~14 min |

Mathematically feasible in memory, but practically unpleasant. PP t/s will also degrade from benchmark values at this context depth. Use the API for 256k+.

---

## opencode Specific

### Configuration

opencode routes model calls through an OpenAI-compatible endpoint. vMLX exposes one at `http://127.0.0.1:PORT/v1`.

```json
{
  "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
  "baseURL": "http://127.0.0.1:1234/v1",
  "apiKey": "local"
}
```

### Qwen3 thinking mode

Qwen3.6 is a reasoning model — it emits thinking traces (`<think>...</think>`) before responding. In opencode:

- Thinking tokens consume context budget without contributing to output
- A single agent turn can balloon from 2k to 6k+ tokens with reasoning
- At 64k+ context this accelerates context exhaustion significantly

Disable thinking in opencode's system prompt or pass `/no_think` if the model supports it. Alternatively use the model's non-thinking variant if available.

### Tool call latency

Each tool call in an agentic loop has this latency profile:

```
TTFT (model decides to call tool)
+ tool execution time
+ TTFT (model processes tool result)
+ generation time (model response)
```

The first TTFT dominates for short decisions. At short prompt lengths:

| Model | Short TTFT | Feel |
|---|---|---|
| granite-4.1-8b | 370ms | Instant |
| granite-4.1-30b-mxfp4 | 350ms | Instant |
| Qwen3.6-35B-A3B-4bit | 1374ms | Noticeable |
| gemma-4-26b-a4b | 1423ms | Noticeable |
| Any dense 27B+ | 2800ms+ | Sluggish |

If your workflow involves many rapid short-decision tool calls (grep, read file, check output), granite-4.1-8b will feel 4× more responsive than Qwen-35B-A3B even though it has lower TPS.

---

## Model Selection Decision Tree

```
Need >64k context?
├─ Yes → Use Qwen3.6-35B-A3B-4bit + --kv-cache-bits 8
│         (granite-4.1-8b if startup time matters)
└─ No  → Priority: quality or speed?
          ├─ Quality → Qwen3.6-35B-A3B-4bit
          └─ Speed   → granite-4.1-8b-mxfp4

Need >128k context?
└─ Use API (Claude Sonnet/Haiku for 200k)
```

---

## What Not to Use

| Model | Why |
|---|---|
| Qwen3.6-35B-A3B-MXFP4-MTP (both) | MTP adds overhead with no benefit for single-user local inference |
| gemma-4-31b-it-mxfp4 | Dense 31B with worst PP t/s (59) — context scaling is brutal |
| Qwen3.6-27B-mxfp4 | Dense 27B underperforms its own MoE sibling (35B-A3B) in every metric |
| Qwen3.6-35B-A3B-OptiQ-4bit | −34% TPS vs standard 4bit for marginal quality gain at 35B scale |

---

## vMLX Server Launch Reference

```bash
# Standard — 32k context
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 1234

# Long context — 128k with INT8 KV cache
mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-cache-bits 8 \
  --context-size 131072 \
  --port 1234

# Speed-priority — granite 8b, 128k
mlx_lm.server \
  --model mlx-community/granite-4.1-8b-mxfp4 \
  --kv-cache-bits 8 \
  --context-size 131072 \
  --port 1234
```
