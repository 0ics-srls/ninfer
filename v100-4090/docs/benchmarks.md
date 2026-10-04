# Benchmarks

Machine: Tesla V100 PCIe 32 GB (150 W) + RTX 4090 (400 W), ASRock X570 Taichi, Ryzen 5 5600X, 64 GB DDR4, both cards
on PCIe 3.0 x8, no P2P. Model: Qwen3.8-27B (official and orcarouter Uncensored), MLP Q8_0, attention/GDN FP8, int8 KV
cache, MTP 4 drafts. Engine configuration = `scripts/start-tp.sh` defaults. September–October 2026.

## Real agentic work

Measured from the engine's request log (`bench/session-times.py`) during complete runs of the
[ninfer-code-test](https://github.com/0ics-srls/ninfer-code-test) plan in OpenCode (~330–420 requests per run,
contexts from 40k to 230k).

| run | model | engine state | active work | generation <100k / 100–160k / >160k | first token, short turns (median / p90 / max) |
|---|---|---|---|---|---|
| 1 | official | before the first-token fixes | 2 h 05 | 94 / 87 / 85 t/s | 0.49 / 3.08 / 6.58 s |
| 2 | official | + flash threshold 17, proxy fix, preserve-thinking, 16k cap | 1 h 23 | 96 / 93 / 89 t/s | 0.37 / 0.53 / 0.70 s |
| 3 | Uncensored | same as run 2 | 1 h 29 | 104 / 97 / 90 t/s | 0.35 / 0.50 / 0.64 s |
| 4 | Uncensored | + 8-bit wire, V100 GEMV KB 2 | 2 h 28 | 106 / 100 / 91 t/s | 0.35 / 0.51 / 0.74 s |

All four: 6/6 blocks, e2e 25/25 twice (re-run by us on a separate checkout), code review 20–21/25 from three
independent reviewers. Run 4 was longer with the fastest engine because the model produced 458k reasoning tokens
instead of 149–165k; replaying the same 40 requests on the run-3 and run-4 engine builds gave 57.5k vs 57.0k reasoning
tokens, so the difference is the conversation, not the engine.

## Synthetic

| test | result |
|---|---|
| code generation, `bench/gen-speed.py` (3,000 tokens, t=0) | 132.0 → **138.1 t/s** with the V100 GEMV KB 2 change, identical text |
| code rewrite with MTP (file in the prompt) | **305 t/s** |
| prose, 4k new tokens | **86 t/s** |
| generation at 130k / 196k / 261k (growth curve) | **140 / 111 / 92 t/s** |
| cold prefill at 55k | 1,056 → **1,163 t/s** with the 8-bit wire |
| cold prefill at 65k / 100k (before the 8-bit wire) | 1,030 / 960 t/s |
| needles, 3 codes in ~120k / ~200k | 3/3 |
| one image, ~3,000 vision tokens | 28.5 s (original) → 14.3 s (tiled) → **9.0 s** (tensor cores) |
| vision attention kernel, 12,288 + 3,072 patches, V100 / 4090 | 891 / 261 ms → 361 / 129 ms → **180 / 52 ms** |
| load time | ~75 s |

## Where the prefill time goes (Nsight Systems, 32k cold read)

21.4 s = 15.4 s of V100 compute + 5.9 s of exchange between the cards (NCCL through host memory, 2 channels). The 4090
waits ~8 s for the V100 in GDN and attention. Remaining levers: overlap the exchange with compute; split the GDN /
attention heads by compute as the MLP already is.

## Other engines on the same machine and model

| engine | generation on real work | notes |
|---|---|---|
| **this engine** | **89–106 t/s up to 262k** | |
| llama.cpp, tensor parallel `-sm tensor` 50/50, Q8_0 + MTP | 57 / 51 t/s at 30k / 100k, 23–46 t/s past 200k | fixed 50/50 split, generic kernels |
| llama.cpp, layer split 51/49, Q8_0 + MTP | 55.5 t/s at short context | the default 43/57 split is 15% slower in prefill |
| vLLM (1Cat fork) tensor parallel, FP8 | 17–19 t/s at 90–100k; 25.8 t/s at 259k (synthetic) | needed an FP4 scale-layout fix for the 4090 |

## Things that did not pay off (measured, dropped)

- NCCL `Simple` protocol, 4 channels instead of 2 (no change); `NCCL_SHM_USE_CUDA_MEMCPY` (blocks the startup).
- V100 at 250 W: +4% prefill, +0.5% generation.
- f16 / bf16 KV cache on the V100 instead of int8: by bandwidth it cannot beat int8 with this kernel.
- 2 or 5–6 MTP drafts instead of 4; full LM head for the drafts.
- Two prefetch sets, 16-byte loads on the single-CTA layout, QK split across 8 warps in the V100 attention kernel.
- Splitting the KV heads 1:3 between the cards (estimated, not built): moves the wait to the other card, about −25% at
  long context.
