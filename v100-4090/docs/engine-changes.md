# Engine changes on this branch

Starting point: [huangserva/ninfer-v100-tpx](https://github.com/huangserva/ninfer-v100-tpx) (V100 port of NInfer by
geoffwatts + rewritten sm70 attention + tensor parallel across V100s) and memoriaru's sm_89 port. Everything below was
added to run **one V100 and one RTX 4090 as the two ranks of the same tensor-parallel group**. Each change was measured
with an A/B before it stayed; the environment switch, when there is one, turns it off for comparisons.

## Rank 1 = RTX 4090 on the Volta code path

The two ranks must use the same weight layouts and the same reduction protocol, so the 4090 build compiles the Volta
code path for sm_89 (`-DNINFER_VOLTA_PATH=ON`) and then replaces individual kernels where Ada pays off.

| commit | change |
|---|---|
| `297679e6` | the 4090 runs the Volta path (`NINFER_VOLTA_PATH` on sm_89): same kernels, software NVFP4 and TP2 as the V100 |
| `4a01affa` | FP16 GEMMs of the Volta path on Ada tensor cores via CUTLASS Sm80 (16x8x16, 3 stages) |
| `94c17c2b` | NVFP4 decode GEMV on Ada tensor cores (`mma.m16n8k16`) |
| `fbeb1f4f` | the 4090 decodes attention with the Ampere int8 kernel (`mma.s8`) instead of the Volta one |
| `934b667c`, `bdb3b392`, `a16f0831`, `43967133`, `1ea54b67` | proxy: one binary per rank, GPUs by UUID, key injection for local clients, per-request seed shared by both ranks, lockstep block mapped per generation, watchdog that returns at once (was a 2 s grid) |

## Asymmetric split

| commit | change | measured |
|---|---|---|
| `d6aca0a8` | MLP intermediate width per rank, set at build time (`NINFER_TP2_INTERMEDIATE`) and in the shard (`--mlp-rank0`); per-block wait counters (`NINFER_TP_STATS=1`) | with an even split the 4090 waited in the MLP |
| `45c3f6e5` | split 7,680 (V100) / 9,728 (4090) | the two cards finish the MLP together |

## 8-bit MLP

| commit | change | measured |
|---|---|---|
| `73d62c19` | text MLP from the Q8_0 blocks of the official GGUF, bit for bit (`W8G32_F16S`); Ada W8 decode on `m16n8k16` | the NVFP4 MLP of the artifact differed 11–15% per weight from the original |
| `efbcd23b`, `45c3f6e5` | W8 decode kernels at the TP2 MLP shapes on both ranks; V100 prepacked GEMV for any W8 shape, MTP projections included | V100 GEMV 660–675 GB/s, 77% of peak |
| `9db6b608` | wide-T W8 GEMMs (prefill) on CUTLASS: Sm70 on the V100, Sm80 on the 4090 | prefill 1,030 t/s at 65k |
| `d2f64072` | V100 prepacked GEMV: KB 2 configurations for T ≤ 8 (decode and MTP verify) — `NINFER_SM70_W8_KB2=0` disables | generation 132.0 → 138.1 t/s, identical output |

## Attention and KV cache on the V100

| commit | change | measured |
|---|---|---|
| `a973fcc3` | small-T int8 attention: splits rounded up to whole waves | |
| `7934b9f6` | small-T int8 attention with two CTAs per SM (v2d) — `NINFER_SM70_ATTN_V2D=0` disables | 400 → 470–480 GB/s, +3% at 80–180k |
| `1ea54b67` | flash route from 17 query tokens (was 64): short follow-up turns no longer hit the scalar prompt kernel — `NINFER_VOLTA_FLASH_MIN_WIDTH=64` restores | first token 2–6 s → 0.3 s at 120–200k |
| (this branch) | GDN controls: upstream's fused RMSNorm + a/b projection (`02be37cb`, the a/b dots read the FP32-normalized input instead of the BF16 `h`) was dropped by the Volta port's merge (`7bf47863`), leaving the op test red on Volta. It is plain SIMT: restored on both ranks up to 42 tokens and generalized to the 24 heads of a TP2 rank — `NINFER_GDN_NORM_FUSED=0` restores the port's composed route | `gdn_gating_proj` green (red again with `=0`); short turns, cold read 1,163 t/s and generation 134.6 → 135.2 t/s unchanged, same VRAM |
| (this branch) | the vendored llama.cpp flash kernel accumulates P·V in FP16 on Volta (no room for FP32 registers at head size 256), so its error grows with the keys one block sums. For 17–63-token reads each stream-K block now sums at most 512 keys and the partials are combined in FP32 — `NINFER_VOLTA_FLASH_SPLIT_KEYS=0` restores, `=N` sets the cap; `NINFER_VOLTA_FLASH_SPLIT_WIDE=1` extends it to wide prompts, which does not fit the 4090 at 262k | rel_l2 at 30k int8 keys 0.0032 → 0.0019 (exact routes: 0.0017); short turns 0.33 → 0.34 s, needles 3/3, cold read 1,166 → 1,162 t/s, same VRAM |

## Communication between the cards

| commit | change | measured |
|---|---|---|
| `bf469dbb`, `19be46ce` | wait counters per rank and NCCL timing with CUDA events (`NINFER_TP_NCCL_STATS=1`) | a third of a cold read was NCCL exchange |
| `0367cbf9` | 8-bit wire for the prefill reductions: only the partial sums travel, int8 with one FP32 scale per 64 values via `ncclAllGather`; the sum `residual + p0 + p1` is computed identically on both ranks (`__fadd_rn`/`__fmul_rn`), so the residual stream never diverges — `NINFER_TP_WIRE8=0` restores bf16. Row-parallel projections in prefill go through `linear_add` — `NINFER_TP_PREFILL_ADD=0` restores | prefill 1,056 → 1,163 t/s at 55k; 83 replayed turns up to 147k with no collapse |

## Vision

| commit | change | measured |
|---|---|---|
| `421c21ea` | `NINFER_MAX_PROMPT_VISION_TOKENS`: per-request aggregate vision-token budget (was a fixed 32,768 ≈ 10 phone photos per conversation). Host memory only: the GPU vision workspace is per image. Needs `--media-live-mib` ≥ 3,072 at 262,144 | no "media budget exceeded" at the 11th photo |
| `b27b48a5` | vision-encoder attention on the Volta path (both ranks): tiled SIMT kernel, K/V in shared memory, 32 queries per block, FP32 — `NINFER_VISION_ATTN=1` | 28.5 → 14.3 s per ~3k-token image, same output at t=0 |
| `397f8f36` | the same attention on FP16 tensor cores (WMMA 16x16x16, FP32 accumulation and softmax), default — `NINFER_VISION_ATTN=0` restores the original scalar kernel | 14.3 → 9.0 s per image, mean relative error 1.8e-4 |

## Tools

| commit | change |
|---|---|
| `d0a89ff4`, `1a38f8da` | native NInfer v3 artifacts in the engine and in the Python tools; TP2 shards of a v3 source are v3 |
| `e56d3e45` | `shard_qwen38_27b.py --drop-dflash2` |
| `f2269833` | `patch_from_gguf.py`: builds a derived Qwen3.8-27B (abliteration) from its Q8_0 GGUF, with a `--check` that detects permuted heads (the GGUF stores the GDN value heads in a different order) |
