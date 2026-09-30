#pragma once

// 8-bit wire for the large (prefill) TP2 reductions. Only the row-parallel partials travel: each rank
// quantizes its partial to int8 with one fp32 scale per 64 consecutive values, one all-gather swaps them
// (half the bytes of a bf16 all-reduce), and both ranks add residual + p0 + p1 in rank order with
// round-to-nearest intrinsics (no FMA contraction), so the result is bit-identical on both ranks.
// The residual stream itself is never quantized.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {

inline constexpr int kWire8Block = 64;  // values per scale

// Bytes of one rank's slot for `elements` values: codes, then scales, each 256-byte aligned.
[[nodiscard]] std::size_t wire8_slot_bytes(std::int64_t elements);

// slot <- codes + scales of x (elements % kWire8Block == 0).
void launch_wire8_quantize(const __nv_bfloat16* x, std::int64_t elements, void* slot, cudaStream_t stream);

// residual <- residual + dequant(slot0) + dequant(slot1), in that order.
void launch_wire8_combine(__nv_bfloat16* residual, std::int64_t elements, const void* slot0, const void* slot1,
                          cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
