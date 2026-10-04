#pragma once

// Per-rank MLP intermediate width of the Qwen3.8-27B TP2 shard this binary serves. 8704 is the even split of 17408;
// the V100 and RTX 4090 ranks are separate builds, so an uneven split compiles each rank with its own width
// (NINFER_TP2_INTERMEDIATE, CMake). The shard tool cuts the artifact at the same point (--mlp-rank0).

#include <cstdint>

#ifndef NINFER_TP2_INTERMEDIATE
#define NINFER_TP2_INTERMEDIATE 8704
#endif

namespace ninfer::ops::detail {

inline constexpr std::int32_t kTp2FullIntermediate = 17408;
inline constexpr std::int32_t kTp2EvenIntermediate = kTp2FullIntermediate / 2;
inline constexpr std::int32_t kTp2Intermediate     = NINFER_TP2_INTERMEDIATE;

// MLP shard shapes a TP2 build serves: its own width, and always the even split (the original TP2 contract,
// still used by the MTP layer of even-split artifacts and by the shape-generic tests).
inline constexpr bool is_tp2_mlp_shard_width(std::int32_t width) {
    return width == kTp2EvenIntermediate || width == kTp2Intermediate;
}
inline constexpr bool is_tp2_mlp_gate_up_shard(std::int32_t n, std::int32_t k) {
    return k == 5120 && n % 2 == 0 && is_tp2_mlp_shard_width(n / 2);
}
inline constexpr bool is_tp2_mlp_down_shard(std::int32_t n, std::int32_t k) {
    return n == 5120 && is_tp2_mlp_shard_width(k);
}

// Multiple of 256: an 8-bit (W8G32) down projection has K = this width, and the Ampere W8 MMA stages eight
// group scales per row with one 16-byte cp.async, so every row of the scale plane (K/32 * 2 bytes) must stay
// 16-byte aligned (9600 = 300 groups gave the RTX 4090 a wrong down projection in prefill).
static_assert(kTp2Intermediate > 0 && kTp2Intermediate < kTp2FullIntermediate && kTp2Intermediate % 256 == 0,
              "TP2 MLP shard width must be a positive multiple of 256 below 17408");
// The shape predicates below tell MLP shards from the other half-shapes by size alone.
static_assert(kTp2Intermediate != 3072 && kTp2Intermediate != 6144 && 2 * kTp2Intermediate != 7168 &&
                  2 * kTp2Intermediate != 8192 && 2 * kTp2Intermediate != 6144 && 2 * kTp2Intermediate != 12288,
              "TP2 MLP shard width collides with another projection shape");

} // namespace ninfer::ops::detail
