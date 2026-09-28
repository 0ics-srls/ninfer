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
inline constexpr std::int32_t kTp2Intermediate     = NINFER_TP2_INTERMEDIATE;

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
