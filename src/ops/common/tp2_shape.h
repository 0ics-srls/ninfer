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

static_assert(kTp2Intermediate > 0 && kTp2Intermediate < kTp2FullIntermediate && kTp2Intermediate % 128 == 0,
              "TP2 MLP shard width must be a positive multiple of 128 below 17408");
// The shape predicates below tell MLP shards from the other half-shapes by size alone.
static_assert(kTp2Intermediate != 3072 && kTp2Intermediate != 6144 && 2 * kTp2Intermediate != 7168 &&
                  2 * kTp2Intermediate != 8192 && 2 * kTp2Intermediate != 6144 && 2 * kTp2Intermediate != 12288,
              "TP2 MLP shard width collides with another projection shape");

} // namespace ninfer::ops::detail
