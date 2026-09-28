#pragma once

// V100 (sm_70 build only) W8G32 GEMV on load-time prepacked weights, for the TP2 8-bit MLP.
//
// The row-major W8 QPN kernel gives each lane its own weight row, so a warp load touches 32 rows and the MLP shapes
// ran at 300-390 GB/s. Prepacked, lane L of a 32-row tile owns the same row as in the QPN fragment map but its 32
// code bytes of a group sit next to its neighbours': one warp reads 1 KiB contiguous per group. Measured on the
// rank shapes at T = 4 (bench_w8v): 15872x5120 382 -> 649 GB/s, 5120x7936 308 -> 680 GB/s (V100 ceiling 879).
//
//   codes  [(tile * groups + g) * 32 + lane][32]   (row = tile * 32 + qpn_row(lane))
//   scales [(tile * groups + g) * 32 + lane]       fp16

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if defined(NINFER_VOLTA_BUILD) && !defined(NINFER_ADA_BUILD)
// Grows the device scratch (fp16 activation stage and a row-major copy for wide T) to cover an n x k weight.
// Call at load time, outside graph capture.
void w8_prepacked_reserve(std::int64_t n, std::int64_t k);
// In-place permutation of a row-split W8G32 weight (k == padded k, n % 32 == 0) into the prepacked order.
void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);
// y[T, N] = x[T, K] * W^T for a prepacked weight: T <= 32 on the prepacked kernel, wider T through the row-major
// Volta MMA route on an unpacked copy.
void launch_w8_prepacked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
