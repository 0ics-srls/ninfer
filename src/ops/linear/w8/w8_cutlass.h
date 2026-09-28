#pragma once

// Wide-T W8G32 GEMM on CUTLASS for the Volta-path builds (V100 and the RTX 4090 rank): the weight is dequantized
// to fp16 into a load-time scratch (row-major, or the V100 lane-order prepack), x is staged in fp16, and one
// CUTLASS GEMM runs on the arch picked by ops/common/cutlass_fp16_arch.h.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD
// Grows the scratch to cover an n x k weight and max_t tokens. Call at load time, outside graph capture.
void w8_cutlass_reserve(std::int64_t n, std::int64_t k, std::int64_t max_t);
// true when the CUTLASS route is enabled (NINFER_W8_CUTLASS != 0) and the scratch covers this problem.
[[nodiscard]] bool w8_cutlass_ready(std::int64_t n, std::int64_t k, std::int64_t t);
void w8_cutlass_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
