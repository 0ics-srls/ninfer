#pragma once

// Tensor-core flavour of the FP16 CUTLASS GEMMs used by the Volta execution path (weights are dequantized to
// FP16 first, then multiplied). On sm_70 that is the Volta mma.m8n8k4 with two stages. The Volta path compiled
// for Ada (the RTX 4090 rank of a V100 + 4090 tensor parallel) keeps the same GEMMs but runs them on Ampere-class
// tensor cores: mma.m16n8k16 with cp.async multistage pipelining.

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>

namespace ninfer::ops::detail {

#ifdef NINFER_ADA_BUILD
using CutlassFp16TensorArch                     = cutlass::arch::Sm80;
using CutlassFp16TensorOpShape                  = cutlass::gemm::GemmShape<16, 8, 16>;
inline constexpr int kCutlassFp16TensorStages   = 3;
#else
using CutlassFp16TensorArch                     = cutlass::arch::Sm70;
using CutlassFp16TensorOpShape                  = cutlass::gemm::GemmShape<8, 8, 4>;
inline constexpr int kCutlassFp16TensorStages   = 2;
#endif

} // namespace ninfer::ops::detail
