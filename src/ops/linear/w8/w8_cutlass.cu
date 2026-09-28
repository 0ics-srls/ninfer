#include "ops/linear/w8/w8_cutlass.h"

#include "core/device.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include "ops/common/cutlass_fp16_arch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, ElementAccumulator,
    ElementComputeEpilogue>;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    CutlassFp16TensorArch, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    CutlassFp16TensorOpShape, EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kCutlassFp16TensorStages>;

__device__ __forceinline__ int qpn_lane_of_row(int local_row) {
    const int qp = local_row / 8;
    const int r  = local_row & 7;
    return (qp << 2) | (r & 3) | ((r & 4) << 2);
}

// One thread per 8 codes: out[row][k0..k0+8) = code * scale (fp16, the decode arithmetic of the GEMV kernels).
__global__ void w8_dequant_fp16_kernel(const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
                                       half* __restrict__ out, int n, int k, bool prepacked) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int chunks_per_row = k / 8;
    if (idx >= static_cast<std::int64_t>(n) * chunks_per_row) { return; }
    const int row    = static_cast<int>(idx / chunks_per_row);
    const int chunk  = static_cast<int>(idx % chunks_per_row);
    const int groups = k / 32;
    const int g      = chunk / 4;
    const int b      = (chunk % 4) * 8;
    std::int64_t code_off;
    std::int64_t scale_off;
    if (prepacked) {
        const std::int64_t tuple = (static_cast<std::int64_t>(row / 32) * groups + g) * 32 + qpn_lane_of_row(row & 31);
        code_off  = tuple * 32 + b;
        scale_off = tuple;
    } else {
        code_off  = static_cast<std::int64_t>(row) * k + g * 32 + b;
        scale_off = static_cast<std::int64_t>(row) * groups + g;
    }
    const uint2 raw  = *reinterpret_cast<const uint2*>(codes + code_off);
    const half2 sc2  = __half2half2(__ushort_as_half(scales[scale_off]));
    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0
    const std::uint32_t w0 = raw.x ^ 0x80808080u;
    const std::uint32_t w1 = raw.y ^ 0x80808080u;
    std::uint32_t h[4] = {__byte_perm(w0, 0x64646464u, 0x5150), __byte_perm(w0, 0x64646464u, 0x5352),
                          __byte_perm(w1, 0x64646464u, 0x5150), __byte_perm(w1, 0x64646464u, 0x5352)};
    half2 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&h[j]), bias), sc2); }
    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(row) * k + chunk * 8) = *reinterpret_cast<uint4*>(v);
}

__global__ void w8_x_fp16_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ y, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { y[i] = __float2half(__bfloat162float(x[i])); }
}

struct Scratch {
    std::int64_t w_elems = 0;
    std::int64_t x_elems = 0;
    half* w              = nullptr;
    half* x              = nullptr;
};
Scratch g_scratch;

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_W8_CUTLASS");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

} // namespace

void w8_cutlass_reserve(std::int64_t n, std::int64_t k, std::int64_t max_t) {
    if (!enabled()) { return; }
    if (n * k > g_scratch.w_elems) {
        if (g_scratch.w != nullptr) { CUDA_CHECK(cudaFree(g_scratch.w)); }
        CUDA_CHECK(cudaMalloc(&g_scratch.w, static_cast<std::size_t>(n * k) * sizeof(half)));
        g_scratch.w_elems = n * k;
    }
    if (max_t * k > g_scratch.x_elems) {
        if (g_scratch.x != nullptr) { CUDA_CHECK(cudaFree(g_scratch.x)); }
        CUDA_CHECK(cudaMalloc(&g_scratch.x, static_cast<std::size_t>(max_t * k) * sizeof(half)));
        g_scratch.x_elems = max_t * k;
    }
}

bool w8_cutlass_ready(std::int64_t n, std::int64_t k, std::int64_t t) {
    return enabled() && n * k <= g_scratch.w_elems && t * k <= g_scratch.x_elems && n % 8 == 0 && k % 32 == 0;
}

void w8_cutlass_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = w.n;
    const std::int32_t k = w.k;
    const std::int32_t t = x.ne[1];
    if (w.padded_shape[1] != k) { throw std::invalid_argument("w8_cutlass: padded K is not supported"); }
    const std::int64_t chunks = static_cast<std::int64_t>(n) * (k / 8);
    w8_dequant_fp16_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales), g_scratch.w, n, k,
        w.layout == QuantLayout::VoltaQpnPrepacked);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t x_count = static_cast<std::int64_t>(t) * k;
    w8_x_fp16_kernel<<<static_cast<unsigned>((x_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), g_scratch.x, x_count);
    CUDA_CHECK(cudaGetLastError());

    auto* xa = reinterpret_cast<cutlass::half_t*>(g_scratch.x);
    auto* wb = reinterpret_cast<cutlass::half_t*>(g_scratch.w);
    auto* y  = static_cast<cutlass::bfloat16_t*>(out.data);
    Gemm gemm;
    typename Gemm::Arguments args{{t, n, k}, {xa, k}, {wb, k}, {y, n}, {y, n},
                                  {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        throw std::runtime_error("w8_cutlass: CUTLASS can_implement failed");
    }
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("w8_cutlass: CUTLASS initialize failed");
    }
    if (gemm(stream) != cutlass::Status::kSuccess) { throw std::runtime_error("w8_cutlass: CUTLASS gemm failed"); }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
