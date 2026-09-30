#include "targets/qwen3_6_27b_tp2/impl/tp2_wire8.h"

#include "core/device.h"

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// One warp per block of 64 values, two per lane.
__global__ void wire8_quantize_kernel(const __nv_bfloat162* __restrict__ x, std::int64_t blocks,
                                      char2* __restrict__ codes, float* __restrict__ scales) {
    const std::int64_t block = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane           = static_cast<int>(threadIdx.x & 31);
    if (block >= blocks) { return; }
    const __nv_bfloat162 v = x[block * 32 + lane];
    const float a = __low2float(v);
    const float b = __high2float(v);
    float m = fmaxf(fabsf(a), fabsf(b));
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    char2 q;
    q.x = static_cast<signed char>(max(-127, min(127, __float2int_rn(a * inv))));
    q.y = static_cast<signed char>(max(-127, min(127, __float2int_rn(b * inv))));
    codes[block * 32 + lane] = q;
    if (lane == 0) { scales[block] = m / 127.0f; }
}

__global__ void wire8_combine_kernel(__nv_bfloat162* __restrict__ residual, std::int64_t pairs,
                                     const char2* __restrict__ c0, const float* __restrict__ s0,
                                     const char2* __restrict__ c1, const float* __restrict__ s1) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) { return; }
    const std::int64_t block = i / (kWire8Block / 2);
    const float sc0 = s0[block];
    const float sc1 = s1[block];
    const char2 q0 = c0[i];
    const char2 q1 = c1[i];
    const __nv_bfloat162 r = residual[i];
    // Explicit _rn: identical rounding on sm_70 and sm_89, never contracted into an FMA.
    const float x0 = __fadd_rn(__fadd_rn(__low2float(r), __fmul_rn(static_cast<float>(q0.x), sc0)),
                               __fmul_rn(static_cast<float>(q1.x), sc1));
    const float x1 = __fadd_rn(__fadd_rn(__high2float(r), __fmul_rn(static_cast<float>(q0.y), sc0)),
                               __fmul_rn(static_cast<float>(q1.y), sc1));
    residual[i] = __floats2bfloat162_rn(x0, x1);
}

} // namespace

std::size_t wire8_slot_bytes(std::int64_t elements) {
    return align256(static_cast<std::size_t>(elements)) +
           align256(static_cast<std::size_t>(elements / kWire8Block) * sizeof(float));
}

void launch_wire8_quantize(const __nv_bfloat16* x, std::int64_t elements, void* slot, cudaStream_t stream) {
    const std::int64_t blocks = elements / kWire8Block;
    auto* codes  = static_cast<char2*>(slot);
    auto* scales = reinterpret_cast<float*>(static_cast<std::byte*>(slot) + align256(static_cast<std::size_t>(elements)));
    const std::int64_t threads = blocks * 32;
    wire8_quantize_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(x), blocks, codes, scales);
    CUDA_CHECK(cudaGetLastError());
}

void launch_wire8_combine(__nv_bfloat16* residual, std::int64_t elements, const void* slot0, const void* slot1,
                          cudaStream_t stream) {
    const std::size_t codes = align256(static_cast<std::size_t>(elements));
    const std::int64_t pairs = elements / 2;
    wire8_combine_kernel<<<static_cast<unsigned>((pairs + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat162*>(residual), pairs, static_cast<const char2*>(slot0),
        reinterpret_cast<const float*>(static_cast<const std::byte*>(slot0) + codes), static_cast<const char2*>(slot1),
        reinterpret_cast<const float*>(static_cast<const std::byte*>(slot1) + codes));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
