#include "ops/linear/w8/w8_cutlass.h"
#include "ops/linear/w8/w8_volta_qpn_prepacked.h"

#include "core/device.h"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/w8/w8_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace ninfer::ops::detail {

#if defined(NINFER_VOLTA_BUILD) && !defined(NINFER_ADA_BUILD)

namespace {

constexpr int kGroupK = 32;

// Lane (qp, r) of the Volta quadpair owns output row qp * 8 + r of a 32-row tile.
__host__ __device__ __forceinline__ int qpn_lane_row(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
}

__global__ void w8_pack_kernel(const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
                               std::uint8_t* __restrict__ packed_codes, std::uint16_t* __restrict__ packed_scales,
                               int n, int groups, bool unpack) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(n / 32) * groups * 32;
    if (idx >= total) { return; }
    const int lane = static_cast<int>(idx % 32);
    const int g    = static_cast<int>((idx / 32) % groups);
    const int tile = static_cast<int>(idx / (32LL * groups));
    const std::int64_t row_major = (static_cast<std::int64_t>(tile * 32 + qpn_lane_row(lane)) * groups + g);
    if (!unpack) {
        const uint4* src = reinterpret_cast<const uint4*>(codes + row_major * 32);
        uint4* dst       = reinterpret_cast<uint4*>(packed_codes + idx * 32);
        dst[0] = src[0];
        dst[1] = src[1];
        packed_scales[idx] = scales[row_major];
    } else {
        const uint4* src = reinterpret_cast<const uint4*>(codes + idx * 32);
        uint4* dst       = reinterpret_cast<uint4*>(packed_codes + row_major * 32);
        dst[0] = src[0];
        dst[1] = src[1];
        packed_scales[row_major] = scales[idx];
    }
}

__global__ void w8_stage_half_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ y, int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { y[i] = __float2half(__bfloat162float(x[i])); }
}

template <int SPLITK, int KB, int MINB, int TILES>
__global__ __launch_bounds__(SPLITK * 32, MINB) void w8_volta_prepacked_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const half* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t, int out_ld) {
    __shared__ float cs[SPLITK][TILES * 8 * 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int groups = k / kGroupK;
    const int gq   = groups / SPLITK;
    const int g0   = warp * gq;
    const int gend = warp == SPLITK - 1 ? groups : g0 + gq;
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * groups * 32 + lane;
    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0
    float c[TILES][8];
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    // Two statically named register sets: a runtime-indexed [2] array would land in local memory.
    uint4 ca[KB][2], cb[KB][2];
    std::uint16_t sa[KB], sb[KB];
    auto load = [&](uint4 (&cw)[KB][2], std::uint16_t (&sc)[KB], int gb) {
#pragma unroll
        for (int e = 0; e < KB; ++e) {
            const int g      = min(gb + e, gend - 1);
            const uint4* p   = reinterpret_cast<const uint4*>(codes + (base + static_cast<std::int64_t>(g) * 32) * 32);
            cw[e][0] = __ldg(p);
            cw[e][1] = __ldg(p + 1);
            sc[e]    = __ldg(scales + base + static_cast<std::int64_t>(g) * 32);
        }
    };
    auto compute = [&](const uint4 (&cw)[KB][2], const std::uint16_t (&sc)[KB], int gb) {
#pragma unroll
        for (int e = 0; e < KB; ++e) {
            const int g = gb + e;
            if (g >= gend) { break; }
            const half2 sc2 = __half2half2(__ushort_as_half(sc[e]));
            const std::uint32_t words[8] = {cw[e][0].x, cw[e][0].y, cw[e][0].z, cw[e][0].w,
                                            cw[e][1].x, cw[e][1].y, cw[e][1].z, cw[e][1].w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const std::uint32_t w0 = words[2 * u] ^ 0x80808080u;
                const std::uint32_t w1 = words[2 * u + 1] ^ 0x80808080u;
                // One byte permute per pair: (c0, 0x64, c1, 0x64) is half2(1024 + c0, 1024 + c1).
                std::uint32_t h[4] = {__byte_perm(w0, 0x64646464u, 0x5150), __byte_perm(w0, 0x64646464u, 0x5352),
                                      __byte_perm(w1, 0x64646464u, 0x5150), __byte_perm(w1, 0x64646464u, 0x5352)};
                half2 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) { b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&h[j]), bias), sc2); }
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
#pragma unroll
                for (int tile = 0; tile < TILES; ++tile) {
                    const int row = tile * 8 + r;
                    uint4 a4      = make_uint4(0u, 0u, 0u, 0u);
                    if (row < t) {
                        a4 = *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(row) * k + g * kGroupK + u * 8);
                    }
                    volta_mma_qp_n(c[tile], a4.x, a4.y, B[0], B[1]);
                    volta_mma_qp_n(c[tile], a4.z, a4.w, B[2], B[3]);
                }
            }
        }
    };
    load(ca, sa, g0);
    for (int gb = g0; gb < gend; gb += 2 * KB) {
        if (gb + KB < gend) { load(cb, sb, gb + KB); }
        compute(ca, sa, gb);
        if (gb + 2 * KB < gend) { load(ca, sa, gb + 2 * KB); }
        if (gb + KB < gend) { compute(cb, sb, gb + KB); }
    }
    // C map (v100-skinny mma8_probe.cu, roles swapped), as in the row-major QPN kernel.
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = tile * 8 + ((i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1));
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + cl] = c[tile][i];
        }
    }
    __syncthreads();
    for (int e = static_cast<int>(threadIdx.x); e < TILES * 8 * 32; e += SPLITK * 32) {
        const int row  = e / 32;
        const int ocol = static_cast<int>(blockIdx.x) * 32 + e % 32;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            out[static_cast<std::int64_t>(row) * out_ld + ocol] = __float2bfloat16(v);
        }
    }
}

struct Scratch {
    std::int64_t code_bytes  = 0;
    std::int64_t stage_elems = 0;
    std::uint8_t* codes      = nullptr;
    std::uint16_t* scales    = nullptr;
    half* stage              = nullptr;
};
Scratch g_scratch;

} // namespace

void w8_prepacked_reserve(std::int64_t n, std::int64_t k) {
    const std::int64_t code_bytes = n * k;
    if (code_bytes > g_scratch.code_bytes) {
        if (g_scratch.codes != nullptr) { CUDA_CHECK(cudaFree(g_scratch.codes)); }
        if (g_scratch.scales != nullptr) { CUDA_CHECK(cudaFree(g_scratch.scales)); }
        CUDA_CHECK(cudaMalloc(&g_scratch.codes, static_cast<std::size_t>(code_bytes)));
        CUDA_CHECK(cudaMalloc(&g_scratch.scales, static_cast<std::size_t>(code_bytes / kGroupK * 2)));
        g_scratch.code_bytes = code_bytes;
    }
    if (32 * k > g_scratch.stage_elems) {
        if (g_scratch.stage != nullptr) { CUDA_CHECK(cudaFree(g_scratch.stage)); }
        CUDA_CHECK(cudaMalloc(&g_scratch.stage, static_cast<std::size_t>(32 * k) * sizeof(half)));
        g_scratch.stage_elems = 32 * k;
    }
}

void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit || weight.n % 32 != 0 ||
        weight.k % kGroupK != 0 || weight.padded_shape[1] != weight.k) {
        throw std::invalid_argument("W8 QPN prepack: needs a row-split W8G32 weight with n % 32 == 0 and unpadded k");
    }
    w8_prepacked_reserve(weight.n, weight.k);
    const int groups           = weight.k / kGroupK;
    const std::int64_t code_b  = static_cast<std::int64_t>(weight.n) * weight.k;
    const std::int64_t scale_b = code_b / kGroupK * 2;
    CUDA_CHECK(cudaMemcpyAsync(g_scratch.codes, weight.qdata, static_cast<std::size_t>(code_b), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_scratch.scales, weight.scales, static_cast<std::size_t>(scale_b), cudaMemcpyDeviceToDevice, stream));
    const std::int64_t total = static_cast<std::int64_t>(weight.n / 32) * groups * 32;
    w8_pack_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        g_scratch.codes, g_scratch.scales, static_cast<std::uint8_t*>(const_cast<void*>(weight.qdata)),
        static_cast<std::uint16_t*>(const_cast<void*>(weight.scales)), weight.n, groups, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

void launch_w8_prepacked(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const int groups     = k / kGroupK;
    if (t > 32) {
        static int logged = 0;
        if (logged < 40) { ++logged; std::fprintf(stderr, "[w8pp] wide T=%d n=%d k=%d\n", t, n, k); }
        if (w8_cutlass_ready(n, k, t)) {   // prefill: dequantize to fp16 and one CUTLASS GEMM
            w8_cutlass_launch(x, w, out, stream);
            return;
        }
        // Wide T: row-major copy in the load-time scratch, then the existing Volta MMA route.
        const std::int64_t total = static_cast<std::int64_t>(n / 32) * groups * 32;
        w8_pack_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint16_t*>(w.scales),
            g_scratch.codes, g_scratch.scales, n, groups, true);
        CUDA_CHECK(cudaGetLastError());
        Weight row_major = w;
        row_major.qdata  = g_scratch.codes;
        row_major.scales = g_scratch.scales;
        row_major.layout = QuantLayout::RowSplit;
        if (w8_volta_mma_supported(n, k, t)) {
            launch_w8_volta_mma(x, row_major, out, stream);
        } else {
            launch_w8_simt_r8_c8(x, row_major, out, stream);
        }
        return;
    }
    const std::int32_t count = k * t;
    w8_stage_half_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), g_scratch.stage, count);
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(n / 32));
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint16_t*>(w.scales);
    auto* y            = static_cast<__nv_bfloat16*>(out.data);
    const half* xs = g_scratch.stage;
    // T <= 8 (decode and MTP verify): two groups in flight per warp (KB 2) instead of four, more resident warps.
    // Measured on the V100 rank shapes (bench_w8pp.cu, T=5): gate_up 15360x5120 126.0 -> 108.3 us (s4 kb2 m6),
    // down 5120x7680 63.6 -> 56.8 us (s8 kb2 m3).
    // NINFER_SM70_W8_KB2=0 goes back to the previous KB 4 configurations (A/B).
    static const bool kb2 = [] {
        const char* v = std::getenv("NINFER_SM70_W8_KB2");
        return v == nullptr || v[0] != '0';
    }();
    if (t <= 8 && kb2) {
        if (n < 8192) { w8_volta_prepacked_kernel<8, 2, 3, 1><<<grid, 256, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
        else { w8_volta_prepacked_kernel<4, 2, 6, 1><<<grid, 128, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
    } else if (t <= 8) {
        if (n < 8192) { w8_volta_prepacked_kernel<8, 4, 2, 1><<<grid, 256, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
        else { w8_volta_prepacked_kernel<4, 4, 4, 1><<<grid, 128, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
    } else if (t <= 16) {
        if (n < 8192) { w8_volta_prepacked_kernel<8, 4, 2, 2><<<grid, 256, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
        else { w8_volta_prepacked_kernel<4, 4, 3, 2><<<grid, 128, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
    } else {
        if (n < 8192) { w8_volta_prepacked_kernel<8, 2, 2, 4><<<grid, 256, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
        else { w8_volta_prepacked_kernel<4, 2, 3, 4><<<grid, 128, 0, stream>>>(codes, scales, xs, y, n, k, t, out_ld); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif

} // namespace ninfer::ops::detail
