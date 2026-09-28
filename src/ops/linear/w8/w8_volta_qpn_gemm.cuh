#pragma once

// Fused-dequant W8 x BF16 GEMM on Volta tensor cores, quadpair-split-N form (sm_70 only).
//
// The W8 sibling of q4_volta_qpn_gemm.cuh, for the same reason and on the shape where it matters
// most. The 27B output head is W8 [248320,5120] and the narrow verify widths are where it is worst:
//
//     T:            1       2       3       4       5       6       7       8
//     us:        1797    2122    3837    4469    5137    5794    5280    5302
//     GB/s:       752     637     352     303     263     234     257     256
//
// T=1 is already *at* the machine -- 752 GB/s against this card's ~794 GB/s measured ceiling, so
// the GEMV route leaves nothing on the table. Everything above it falls off a cliff, to a third of
// that by T=6, against a physical floor of ~1.8ms for reading the 1.31 GB weight once. Neither
// incumbent addresses it: sliced r8_c8 re-reads the weight per 8 output columns, and the 32x8
// fused route maps T to the 32-row A axis, so at T=4 it pads 28 of 32 rows with zeros and measures
// *worse* than SIMT (5702us).
//
// Quadpair-split-N puts T on the 8-row axis instead, which is the right shape for exactly this
// band. See the Q4 sibling for the mapping, the fragment maps (v100-skinny's, byte-verified) and
// why the four quadpairs share one activation tile.
//
// Two W8 specifics:
//
//   - kGroupK is 32 and kCodeBytesPerGroup is 32, so one group is 32 code bytes covering 32 k --
//     half the k per byte of the Q4 sibling, but the same 128B blocked read covers 4 groups, so
//     the loop structure is identical.
//   - The decode is the 8-bit magic-number identity already used by w8_volta_mma_gemm.cuh:
//     0x6400|u is exactly 1024+u as fp16, so biasing a code byte by XOR 0x80 into u = b+128 makes
//     it 1152+b, and one __hsub2 against 1152.0 recovers the signed code. One uint2 of codes is 8
//     adjacent k, decoding straight into the four half2 that two mma slices consume -- no
//     repacking, exactly as in Q4.

#include "ops/common/volta_mma.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 890)

struct W8VoltaQpnSchedule {
    static constexpr int kWarps       = 4;  // warps per CTA; they split K, not N
    static constexpr int kColsPerCta  = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile = 8;  // tokens per A tile (mma's M axis)
    static constexpr int kThreads     = kWarps * 32;
};

// `kTiles` is the number of 8-row A tiles (T <= 8 * kTiles); `kBlk` is how many W8 groups a lane
// reads before consuming any, in units of kCodeBytesPerGroup. See the Q4 sibling: a lane streams
// its own weight row, but a warp's 32 lanes stream 32 different rows at once, so consuming a whole
// 128B line per lane per iteration is what keeps that off DRAM.
template <int kTiles, int kBlk, bool kDynamicConvAdd = false, bool kSwiGluUp = false>
__global__ __launch_bounds__(W8VoltaQpnSchedule::kThreads, 8) void w8_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld, const __nv_bfloat16* __restrict__ finish_delta = nullptr,
    const __nv_bfloat16* __restrict__ base = nullptr,
    __nv_bfloat16* __restrict__ residual = nullptr, int width = 0) {
    using S = W8VoltaQpnSchedule;
    constexpr int kGroupK = W8RowSplitStorage::kGroupK;
    constexpr int kCodeB  = W8RowSplitStorage::kCodeBytesPerGroup;
    static_assert(kGroupK == kCodeB, "W8 stores one code byte per k");

    __shared__ float cs[S::kWarps][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // Quadpair index, and this lane's position inside it. `r` is both the A row (token) and the
    // B column local to the quadpair.
    const int qp = (lane >> 2) & 3;
    const int r  = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;

    const int groups = k / kGroupK;
    const int gq     = groups / S::kWarps;
    const int g0     = warp * gq;
    const int gend   = (warp == S::kWarps - 1) ? groups : g0 + gq;

    const std::uint8_t* crow =
        codes + static_cast<std::int64_t>(good ? col : 0) * padded_groups * kCodeB;
    const std::uint16_t* srow = reinterpret_cast<const std::uint16_t*>(
        scales + static_cast<std::int64_t>(good ? col : 0) * padded_groups *
                     W8RowSplitStorage::kScaleBytesPerGroup);

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0

    static_assert(kBlk == 4, "the vector scale load below assumes kBlk fp16 scales are 8 bytes");
    // 32 code bytes per group, so kBlk groups is kBlk * 2 uint4 and kBlk * 32 k-elements.
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = g0; gb < gend; gb += kBlk) {
        const int blk  = min(kBlk, gend - gb);
        const int last = gend - 1;
#pragma unroll
        for (int e = 0; e < kBlk; ++e) {
            const int g           = min(gb + e, last);
            const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * kCodeB;
            cw[2 * e]             = __ldg(reinterpret_cast<const uint4*>(p));
            cw[2 * e + 1]         = __ldg(reinterpret_cast<const uint4*>(p + 16));
        }
        // Scales blocked as well, and this matters more than it looks. One fp16 scale per 32 k
        // means the plane is 79 MB on this shape, and a 2-byte load per lane fetches a whole 32B
        // sector for it -- a 16x amplification that measured as 1.89 GB of DRAM read against a
        // 1.31 GB weight. Reading the block's kBlk scales as one vector cuts that to 4x.
        if (blk == kBlk && (gb % kBlk) == 0) {
            const uint2 packed =
                __ldg(reinterpret_cast<const uint2*>(srow + gb)); // kBlk == 4 fp16 scales
            const auto* sc = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int e = 0; e < kBlk; ++e) { scv[e] = good ? sc[e] : static_cast<std::uint16_t>(0); }
        } else {
#pragma unroll
            for (int e = 0; e < kBlk; ++e) {
                const int g = min(gb + e, last);
                scv[e]      = good ? srow[g] : static_cast<std::uint16_t>(0);
            }
        }

        for (int e = 0; e < blk; ++e) {
            const int g     = gb + e;
            const half2 sc2 = __half2half2(__ushort_as_half(scv[e]));
            const std::uint32_t words[8] = {cw[2 * e].x,     cw[2 * e].y,     cw[2 * e].z,
                                            cw[2 * e].w,     cw[2 * e + 1].x, cw[2 * e + 1].y,
                                            cw[2 * e + 1].z, cw[2 * e + 1].w};

#pragma unroll
            for (int u = 0; u < 4; ++u) {
                // One uint2 = 8 code bytes = 8 adjacent k, decoding into the four half2 that two
                // mma slices consume.
                const std::uint32_t w0 = words[2 * u] ^ 0x80808080u;
                const std::uint32_t w1 = words[2 * u + 1] ^ 0x80808080u;
                half2 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t src = (j < 2) ? w0 : w1;
                    const int shift         = (j & 1) * 16;
                    std::uint32_t bits =
                        (((src >> shift) & 0xffu) | (((src >> shift) & 0xff00u) << 8)) |
                        0x64006400u;
                    b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
                }
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                const int kbase   = g * kGroupK + u * 8;

#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const __nv_bfloat16* xrow =
                            x + static_cast<std::int64_t>(row) * k + kbase;
                        const uint4 raw = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src = reinterpret_cast<const __nv_bfloat16*>(&raw);
                        __half tmp[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tmp[j] = __float2half(__bfloat162float(src[j]));
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(a);
                    volta_mma_qp_n(c[tile], A[0], A[1], B[0], B[1]); // k slice 0
                    volta_mma_qp_n(c[tile], A[2], A[3], B[2], B[3]); // k slice 1
                }
            }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped); see the Q4 sibling.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += S::kThreads) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < S::kWarps; ++w) { v += cs[w][e]; }
            if constexpr (kDynamicConvAdd) {
                // Preserve the old materialized projection boundary exactly: convolution sees
                // BF16-rounded current and predecessor projections, even though both live in
                // this CTA's reduction tile. Width is request-local, so row%width guards against
                // carrying the second tap across batched request boundaries.
                constexpr int kHidden = 5120;
                constexpr int kGroups = 320;
                const int group = ocol / 16;
                const float projected0 = __bfloat162float(__float2bfloat16(v));
                const float base0 = __bfloat162float(base[ocol + 2LL * kHidden]);
                const float delta0 = __bfloat162float(
                    finish_delta[group + static_cast<std::int64_t>(kGroups) * (2 * row)]);
                float acc = (base0 + delta0) * projected0;
                if ((row % width) != 0) {
                    float previous = 0.0f;
#pragma unroll
                    for (int w = 0; w < S::kWarps; ++w) {
                        previous += cs[w][(row - 1) * S::kColsPerCta + cl];
                    }
                    const float projected1 = __bfloat162float(__float2bfloat16(previous));
                    const float base1 = __bfloat162float(base[ocol + 3LL * kHidden]);
                    const float delta1 = __bfloat162float(finish_delta[
                        group + static_cast<std::int64_t>(kGroups) * (1 + 2 * row)]);
                    acc += (base1 + delta1) * projected1;
                }
                const std::int64_t offset = ocol + static_cast<std::int64_t>(row) * kHidden;
                residual[offset] = __float2bfloat16_rn(__bfloat162float(residual[offset]) + acc);
            } else if constexpr (kSwiGluUp) {
                const std::int64_t offset = static_cast<std::int64_t>(row) * out_ld + ocol;
                const float gate = __bfloat162float(out[offset]);
                const float up = __bfloat162float(__float2bfloat16(v));
                out[offset] = __float2bfloat16_rn(silu(gate) * up);
            } else {
                out[static_cast<std::int64_t>(row) * out_ld + ocol] = __float2bfloat16(v);
            }
        }
    }
}

#ifdef NINFER_ADA_BUILD
// Ada (sm_89) W8 decode GEMV: the Volta kernel's CTA (32 output rows, warps splitting K, one barrier for the
// K reduce) on mma.m16n8k16, weights as the A operand (two 16-row halves), tokens on N (T <= 8). Lane (g, q)
// reads 8 contiguous code bytes k = 8q..8q+7 of a 32-k group from rows g and g+8 of each half, and the same 8
// activations of token g. The m16n8k16 k slots of lane q ({2q, 2q+1} and {2q+8, 2q+9} of two 16-k slices) are
// relabelled onto those 8 real k identically for A and B, so each slice still contracts matching k.
__device__ __forceinline__ void w8_ada_mma(float (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                           std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 8 signed code bytes -> 4 half2 (bytes 0-1, 2-3, 4-5, 6-7) times the group scale, as the Volta decode does.
__device__ __forceinline__ void w8_ada_decode(uint2 raw, half2 scale, std::uint32_t (&out)[4]) {
    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t src   = ((j < 2) ? raw.x : raw.y) ^ 0x80808080u;
        const int shift           = (j & 1) * 16;
        std::uint32_t bits = (((src >> shift) & 0xffu) | (((src >> shift) & 0xff00u) << 8)) | 0x64006400u;
        half2 v            = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), scale);
        out[j]             = *reinterpret_cast<std::uint32_t*>(&v);
    }
}

template <int SPLITK>
__global__ __launch_bounds__(SPLITK * 32) void w8_ada_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld) {
    constexpr int kCols   = 32;
    constexpr int kGroupK = W8RowSplitStorage::kGroupK;
    constexpr int kBlk    = 4;
    __shared__ float cs[SPLITK][8 * kCols];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int g      = lane >> 2;
    const int q      = lane & 3;
    const int groups = k / kGroupK;
    const int gq     = groups / SPLITK;
    const int g0     = warp * gq;
    const int gend   = warp == SPLITK - 1 ? groups : g0 + gq;

    // Rows g, g+8, g+16, g+24 of this CTA's 32 (clamped reads past n are discarded at the store).
    const std::uint8_t* crow[4];
    const std::uint16_t* srow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = static_cast<int>(blockIdx.x) * kCols + g + 8 * i;
        row     = row < n ? row : n - 1;
        crow[i] = codes + static_cast<std::int64_t>(row) * padded_groups * kGroupK + 8 * q;
        srow[i] = reinterpret_cast<const std::uint16_t*>(scales) + static_cast<std::int64_t>(row) * padded_groups;
    }
    const bool live = g < t;
    const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(live ? g : 0) * k + 8 * q;

    float c[2][2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { c[h][a][i] = 0.0f; }
        }
    }

    auto body = [&](int grp, int acc, const uint2 (&cw)[4], const std::uint16_t (&sc)[4]) {
        std::uint32_t b[2][2] = {{0u, 0u}, {0u, 0u}};
        if (live) {
            const uint4 raw = *reinterpret_cast<const uint4*>(xrow + static_cast<std::int64_t>(grp) * kGroupK);
            const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                half2 h2 = __halves2half2(__float2half(__bfloat162float(v[2 * j])),
                                          __float2half(__bfloat162float(v[2 * j + 1])));
                b[j >> 1][j & 1] = *reinterpret_cast<std::uint32_t*>(&h2);
            }
        }
        std::uint32_t w[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { w8_ada_decode(cw[i], __half2half2(__ushort_as_half(sc[i])), w[i]); }
        // Half h: rows (g, g+8) = w[2h], w[2h+1]. Slice 0 uses code pairs 0-1 and 2-3, slice 1 pairs 4-5 and 6-7.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            w8_ada_mma(c[h][acc], w[2 * h][0], w[2 * h + 1][0], w[2 * h][1], w[2 * h + 1][1], b[0][0], b[0][1]);
            w8_ada_mma(c[h][acc], w[2 * h][2], w[2 * h + 1][2], w[2 * h][3], w[2 * h + 1][3], b[1][0], b[1][1]);
        }
    };

    int gb = g0;
    for (; gb + kBlk <= gend; gb += kBlk) {
        uint2 cw[kBlk][4];
        std::uint16_t sc[kBlk][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int e = 0; e < kBlk; ++e) {
                cw[e][i] = __ldg(reinterpret_cast<const uint2*>(crow[i] + static_cast<std::int64_t>(gb + e) * kGroupK));
            }
            if ((gb % kBlk) == 0) {
                const uint2 packed = __ldg(reinterpret_cast<const uint2*>(srow[i] + gb));
                const auto* s4     = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
                for (int e = 0; e < kBlk; ++e) { sc[e][i] = s4[e]; }
            } else {
#pragma unroll
                for (int e = 0; e < kBlk; ++e) { sc[e][i] = __ldg(srow[i] + gb + e); }
            }
        }
#pragma unroll
        for (int e = 0; e < kBlk; ++e) { body(gb + e, e & 1, cw[e], sc[e]); }
    }
    for (; gb < gend; ++gb) {
        uint2 cw[4];
        std::uint16_t sc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            cw[i] = __ldg(reinterpret_cast<const uint2*>(crow[i] + static_cast<std::int64_t>(gb) * kGroupK));
            sc[i] = __ldg(srow[i] + gb);
        }
        body(gb, 0, cw, sc);
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int token = 2 * q + (i & 1);
            const int col   = h * 16 + g + 8 * (i >> 1);
            cs[warp][token * kCols + col] = c[h][0][i] + c[h][1][i];
        }
    }
    __syncthreads();
    for (int e = static_cast<int>(threadIdx.x); e < 8 * kCols; e += SPLITK * 32) {
        const int token = e / kCols;
        const int ocol  = static_cast<int>(blockIdx.x) * kCols + e % kCols;
        if (token < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            out[static_cast<std::int64_t>(token) * out_ld + ocol] = __float2bfloat16(v);
        }
    }
}
#endif // NINFER_ADA_BUILD

#endif // sm_70

} // namespace ninfer::ops::detail
