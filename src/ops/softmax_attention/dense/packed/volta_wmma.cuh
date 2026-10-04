#pragma once

#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/packed/kernel.cuh"

#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

namespace ninfer::ops {

// Vision attention on FP16 tensor cores (WMMA 16x16x16, FP32 accumulate), for the Volta path of both ranks.
// One block = 64 queries of one head (16 per warp); K/V staged 64 keys at a time. BF16 -> FP16 is exact for
// these magnitudes (FP16 has 10 mantissa bits, BF16 7). Scores, softmax and the output accumulator stay FP32.
inline constexpr int kPackedAttentionWmmaQ       = 64;
inline constexpr int kPackedAttentionWmmaK       = 64;
inline constexpr int kPackedAttentionWmmaThreads = 128;
inline constexpr int kPackedAttentionWmmaD       = 80;  // 72 padded to a multiple of 16
inline constexpr int kPackedAttentionWmmaLdH     = 88;  // half row stride (multiple of 8)
inline constexpr int kPackedAttentionWmmaLdS     = 68;  // float score row stride (multiple of 4)
inline constexpr int kPackedAttentionWmmaLdP     = 72;  // half probability row stride
inline constexpr int kPackedAttentionWmmaLdO     = 84;  // float output row stride

struct PackedAttentionWmmaSmem {
    __half q[kPackedAttentionWmmaQ][kPackedAttentionWmmaLdH];
    __half k[kPackedAttentionWmmaK][kPackedAttentionWmmaLdH];
    __half v[kPackedAttentionWmmaK][kPackedAttentionWmmaLdH];
    float s[4][16][kPackedAttentionWmmaLdS];
    __half p[4][16][kPackedAttentionWmmaLdP];
    float o[4][16][kPackedAttentionWmmaLdO];
    float corr[4][16];
    int begin[kPackedAttentionWmmaQ];
    int end[kPackedAttentionWmmaQ];
    int range[2];
};

__launch_bounds__(kPackedAttentionWmmaThreads) __global__ void packed_attention_volta_wmma_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ cu_seqlens,
    int segments, int uniform_segment_length, int tokens, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    using namespace nvcuda;
    constexpr int D           = kPackedAttentionHeadDim;
    constexpr int DP          = kPackedAttentionWmmaD;
    constexpr int TQ          = kPackedAttentionWmmaQ;
    constexpr int TK          = kPackedAttentionWmmaK;
    constexpr int Threads     = kPackedAttentionWmmaThreads;
    constexpr float ScaleLog2 = 0.11785113019775792073f * 1.4426950408889634f;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    PackedAttentionWmmaSmem& sm = *reinterpret_cast<PackedAttentionWmmaSmem*>(smem_raw);

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int head = static_cast<int>(blockIdx.y);
    const int q0   = static_cast<int>(blockIdx.x) * TQ;

    if (tid < TQ) {
        const int token = q0 + tid;
        int begin = 0, end = 0;
        if (token < tokens) {
            if (uniform_segment_length > 0) {
                begin = (token / uniform_segment_length) * uniform_segment_length;
                end   = min(tokens, begin + uniform_segment_length);
            } else {
                for (int segment = 0; segment < segments; ++segment) {
                    if (token >= cu_seqlens[segment] && token < cu_seqlens[segment + 1]) {
                        begin = cu_seqlens[segment];
                        end   = cu_seqlens[segment + 1];
                        break;
                    }
                }
            }
        }
        sm.begin[tid] = begin;
        sm.end[tid]   = end;
    }
    for (int i = tid; i < TQ * DP; i += Threads) {
        const int r = i / DP, d = i % DP, token = q0 + r;
        float x = 0.0f;
        if (token < tokens && d < D) {
            x = __bfloat162float(
                *packed_attention_ptr(q, q_stride_d, q_stride_h, q_stride_t, d, head, token));
        }
        sm.q[r][d] = __float2half_rn(x);
    }
    for (int i = lane; i < 16 * kPackedAttentionWmmaLdO; i += kWarpSize) {
        sm.o[warp][i / kPackedAttentionWmmaLdO][i % kPackedAttentionWmmaLdO] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        int lo = INT_MAX, hi = 0;
        for (int r = 0; r < TQ; ++r) {
            if (sm.end[r] > sm.begin[r]) {
                lo = min(lo, sm.begin[r]);
                hi = max(hi, sm.end[r]);
            }
        }
        sm.range[0] = lo == INT_MAX ? 0 : lo;
        sm.range[1] = hi;
    }
    __syncthreads();
    const int key_begin = sm.range[0], key_end = sm.range[1];

    // Lane pair (2r, 2r+1) owns row r of the warp's 16 rows: 32 score columns each.
    const int row    = lane >> 1;
    const int half   = lane & 1;
    const int qrow   = warp * 16 + row;
    const int qb     = sm.begin[qrow];
    const int qe     = sm.end[qrow];
    float row_max    = -CUDART_INF_F;
    float row_sum    = 0.0f;
    int warp_lo = INT_MAX, warp_hi = 0;
    for (int r = 0; r < 16; ++r) {
        const int b = sm.begin[warp * 16 + r], e = sm.end[warp * 16 + r];
        if (e > b) {
            warp_lo = min(warp_lo, b);
            warp_hi = max(warp_hi, e);
        }
    }

    for (int kt = key_begin; kt < key_end; kt += TK) {
        __syncthreads();
        for (int i = tid; i < TK * DP; i += Threads) {
            const int r = i / DP, d = i % DP, key = kt + r;
            float xk = 0.0f, xv = 0.0f;
            if (key < key_end && d < D) {
                xk = __bfloat162float(
                    *packed_attention_ptr(k, k_stride_d, k_stride_h, k_stride_t, d, head, key));
                xv = __bfloat162float(
                    *packed_attention_ptr(v, v_stride_d, v_stride_h, v_stride_t, d, head, key));
            }
            sm.k[r][d] = __float2half_rn(xk);
            sm.v[r][d] = __float2half_rn(xv);
        }
        __syncthreads();
        if (warp_hi <= warp_lo || kt + TK <= warp_lo || kt >= warp_hi) { continue; } // warp-uniform

        // S = Q K^T for this warp's 16 rows and the 64 keys of the tile.
#pragma unroll
        for (int n = 0; n < TK / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < DP / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, &sm.q[warp * 16][kk * 16], kPackedAttentionWmmaLdH);
                wmma::load_matrix_sync(b, &sm.k[n * 16][kk * 16], kPackedAttentionWmmaLdH);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(&sm.s[warp][0][n * 16], acc, kPackedAttentionWmmaLdS,
                                    wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax on the 16x64 scores: each lane does half a row.
        float local_max = -CUDART_INF_F;
        float s_reg[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int col = half * 32 + j, key = kt + col;
            const float s = key >= qb && key < qe ? sm.s[warp][row][col] * ScaleLog2 : -CUDART_INF_F;
            s_reg[j]  = s;
            local_max = fmaxf(local_max, s);
        }
        local_max        = fmaxf(local_max, __shfl_xor_sync(kFullWarpMask, local_max, 1));
        const float next = fmaxf(row_max, local_max);
        const float corr = row_max == -CUDART_INF_F ? 0.0f : exp2f(row_max - next);
        float psum       = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float p = s_reg[j] == -CUDART_INF_F ? 0.0f : exp2f(s_reg[j] - next);
            psum += p;
            sm.p[warp][row][half * 32 + j] = __float2half_rn(p);
        }
        psum += __shfl_xor_sync(kFullWarpMask, psum, 1);
        row_sum = row_sum * corr + psum;
        row_max = next;
        if (half == 0) { sm.corr[warp][row] = corr; }
        __syncwarp();
        for (int i = lane; i < 16 * DP; i += kWarpSize) {
            const int r = i / DP, c = i % DP;
            sm.o[warp][r][c] *= sm.corr[warp][r];
        }
        __syncwarp();

        // O += P V
#pragma unroll
        for (int n = 0; n < DP / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, &sm.o[warp][0][n * 16], kPackedAttentionWmmaLdO,
                                   wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < TK / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, &sm.p[warp][0][kk * 16], kPackedAttentionWmmaLdP);
                wmma::load_matrix_sync(b, &sm.v[kk * 16][n * 16], kPackedAttentionWmmaLdH);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(&sm.o[warp][0][n * 16], acc, kPackedAttentionWmmaLdO,
                                    wmma::mem_row_major);
        }
        __syncwarp();
    }

    if (half == 0) { sm.corr[warp][row] = row_sum > 0.0f ? 1.0f / row_sum : 0.0f; }
    __syncwarp();
    for (int i = lane; i < 16 * D; i += kWarpSize) {
        const int r = i / D, d = i % D, token = q0 + warp * 16 + r;
        if (token >= tokens || sm.end[warp * 16 + r] <= sm.begin[warp * 16 + r]) { continue; }
        out[(static_cast<std::int64_t>(token) * kPackedAttentionHeads + head) * D + d] =
            __float2bfloat16_rn(sm.o[warp][r][d] * sm.corr[warp][r]);
    }
}

} // namespace ninfer::ops
