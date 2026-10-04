#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/packed/kernel.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <climits>

namespace ninfer::ops {

inline constexpr int kPackedAttentionVoltaThreads = 128;
inline constexpr int kPackedAttentionVoltaQueriesPerBlock =
    kPackedAttentionVoltaThreads / kWarpSize;

__launch_bounds__(kPackedAttentionVoltaThreads, 1) __global__ void
packed_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ cu_seqlens,
    int segments, int uniform_segment_length, int tokens, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    constexpr int D = kPackedAttentionHeadDim;
    constexpr float Scale = 0.11785113019775792073f;

    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp  = static_cast<int>(threadIdx.x) / kWarpSize;
    const int token = static_cast<int>(blockIdx.x) * kPackedAttentionVoltaQueriesPerBlock + warp;
    const int head  = static_cast<int>(blockIdx.y);
    if (token >= tokens) { return; }

    int begin = 0;
    int end   = 0;
    if (lane == 0) {
        if (uniform_segment_length > 0) {
            begin = (token / uniform_segment_length) * uniform_segment_length;
            end   = min(tokens, begin + uniform_segment_length);
        } else {
            for (int segment = 0; segment < segments; ++segment) {
                const int candidate_begin = cu_seqlens[segment];
                const int candidate_end   = cu_seqlens[segment + 1];
                if (token >= candidate_begin && token < candidate_end) {
                    begin = candidate_begin;
                    end   = candidate_end;
                    break;
                }
            }
        }
    }
    begin = __shfl_sync(kFullWarpMask, begin, 0);
    end   = __shfl_sync(kFullWarpMask, end, 0);

    float q_values[3] = {};
    float acc[3]      = {};
#pragma unroll
    for (int item = 0; item < 3; ++item) {
        const int d = lane + item * kWarpSize;
        if (d < D) {
            q_values[item] = __bfloat162float(
                *packed_attention_ptr(q, q_stride_d, q_stride_h, q_stride_t, d, head, token));
        }
    }
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    for (int key_token = begin; key_token < end; ++key_token) {
        float dot = 0.0f;
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            const int d = lane + item * kWarpSize;
            if (d < D) {
                dot += q_values[item] * __bfloat162float(*packed_attention_ptr(
                                            k, k_stride_d, k_stride_h, k_stride_t, d, head,
                                            key_token));
            }
        }
        dot = warp_reduce_sum(dot);

        float old_scale   = 0.0f;
        float probability = 0.0f;
        if (lane == 0) {
            const float score    = dot * Scale;
            const float next_max = fmaxf(row_max, score);
            old_scale = row_max == -CUDART_INF_F
                            ? 0.0f
                            : exp2_approx((row_max - next_max) * 1.4426950408889634f);
            probability = exp2_approx((score - next_max) * 1.4426950408889634f);
            row_sum      = row_sum * old_scale + probability;
            row_max      = next_max;
        }
        old_scale   = __shfl_sync(kFullWarpMask, old_scale, 0);
        probability = __shfl_sync(kFullWarpMask, probability, 0);
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            const int d = lane + item * kWarpSize;
            if (d < D) {
                const float value = __bfloat162float(*packed_attention_ptr(
                    v, v_stride_d, v_stride_h, v_stride_t, d, head, key_token));
                acc[item] = acc[item] * old_scale + probability * value;
            }
        }
    }

    row_sum = __shfl_sync(kFullWarpMask, row_sum, 0);
#pragma unroll
    for (int item = 0; item < 3; ++item) {
        const int d = lane + item * kWarpSize;
        if (d < D) {
            out[(static_cast<std::int64_t>(token) * kPackedAttentionHeads + head) * D + d] =
                __float2bfloat16_rn(row_sum > 0.0f ? acc[item] / row_sum : 0.0f);
        }
    }
}


// Tiled SIMT variant (NINFER_VISION_ATTN, default on): one block = 32 queries of one head, K/V staged in
// shared memory 64 keys at a time and reused by all 32 queries, FP32 throughout (no FP16 range risk on
// Volta). The scalar kernel above re-reads K/V from global memory for every query and does a warp
// reduction per key: ~40 s per phone photo on both ranks.
inline constexpr int kPackedAttentionTiledQ       = 32;
inline constexpr int kPackedAttentionTiledK       = 64;
inline constexpr int kPackedAttentionTiledThreads = 128;
inline constexpr int kPackedAttentionTiledLd      = kPackedAttentionHeadDim + 1;

__launch_bounds__(kPackedAttentionTiledThreads) __global__ void packed_attention_volta_tiled_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ cu_seqlens,
    int segments, int uniform_segment_length, int tokens, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    constexpr int D           = kPackedAttentionHeadDim;
    constexpr int TQ          = kPackedAttentionTiledQ;
    constexpr int TK          = kPackedAttentionTiledK;
    constexpr int LD          = kPackedAttentionTiledLd;
    constexpr int Warps       = kPackedAttentionTiledThreads / kWarpSize;
    constexpr int QPerW       = TQ / Warps;
    constexpr float ScaleLog2 = 0.11785113019775792073f * 1.4426950408889634f;

    __shared__ float sq[TQ][LD];
    __shared__ float sk[TK][LD];
    __shared__ float sv[TK][LD];
    __shared__ float sp[Warps][TK];
    __shared__ int sb[TQ];
    __shared__ int se[TQ];
    __shared__ int range[2];

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
        sb[tid] = begin;
        se[tid] = end;
    }
    for (int i = tid; i < TQ * D; i += kPackedAttentionTiledThreads) {
        const int r = i / D, d = i % D, token = q0 + r;
        sq[r][d] = token < tokens ? __bfloat162float(*packed_attention_ptr(
                                        q, q_stride_d, q_stride_h, q_stride_t, d, head, token))
                                  : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        int lo = INT_MAX, hi = 0;
        for (int r = 0; r < TQ; ++r) {
            if (se[r] > sb[r]) {
                lo = min(lo, sb[r]);
                hi = max(hi, se[r]);
            }
        }
        range[0] = lo == INT_MAX ? 0 : lo;
        range[1] = hi;
    }
    __syncthreads();
    const int key_begin = range[0], key_end = range[1];

    float m[QPerW], l[QPerW], acc[QPerW][3];
#pragma unroll
    for (int i = 0; i < QPerW; ++i) {
        m[i]      = -CUDART_INF_F;
        l[i]      = 0.0f;
        acc[i][0] = acc[i][1] = acc[i][2] = 0.0f;
    }

    for (int kt = key_begin; kt < key_end; kt += TK) {
        __syncthreads();
        for (int i = tid; i < TK * D; i += kPackedAttentionTiledThreads) {
            const int r = i / D, d = i % D, key = kt + r;
            const bool ok = key < key_end;
            sk[r][d] = ok ? __bfloat162float(*packed_attention_ptr(k, k_stride_d, k_stride_h,
                                                                    k_stride_t, d, head, key))
                          : 0.0f;
            sv[r][d] = ok ? __bfloat162float(*packed_attention_ptr(v, v_stride_d, v_stride_h,
                                                                    v_stride_t, d, head, key))
                          : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < QPerW; ++i) {
            const int r  = warp * QPerW + i;
            const int qb = sb[r], qe = se[r];
            if (qe <= qb || kt + TK <= qb || kt >= qe) { continue; } // warp-uniform
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll 8
            for (int d = 0; d < D; ++d) {
                const float qd = sq[r][d];
                s0             = fmaf(qd, sk[lane][d], s0);
                s1             = fmaf(qd, sk[lane + kWarpSize][d], s1);
            }
            const int key0 = kt + lane, key1 = key0 + kWarpSize;
            s0 = key0 >= qb && key0 < qe ? s0 * ScaleLog2 : -CUDART_INF_F;
            s1 = key1 >= qb && key1 < qe ? s1 * ScaleLog2 : -CUDART_INF_F;
            float tile_max = fmaxf(s0, s1);
#pragma unroll
            for (int o = kWarpSize / 2; o > 0; o >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullWarpMask, tile_max, o));
            }
            const float next = fmaxf(m[i], tile_max);
            const float corr = m[i] == -CUDART_INF_F ? 0.0f : exp2f(m[i] - next);
            const float p0   = s0 == -CUDART_INF_F ? 0.0f : exp2f(s0 - next);
            const float p1   = s1 == -CUDART_INF_F ? 0.0f : exp2f(s1 - next);
            float psum       = p0 + p1;
#pragma unroll
            for (int o = kWarpSize / 2; o > 0; o >>= 1) {
                psum += __shfl_xor_sync(kFullWarpMask, psum, o);
            }
            l[i]                       = l[i] * corr + psum;
            m[i]                       = next;
            sp[warp][lane]             = p0;
            sp[warp][lane + kWarpSize] = p1;
            __syncwarp();
            float a0 = acc[i][0] * corr, a1 = acc[i][1] * corr, a2 = acc[i][2] * corr;
#pragma unroll 8
            for (int j = 0; j < TK; ++j) {
                const float pj = sp[warp][j];
                a0             = fmaf(pj, sv[j][lane], a0);
                a1             = fmaf(pj, sv[j][lane + kWarpSize], a1);
                if (lane + 2 * kWarpSize < D) { a2 = fmaf(pj, sv[j][lane + 2 * kWarpSize], a2); }
            }
            acc[i][0] = a0;
            acc[i][1] = a1;
            acc[i][2] = a2;
            __syncwarp();
        }
    }
#pragma unroll
    for (int i = 0; i < QPerW; ++i) {
        const int r = warp * QPerW + i, token = q0 + r;
        if (token >= tokens || se[r] <= sb[r]) { continue; }
        const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        __nv_bfloat16* o =
            out + (static_cast<std::int64_t>(token) * kPackedAttentionHeads + head) * D;
        o[lane]             = __float2bfloat16_rn(acc[i][0] * inv);
        o[lane + kWarpSize] = __float2bfloat16_rn(acc[i][1] * inv);
        if (lane + 2 * kWarpSize < D) {
            o[lane + 2 * kWarpSize] = __float2bfloat16_rn(acc[i][2] * inv);
        }
    }
}

} // namespace ninfer::ops
