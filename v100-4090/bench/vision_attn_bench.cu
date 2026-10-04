// vision_attn_bench.cu — vision-encoder attention: tiled SIMT kernel (reference, equal to the original scalar kernel to 1 bf16 ulp)
// against the FP16 tensor-core kernel (WMMA), same random data. Build and run (from the repository root):
//   docker run --rm -v $PWD:/src -w /src ninfer-v100-4090/build:cuda12.8 nvcc -O3 -std=c++20 -arch=sm_70 -DNINFER_VOLTA_BUILD=1 \
//     -Isrc -Iinclude v100-4090/bench/vision_attn_bench.cu -o /src/vision_attn_bench
//   docker run --rm --gpus "device=<V100 UUID>" -v $PWD:/src ninfer-v100-4090/build:cuda12.8 /src/vision_attn_bench 12288 3072
// Reference (V100): tiled ~359 ms, tensor cores ~180 ms, mean relative error ~1.8e-4.
#include "ops/softmax_attention/dense/packed/volta.cuh"
#include "ops/softmax_attention/dense/packed/volta_wmma.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace ninfer::ops;
int main(int argc, char** argv) {
    const int seg0 = argc > 1 ? atoi(argv[1]) : 12288, seg1 = argc > 2 ? atoi(argv[2]) : 3072;
    const float amp = argc > 3 ? atof(argv[3]) : 4.0f;
    const int T = seg0 + seg1, H = kPackedAttentionHeads, D = kPackedAttentionHeadDim;
    const size_t n = (size_t)T * H * D;
    std::vector<__nv_bfloat16> hq(n), hk(n), hv(n);
    srand(1);
    auto rnd = [&] { return (rand() / (float)RAND_MAX - 0.5f) * amp; };
    for (size_t i = 0; i < n; ++i) {
        hq[i] = __float2bfloat16(rnd());
        hk[i] = __float2bfloat16(rnd());
        hv[i] = __float2bfloat16(rnd());
    }
    __nv_bfloat16 *q, *k, *v, *o1, *o2;
    int* cu;
    cudaMalloc(&q, n * 2); cudaMalloc(&k, n * 2); cudaMalloc(&v, n * 2);
    cudaMalloc(&o1, n * 2); cudaMalloc(&o2, n * 2); cudaMalloc(&cu, 12);
    cudaMemcpy(q, hq.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(k, hk.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(v, hv.data(), n * 2, cudaMemcpyHostToDevice);
    int hcu[3] = {0, seg0, T};
    cudaMemcpy(cu, hcu, 12, cudaMemcpyHostToDevice);
    const long sd = 1, sh = D, st = (long)H * D;
    cudaFuncSetAttribute(packed_attention_volta_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(PackedAttentionWmmaSmem));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float ms1, ms2;
    dim3 g1((T + kPackedAttentionTiledQ - 1) / kPackedAttentionTiledQ, H);
    dim3 g2((T + kPackedAttentionWmmaQ - 1) / kPackedAttentionWmmaQ, H);
    auto tiled = [&] {
        packed_attention_volta_tiled_kernel<<<g1, kPackedAttentionTiledThreads>>>(
            q, k, v, cu, 2, 0, T, o1, sd, sh, st, sd, sh, st, sd, sh, st);
    };
    auto tc = [&] {
        packed_attention_volta_wmma_kernel<<<g2, kPackedAttentionWmmaThreads, sizeof(PackedAttentionWmmaSmem)>>>(
            q, k, v, cu, 2, 0, T, o2, sd, sh, st, sd, sh, st, sd, sh, st);
    };
    tiled(); tc();
    cudaDeviceSynchronize();
    cudaEventRecord(a); for (int r = 0; r < 3; ++r) tiled(); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b); ms1 /= 3;
    cudaEventRecord(a); for (int r = 0; r < 10; ++r) tc(); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms2, a, b); ms2 /= 10;
    cudaError_t e = cudaGetLastError();
    if (e) { printf("CUDA %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<__nv_bfloat16> r1(n), r2(n);
    cudaMemcpy(r1.data(), o1, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2.data(), o2, n * 2, cudaMemcpyDeviceToHost);
    double maxd = 0, maxv = 0, sum = 0, sumv = 0;
    size_t diff = 0, nan = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = __bfloat162float(r1[i]), y = __bfloat162float(r2[i]);
        if (std::isnan(y)) { ++nan; continue; }
        maxd = fmax(maxd, fabs(x - y)); maxv = fmax(maxv, fabs(x));
        sum += fabs(x - y); sumv += fabs(x); diff += x != y;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s · patches %d+%d amplitude %.0f · tiled %.1f ms · tensor cores %.1f ms · %.1fx · max diff %.2e (values up to %.2f), "
           "mean relative %.2e, different %.1f%%, NaN %zu · smem %zu B\n",
           p.name, seg0, seg1, amp, ms1, ms2, ms1 / ms2, maxd, maxv, sum / sumv, 100.0 * diff / n, nan,
           sizeof(PackedAttentionWmmaSmem));
}
