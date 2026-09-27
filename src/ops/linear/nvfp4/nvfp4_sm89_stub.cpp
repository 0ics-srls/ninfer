// sm89 port: link-time stubs for the Blackwell-only NVFP4 kernels and their
// host dispatch layer. Engines built for sm_89 (Ada) only run groupwise-int /
// q4-q5 / w8 weight profiles, which never dispatch into these entry points;
// abort loudly if one is ever reached anyway.

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_unavailable() {
    std::fputs("ninfer: NVFP4 kernels require a Blackwell GPU (sm_120a); "
               "this build targets sm_89 and ships without them.\n",
               stderr);
    std::abort();
}

} // namespace

// --- W4A4 TMA kernels (non-RDC archive) -------------------------------------

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    nvfp4_unavailable();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) {
    nvfp4_unavailable();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    nvfp4_unavailable();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    nvfp4_unavailable();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                         std::int32_t, float, cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    nvfp4_unavailable();
}

// --- Per-op plan dispatch layers --------------------------------------------

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    return 0;
}

void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, WorkspaceArena*, cudaStream_t) {
    nvfp4_unavailable();
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    return 0;
}

void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              WorkspaceArena*, cudaStream_t) {
    nvfp4_unavailable();
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    return Nvfp4GdnConvPlan{Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t,
                                                        std::int32_t) {
    return 0;
}

void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t) {
    nvfp4_unavailable();
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena&,
                               cudaStream_t) {
    nvfp4_unavailable();
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t,
                                                         std::int32_t) {
    return 0;
}

void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                  WorkspaceArena&, cudaStream_t) {
    nvfp4_unavailable();
}

// --- KV append and causal attention launchers --------------------------------

void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    nvfp4_unavailable();
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                        cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                           std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                           Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    nvfp4_unavailable();
}

// --- Linear dispatch (dense MLP path) ----------------------------------------

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    return 0;
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                    cudaStream_t) {
    nvfp4_unavailable();
}

// --- GDN record path ----------------------------------------------------------

void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unavailable();
}

} // namespace ninfer::ops::detail
