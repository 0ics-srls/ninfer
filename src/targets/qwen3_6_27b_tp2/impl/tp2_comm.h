#pragma once

// Tensor-parallel (TP2) collectives for the qwen3_6_27b_tp2 target.
//
// Each rank is its own ninfer process pinned to one GPU (CUDA_VISIBLE_DEVICES), loading its rank
// artifact (tools/tp2/shard_qwen38_27b.py). A front proxy sends identical requests to both, so the
// two engines execute the same op sequence; the collectives below are the only coupling.
//
// Environment: NINFER_TP_RANK (0 or 1) and NINFER_TP_ID_FILE (rank 0 writes the NCCL unique id
// there, rank 1 reads it; the launcher deletes it before starting both ranks).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {

// Creates the communicator and warms every message size used by decode, outside graph capture.
void init();

[[nodiscard]] int rank();

// residual (BF16) <- sum over ranks, in place. `slot` labels the block for NINFER_TP_STATS:
// 0 attention, 1 GDN, 2 MLP, 3 other.
void allreduce(Tensor& residual, cudaStream_t stream, int slot = 3);

// residual <- residual + sum over ranks of partial: rank 0 adds its partial, rank 1 replaces the
// residual with its partial, then one all-reduce. With the 8-bit wire (wire8_wanted) the partials are
// exchanged as int8 + per-64 scales instead and the residual itself never crosses the wire.
void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream, int slot = 3);

// NINFER_TP_WIRE8=1 and a reduction too large for the mailbox (prefill): the callers that fuse their
// partial into the residual (linear_add) should produce a separate partial and call combine_partial.
// NINFER_TP_WIRE8=2 keeps the wire to the MLP (slot 2) only, for A/B.
[[nodiscard]] bool wire8_wanted(std::int64_t elements, int slot);

// Vocabulary-sharded output head: this rank computes rows [rank*N/2, (rank+1)*N/2) of
// hidden x head^T and both ranks end with the full [N, T] out (bit-identical on both ranks).
// Returns false when the head/shape is not shardable here; the caller then runs it whole.
bool head_linear(const Tensor& hidden, const Weight& head, Tensor& out, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
