#include "targets/qwen3_6_27b_tp2/impl/tp2_comm.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/residual_add.h"

#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

ncclComm_t g_comm = nullptr;
int g_rank        = -1;

void nccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("TP2 ") + what + ": " + ncclGetErrorString(result));
    }
}

ncclUniqueId exchange_id(int rank, const std::string& path) {
    ncclUniqueId id{};
    if (rank == 0) {
        nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
            if (!out) { throw std::runtime_error("TP2: cannot write " + tmp); }
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("TP2: cannot publish " + path);
        }
        return id;
    }
    for (int attempt = 0; attempt < 6000; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in && in.read(reinterpret_cast<char*>(&id), sizeof(id))) { return id; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("TP2: timed out waiting for rank 0 id at " + path);
}

} // namespace

void init() {
    if (g_comm != nullptr) { return; }
    const char* rank_env = std::getenv("NINFER_TP_RANK");
    const char* file_env = std::getenv("NINFER_TP_ID_FILE");
    if (rank_env == nullptr || file_env == nullptr) {
        throw std::runtime_error("TP2 target requires NINFER_TP_RANK and NINFER_TP_ID_FILE");
    }
    const int rank = std::atoi(rank_env);
    if (rank != 0 && rank != 1) { throw std::runtime_error("TP2: NINFER_TP_RANK must be 0 or 1"); }
    const ncclUniqueId id = exchange_id(rank, file_env);
    ncclComm_t comm       = nullptr;
    nccl_check(ncclCommInitRank(&comm, 2, id, rank), "ncclCommInitRank");

    // NCCL connects lazily on the first collective; do that (for every decode-sized message and a
    // prefill-sized one) before any graph capture.
    void* buffer        = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer, std::size_t{32} << 20));
    CUDA_CHECK(cudaMemset(buffer, 0, std::size_t{32} << 20));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (std::size_t bytes : {std::size_t{10240}, std::size_t{40960}, std::size_t{163840},
                              std::size_t{1} << 20, std::size_t{21} << 20}) {
        nccl_check(ncclAllReduce(buffer, buffer, bytes / 2, ncclBfloat16, ncclSum, comm, stream),
                   "warm-up all-reduce");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(buffer));
    g_comm = comm;
    g_rank = rank;
    std::fprintf(stderr, "[ninfer] TP2 rank %d/2 ready (NCCL %d)\n", rank, NCCL_VERSION_CODE);
}

int rank() { return g_rank; }

void allreduce(Tensor& residual, cudaStream_t stream) {
    if (g_comm == nullptr) { throw std::logic_error("TP2 collectives used before init()"); }
    if (residual.dtype != DType::BF16 || !residual.is_contiguous()) {
        throw std::invalid_argument("TP2 all-reduce requires a contiguous BF16 tensor");
    }
    nccl_check(ncclAllReduce(residual.data, residual.data, static_cast<std::size_t>(residual.numel()),
                             ncclBfloat16, ncclSum, g_comm, stream),
               "ncclAllReduce");
}

void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream) {
    if (g_rank == 0) {
        ops::residual_add(partial, residual, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(residual.data, partial.data, residual.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    allreduce(residual, stream);
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
