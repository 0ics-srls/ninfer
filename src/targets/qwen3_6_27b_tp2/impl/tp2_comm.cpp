#include "targets/qwen3_6_27b_tp2/impl/tp2_comm.h"
#include "runtime/engine/tp_lockstep.h"
#include "targets/qwen3_6_27b_tp2/impl/tp2_mailbox.h"
#include "targets/qwen3_6_27b_tp2/impl/tp2_wire8.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ninfer/ops/residual_add.h"

#include <cuda_bf16.h>

#include <nccl.h>

#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

ncclComm_t g_comm = nullptr;
int g_rank        = -1;
MailboxShared* g_mailbox_dev = nullptr;  // device alias of the shared host block (nullptr: NCCL only)
std::uint64_t* g_mailbox_steps = nullptr;
bool g_mailbox_stats           = false;  // NINFER_TP_STATS=1: wait time per rank in the mailbox padding
std::uint64_t* g_gather_steps  = nullptr;
__nv_bfloat16* g_head_local    = nullptr;  // [N/2, T] scratch for the sharded heads
void* g_head_act               = nullptr;  // fp16 staging of the head activation (K x T <= 5120 x 32)

// Row slice of a head weight. Q4G64 row-split and FP8 row-scaled (plain or QPN-prepacked, whose
// 32-row CTA tiles are laid out in row order) are both addressable by a row offset.
bool head_row_view(const Weight& w, std::int32_t row_begin, std::int32_t rows, Weight& out) {
    out = w;
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (row_begin % 32 != 0) { return false; }
        out.qdata  = static_cast<const std::byte*>(w.qdata) + static_cast<std::uint64_t>(row_begin) * w.k;
        out.scales = static_cast<const std::byte*>(w.scales) + static_cast<std::uint64_t>(row_begin) * 2;
        out.scale_ne[0] = rows;
        for (int d = 1; d < 4; ++d) { out.scale_nb[d] = static_cast<std::int64_t>(rows) * 2; }
    } else if (w.qtype == QType::Q4G64_F16S && w.layout == QuantLayout::RowSplit) {
        const std::uint64_t groups = static_cast<std::uint64_t>(w.padded_shape[1] / w.group);
        out.qdata  = static_cast<const std::byte*>(w.qdata) + static_cast<std::uint64_t>(row_begin) * groups * 32;
        out.scales = static_cast<const std::byte*>(w.scales) + static_cast<std::uint64_t>(row_begin) * groups * 2;
    } else {
        return false;
    }
    out.n               = rows;
    out.shape[0]        = rows;
    out.padded_shape[0] = rows;
    return true;
}

std::uint64_t id_nonce(const ncclUniqueId& id) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char c : id.internal) { h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL; }
    return h | 1;  // never zero (the unset value)
}

// Rank 0 creates and zeroes the block, then publishes this run's nonce; rank 1 attaches only when it
// sees the same nonce, so a stale file from an earlier run can never be mistaken for this one.
void init_mailbox(int rank, const std::string& path, std::uint64_t nonce) {
    const std::size_t bytes = (sizeof(MailboxShared) + 4095) / 4096 * 4096;
    void* host = nullptr;
    if (rank == 0) {
        // Build the block under a private name and publish it with an atomic rename: the peer may
        // still have a previous run's file mapped, and truncating that inode in place would SIGBUS it.
        const std::string tmp = path + ".tmp";
        ::unlink(tmp.c_str());
        const int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            throw std::runtime_error("TP2: cannot create mailbox " + tmp);
        }
        host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (host == MAP_FAILED) { throw std::runtime_error("TP2: mailbox mmap failed"); }
        std::memset(host, 0, bytes);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile std::uint64_t*>(&static_cast<MailboxShared*>(host)->magic) = nonce;
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("TP2: cannot publish mailbox " + path);
        }
    } else {
        // Re-open until the file carries this run's nonce (an older file is simply skipped).
        for (int attempt = 0;; ++attempt) {
            if (attempt > 6000) { throw std::runtime_error("TP2: timed out waiting for mailbox"); }
            const int fd = ::open(path.c_str(), O_RDWR);
            struct stat st {};
            if (fd >= 0 && ::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) == bytes) {
                void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                ::close(fd);
                if (p != MAP_FAILED) {
                    if (*reinterpret_cast<volatile std::uint64_t*>(&static_cast<MailboxShared*>(p)->magic) == nonce) {
                        host = p;
                        break;
                    }
                    ::munmap(p, bytes);
                }
            } else if (fd >= 0) {
                ::close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    std::fprintf(stderr, "[ninfer] TP2 rank %d: mailbox %s attached\n", rank, path.c_str());
    CUDA_CHECK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    std::fprintf(stderr, "[ninfer] TP2 rank %d: mailbox registered\n", rank);
    void* dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    CUDA_CHECK(cudaMalloc(&g_mailbox_steps, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMemset(g_mailbox_steps, 0, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMalloc(&g_gather_steps, sizeof(std::uint64_t) * kGatherMaxBlocks));
    CUDA_CHECK(cudaMemset(g_gather_steps, 0, sizeof(std::uint64_t) * kGatherMaxBlocks));
    CUDA_CHECK(cudaMalloc(&g_head_local, sizeof(__nv_bfloat16) * kGatherMaxElements));
    CUDA_CHECK(cudaMalloc(&g_head_act, sizeof(std::uint16_t) * 5120 * 32));
    g_mailbox_dev = static_cast<MailboxShared*>(dev);
}

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
    std::fprintf(stderr, "[ninfer] TP2 rank %d: NCCL communicator up\n", rank);

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

    const char* mailbox_env = std::getenv("NINFER_TP_MAILBOX");
    if (mailbox_env == nullptr || mailbox_env[0] != '0') {
        // Must live on tmpfs: cudaHostRegister rejects file-backed mappings of a regular filesystem.
        const char* mailbox_path = std::getenv("NINFER_TP_MAILBOX_FILE");
        init_mailbox(rank, mailbox_path != nullptr ? mailbox_path : "/dev/shm/ninfer_tp2_mailbox",
                     id_nonce(id));
        // Self-check: rank r contributes r + 1 everywhere, the sum must be exactly 3 on both ranks.
        const int n = kMailboxMaxElements;
        std::vector<__nv_bfloat16> host(n, __float2bfloat16(static_cast<float>(rank + 1)));
        auto* x = static_cast<__nv_bfloat16*>(buffer);
        for (int round = 0; round < 3; ++round) {
            CUDA_CHECK(cudaMemcpyAsync(x, host.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
            launch_mailbox_allreduce(x, n, rank, g_mailbox_dev, g_mailbox_steps, stream);
            std::vector<__nv_bfloat16> out(n);
            CUDA_CHECK(cudaMemcpyAsync(out.data(), x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < n; ++i) {
                if (__bfloat162float(out[i]) != 3.0f) {
                    throw std::runtime_error("TP2: mailbox all-reduce self-check failed");
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(buffer));
    g_comm = comm;
    g_rank = rank;
    if (const char* stats = std::getenv("NINFER_TP_STATS"); stats != nullptr && stats[0] == '1') {
        g_mailbox_stats = g_mailbox_dev != nullptr;
    }
    std::fprintf(stderr, "[ninfer] TP2 rank %d/2 ready (NCCL %d, mailbox %s)\n", rank, NCCL_VERSION_CODE,
                 g_mailbox_dev != nullptr ? "on" : "off");
    // Create/attach the per-unit lockstep block now so both ranks agree on its lifetime.
    (void)::ninfer::runtime::tp::Lockstep::instance();
}

int rank() { return g_rank; }

namespace {
// NINFER_TP_NCCL_STATS=1: GPU time of every NCCL all-reduce (the large ones, i.e. prefill), measured with CUDA events on
// the stream and collected lazily; every 512 calls the running totals go to stderr. Off by default.
struct NcclTimer {
    bool enabled = [] {
        const char* env = std::getenv("NINFER_TP_NCCL_STATS");
        return env != nullptr && env[0] == '1';
    }();
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> pending;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> spare;
    double total_ms = 0.0;
    std::uint64_t calls = 0, bytes = 0;

    std::pair<cudaEvent_t, cudaEvent_t> take() {
        if (spare.empty()) {
            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));
            return {a, b};
        }
        auto e = spare.back();
        spare.pop_back();
        return e;
    }
    // Starts timing an NCCL exchange on `stream` (not inside a graph capture); returns false when off.
    bool begin(cudaStream_t stream, std::pair<cudaEvent_t, cudaEvent_t>& timing) {
        if (!enabled) { return false; }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) { return false; }  // no event queries inside a graph capture
        timing = take();
        CUDA_CHECK(cudaEventRecord(timing.first, stream));
        return true;
    }
    void end(cudaStream_t stream, std::pair<cudaEvent_t, cudaEvent_t> timing, std::uint64_t wire_bytes, const char* kind) {
        CUDA_CHECK(cudaEventRecord(timing.second, stream));
        pending.push_back(timing);
        calls += 1;
        bytes += wire_bytes;
        collect();
        if (calls % 512 == 0) {
            std::fprintf(stderr, "[ninfer] TP2 NCCL rank %d: %llu %s, %.2f GB, %.1f ms GPU (%.2f ms each)\n", g_rank,
                         static_cast<unsigned long long>(calls), kind, static_cast<double>(bytes) / 1e9, total_ms,
                         total_ms / static_cast<double>(calls - pending.size()));
        }
    }
    void collect() {
        std::size_t keep = 0;
        for (auto& e : pending) {
            if (cudaEventQuery(e.second) == cudaSuccess) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e.first, e.second));
                total_ms += ms;
                spare.push_back(e);
            } else {
                pending[keep++] = e;
            }
        }
        pending.resize(keep);
    }
};
NcclTimer g_nccl_timer;
} // namespace

void allreduce(Tensor& residual, cudaStream_t stream, int slot) {
    if (g_comm == nullptr) { throw std::logic_error("TP2 collectives used before init()"); }
    if (residual.dtype != DType::BF16 || !residual.is_contiguous()) {
        throw std::invalid_argument("TP2 all-reduce requires a contiguous BF16 tensor");
    }
    if (g_mailbox_dev != nullptr && residual.numel() <= kMailboxMaxElements) {
        launch_mailbox_allreduce(static_cast<__nv_bfloat16*>(residual.data),
                                 static_cast<int>(residual.numel()), g_rank, g_mailbox_dev,
                                 g_mailbox_steps, stream, g_mailbox_stats, slot);
        return;
    }
    std::pair<cudaEvent_t, cudaEvent_t> timing{};
    const bool timed = g_nccl_timer.begin(stream, timing);
    nccl_check(ncclAllReduce(residual.data, residual.data, static_cast<std::size_t>(residual.numel()),
                             ncclBfloat16, ncclSum, g_comm, stream),
               "ncclAllReduce");
    if (timed) { g_nccl_timer.end(stream, timing, static_cast<std::uint64_t>(residual.numel()) * 2U, "all-reduce"); }
}

namespace {
// 8-bit wire buffer: both ranks' slots back to back (all-gather in place). Grown outside graph capture only.
void* g_wire8_buffer         = nullptr;
std::size_t g_wire8_capacity = 0;

int wire8_level() {
    static const int level = [] {
        const char* v = std::getenv("NINFER_TP_WIRE8");
        return v == nullptr ? 0 : std::atoi(v);
    }();
    return level;
}

// residual <- residual + p0 + p1 over the 8-bit wire; false when this call must take the bf16 path.
bool wire8_combine(const Tensor& partial, Tensor& residual, cudaStream_t stream, int slot) {
    const std::int64_t n = residual.numel();
    if (!wire8_wanted(n, slot) || partial.dtype != DType::BF16 || !partial.is_contiguous() || partial.numel() != n ||
        residual.dtype != DType::BF16 || !residual.is_contiguous()) {
        return false;
    }
    const std::size_t bytes = wire8_slot_bytes(n);
    if (2 * bytes > g_wire8_capacity) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) { return false; }  // both ranks see the same shapes: same choice
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (g_wire8_buffer != nullptr) { CUDA_CHECK(cudaFree(g_wire8_buffer)); }
        CUDA_CHECK(cudaMalloc(&g_wire8_buffer, 2 * bytes));
        g_wire8_capacity = 2 * bytes;
        std::fprintf(stderr, "[ninfer] TP2 rank %d: 8-bit wire buffer %.1f MB\n", g_rank, static_cast<double>(2 * bytes) / 1e6);
    }
    auto* base = static_cast<std::byte*>(g_wire8_buffer);
    launch_wire8_quantize(static_cast<const __nv_bfloat16*>(partial.data), n, base + g_rank * bytes, stream);
    std::pair<cudaEvent_t, cudaEvent_t> timing{};
    const bool timed = g_nccl_timer.begin(stream, timing);
    nccl_check(ncclAllGather(base + g_rank * bytes, base, bytes, ncclInt8, g_comm, stream), "ncclAllGather");
    if (timed) { g_nccl_timer.end(stream, timing, bytes * 2U, "8-bit all-gather"); }
    launch_wire8_combine(static_cast<__nv_bfloat16*>(residual.data), n, base, base + bytes, stream);
    return true;
}
} // namespace

bool wire8_wanted(std::int64_t elements, int slot) {
    const int level = wire8_level();
    return (level == 1 || (level == 2 && slot == 2)) && g_comm != nullptr && elements > kMailboxMaxElements &&
           elements % kWire8Block == 0;
}

bool head_linear(const Tensor& hidden, const Weight& head, Tensor& out, cudaStream_t stream) {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_TP_SHARD_HEADS");
        return v == nullptr || v[0] != '0';
    }();
    if (!enabled || g_mailbox_dev == nullptr || head.n % 64 != 0) { return false; }
    const std::int32_t nh = head.n / 2;
    const std::int32_t t  = static_cast<std::int32_t>(hidden.ne[1]);
    if (static_cast<std::int64_t>(nh) * t > kGatherMaxElements || (static_cast<std::int64_t>(nh) * t) % 8 != 0 ||
        out.ne[0] != head.n || out.ne[1] != t || out.dtype != DType::BF16 || !out.is_contiguous()) {
        return false;
    }
    Weight half;
    if (!head_row_view(head, g_rank * nh, nh, half)) { return false; }
    Tensor local(g_head_local, DType::BF16, {nh, t});
    if (head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
#ifndef NINFER_VOLTA_BUILD
        // Non-Volta rank (Ada): the FP8 row-slice launcher below is the Volta QPN kernel. Until the
        // Ada rank has its own, fall back to the unsharded head (the caller handles false).
        return false;
#else
        // The public FP8 entry validates the whole-tensor plane geometry, which a row slice cannot
        // satisfy; call the QPN launcher directly (same kernel the vocabulary route uses at T<=32).
        if (head.layout != QuantLayout::VoltaQpnPrepacked || t > ops::detail::kFp8VoltaQpnMaxTokens ||
            hidden.ne[0] != 5120 || !ops::detail::fp8_volta_qpn_supported(nh, head.k, t)) {
            return false;
        }
        ops::detail::fp8_stage_bf16_activation_sm70(hidden, g_head_act, stream);
        ops::detail::launch_fp8_volta_qpn_fp16(hidden, half, g_head_act, local, stream);
#endif
    } else {
        ops::linear(hidden, half, local, stream);
    }
    launch_mailbox_gather_rows(g_head_local, nh, t, static_cast<__nv_bfloat16*>(out.data), g_rank,
                               g_mailbox_dev, g_gather_steps, stream);
    return true;
}

void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream, int slot) {
    if (wire8_combine(partial, residual, stream, slot)) { return; }
    if (g_rank == 0) {
        ops::residual_add(partial, residual, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(residual.data, partial.data, residual.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    allreduce(residual, stream, slot);
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
