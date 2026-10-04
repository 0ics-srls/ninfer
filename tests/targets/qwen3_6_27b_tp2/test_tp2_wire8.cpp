// TP2 8-bit wire (NINFER_TP_WIRE8): the prefill row-parallel partials travel as int8 codes with one fp32 scale per
// 64 values, and each rank computes residual + dequant(p0) + dequant(p1) itself. The contract this suite protects:
//   1. quantization: scale = max|x| / 127, code = round-to-nearest-even(x * 127 / max|x|), clamped to [-127, 127];
//   2. combine: ((residual + q0 * s0) + q1 * s1), one IEEE rounding per operation, then bf16 round-to-nearest-even;
//   3. the slot layout (codes, then scales, each 256-byte aligned).
// Both rank builds (sm_70 and sm_89 on the Volta path) run this same suite against the same host oracle, so passing
// on both proves the residual stream is bit-identical on the two cards: any divergence would break the lockstep.
#include "targets/qwen3_6_27b_tp2/impl/tp2_wire8.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer::test;
using ninfer::DeviceBuffer;
namespace tp2 = ninfer::targets::qwen3_6_27b_tp2::detail::tp2;

namespace {

constexpr int kBlock  = tp2::kWire8Block;
constexpr int kHidden = 5120; // Qwen3.8-27B hidden size: the reductions are [tokens, 5120]

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct Quantized {
    std::vector<std::int8_t> codes;
    std::vector<float> scales;
};

// Host oracle of the quantizer, written from the contract (float ops, round-to-nearest-even).
Quantized quantize_oracle(const std::vector<float>& x) {
    Quantized q{std::vector<std::int8_t>(x.size()), std::vector<float>(x.size() / kBlock)};
    for (std::size_t block = 0; block < q.scales.size(); ++block) {
        float m = 0.0f;
        for (int i = 0; i < kBlock; ++i) { m = std::fmax(m, std::fabs(x[block * kBlock + i])); }
        const float inv = m > 0.0f ? 127.0f / m : 0.0f;
        for (int i = 0; i < kBlock; ++i) {
            const float scaled = x[block * kBlock + i] * inv;
            const long code    = std::lrint(scaled); // default rounding mode: nearest-even
            q.codes[block * kBlock + i] = static_cast<std::int8_t>(code < -127 ? -127 : code > 127 ? 127 : code);
        }
        q.scales[block] = m / 127.0f;
    }
    return q;
}

std::vector<std::uint16_t> combine_oracle(const std::vector<float>& residual, const Quantized& p0,
                                          const Quantized& p1) {
    std::vector<std::uint16_t> out(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        const std::size_t block = i / kBlock;
        const float a = static_cast<float>(p0.codes[i]) * p0.scales[block];
        const float b = static_cast<float>(p1.codes[i]) * p1.scales[block];
        const float s = residual[i] + a;
        out[i]        = f32_to_bf16(s + b);
    }
    return out;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// Partials with the edge cases a real all-reduce sees: an all-zero block, a block whose maximum is reached exactly
// (code +-127), a block of one sign, and large magnitudes.
std::vector<float> make_partial(std::int64_t elements, std::uint32_t seed, float range) {
    std::vector<float> x(static_cast<std::size_t>(elements));
    fill_uniform(x, seed, -range, range);
    for (int i = 0; i < kBlock; ++i) { x[static_cast<std::size_t>(i)] = 0.0f; }
    if (elements >= 4 * kBlock) {
        for (int i = 0; i < kBlock; ++i) { x[kBlock + i] = std::fabs(x[kBlock + i]); }
        x[2 * kBlock + 5]  = range;
        x[2 * kBlock + 6]  = -range;
        x[3 * kBlock + 63] = 3.0e4f * range;
    }
    round_to_bf16(x);
    return x;
}

int run_case(const std::string& label, std::int64_t tokens, std::uint32_t seed) {
    const std::int64_t elements = tokens * kHidden;
    const std::size_t slot      = tp2::wire8_slot_bytes(elements);
    int failures = verify_exact((label + " slot layout").c_str(),
                                std::vector<std::size_t>{slot},
                                std::vector<std::size_t>{align256(static_cast<std::size_t>(elements)) +
                                                         align256(static_cast<std::size_t>(elements / kBlock) * 4)});

    const std::vector<float> p0       = make_partial(elements, seed, 4.0f);
    const std::vector<float> p1       = make_partial(elements, seed + 7, 0.25f);
    std::vector<float> residual(static_cast<std::size_t>(elements));
    fill_uniform(residual, seed + 13, -16.0f, 16.0f);
    round_to_bf16(residual);

    const Quantized q0 = quantize_oracle(p0);
    const Quantized q1 = quantize_oracle(p1);
    const auto expected = combine_oracle(residual, q0, q1);

    DeviceBuffer d_p0 = to_device(to_bf16_bits(p0));
    DeviceBuffer d_p1 = to_device(to_bf16_bits(p1));
    GuardedDeviceBuffer slot0(slot), slot1(slot);
    tp2::launch_wire8_quantize(static_cast<const __nv_bfloat16*>(d_p0.p), elements, slot0.data(), nullptr);
    tp2::launch_wire8_quantize(static_cast<const __nv_bfloat16*>(d_p1.p), elements, slot1.data(), nullptr);
    cuda_synchronize();

    const std::size_t codes_bytes = align256(static_cast<std::size_t>(elements));
    const auto got_codes  = from_device<std::int8_t>(slot0.data(), static_cast<std::size_t>(elements));
    const auto got_scales = from_device<float>(static_cast<const std::uint8_t*>(slot0.data()) + codes_bytes,
                                               static_cast<std::size_t>(elements / kBlock));
    failures += verify_exact((label + " codes").c_str(), got_codes, q0.codes);
    failures += verify_exact((label + " scales").c_str(), got_scales, q0.scales);

    // Each rank combines into its own copy of the residual: both must equal the oracle bit for bit.
    for (int rank = 0; rank < 2; ++rank) {
        GuardedDeviceBuffer d_residual(static_cast<std::size_t>(elements) * sizeof(std::uint16_t));
        const auto residual_bits = to_bf16_bits(residual);
        d_residual.copy_from_host(residual_bits.data(), d_residual.bytes());
        tp2::launch_wire8_combine(static_cast<__nv_bfloat16*>(d_residual.data()), elements, slot0.data(),
                                  slot1.data(), nullptr);
        cuda_synchronize();
        failures += verify_exact((label + " combine rank " + std::to_string(rank)).c_str(),
                                 from_device<std::uint16_t>(d_residual.data(), expected.size()), expected);
        failures += d_residual.verify_guards((label + " residual guards").c_str());
    }
    failures += slot0.verify_guards((label + " slot0 guards").c_str());
    failures += slot1.verify_guards((label + " slot1 guards").c_str());

    // Accuracy against the exact sum: at most half a quantization step per rank (q), then the final bf16 rounding,
    // half an ulp of the quantized sum: <= (|exact| + q) * 2^-8. Bound: q * (1 + 2^-8) + |exact| * 2^-8.
    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const std::size_t block = i / kBlock;
        const double exact      = double(residual[i]) + double(p0[i]) + double(p1[i]);
        const double got        = bf16_to_f32(expected[i]);
        const double quantized  = 0.5 * (double(q0.scales[block]) + double(q1.scales[block]));
        const double limit      = quantized * (1.0 + 0x1p-8) + std::fabs(exact) * 0x1p-8 + 1e-30;
        worst                   = std::fmax(worst, std::fabs(got - exact) / limit);
    }
    if (worst > 1.0) {
        std::cout << "FAIL " << label << " accuracy: error/limit " << worst << "\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("wire8 [1,5120]", 1, 11u);
    failures += run_case("wire8 [17,5120]", 17, 23u);
    failures += run_case("wire8 [2048,5120]", 2048, 37u);
    std::cout << (failures ? "FAIL" : "OK") << " tp2 wire8\n";
    return failures ? 1 : 0;
}
