#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;
using ninfer::test::artifact_fixture::write_v3_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_BF16S"},
                         {"layout", "row-scale-v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct   = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows     = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout fp8_rows = StorageLayout::RowScaleV1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};
    const std::array<std::uint64_t, 2> fp8_shape = {2, 4};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape) != 260) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
    expect_artifact_error([&] { tensor_encoded_size(fp8_rows, NumericFormat::NVFP4, fp8_shape); },
                          "row-scale format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape_2); },
        "row-scale rank mismatch");
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 9 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 9> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8", "fp8_row",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* fp8      = std::get_if<TensorDescriptor>(reader.find("fp8_row"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1 ||
        fp8 == nullptr || fp8->shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        fp8->layout != StorageLayout::RowScaleV1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        constexpr std::array<std::uint8_t, 8> invalid_magic = {
            'I', 'N', 'V', 'A', 'L', 'I', 'D', '!',
        };
        auto fixture = write_fixture(normative_directory(), "invalid_magic", invalid_magic);
        expect_artifact_error([&] { Reader reader(fixture.path); }, "invalid magic");
    }
}

// tests/fixtures/artifact/v3_directory.json is shared with tests/artifact/test_v3.py: both readers must map the same
// v3 directory onto the same stored-object names and identity.
Json v3_fixture() {
    std::ifstream input(NINFER_SOURCE_DIR "/tests/fixtures/artifact/v3_directory.json");
    if (!input) { throw std::runtime_error("cannot open tests/fixtures/artifact/v3_directory.json"); }
    return Json::parse(input);
}

void test_v3_fixture() {
    const Json fixture  = v3_fixture();
    const Json& expected = fixture.at("expected");
    auto file           = write_v3_fixture(fixture.at("directory"), "valid");
    Reader reader(file.path);
    if (reader.identity().model_id != expected.at("identity").at("model_id").get<std::string>() ||
        reader.identity().weights_id != expected.at("identity").at("weights_id").get<std::string>()) {
        throw std::runtime_error("v3 identity mismatch: " + reader.identity().model_id + " / " +
                                 reader.identity().weights_id);
    }
    // Each object's payload carries its directory-order marker; duplicated activation scalars resolve to the first.
    const Json& objects = fixture.at("directory").at("objects");
    std::vector<std::string> distinct;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        const auto id   = objects[i].at("id").get<std::string>();
        const auto name = expected.at("names").at(id).get<std::string>();
        const auto* object = reader.find(name);
        if (object == nullptr) { throw std::runtime_error("v3 object " + id + " not found as " + name); }
        if (std::find(distinct.begin(), distinct.end(), name) != distinct.end()) { continue; }
        distinct.push_back(name);
        const auto payload = reader.payload(*object);
        if (payload.data.size() != objects[i].at("bytes").get<std::uint64_t>() ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("v3 payload span mismatch for " + name);
        }
    }
    if (reader.objects().size() != distinct.size() || reader.payload_offset() != 4096) {
        throw std::runtime_error("v3 object count or payload offset mismatch");
    }
    const auto* gate_up = std::get_if<TensorDescriptor>(reader.find("text/layers/3/mlp/gate_up"));
    const auto* output  = std::get_if<TensorDescriptor>(reader.find("text/layers/1/attention/output"));
    if (gate_up == nullptr || gate_up->format != NumericFormat::W8G32_F16S ||
        gate_up->layout != StorageLayout::RowSplitK128V1 || gate_up->shape != std::vector<std::uint64_t>({4, 32}) ||
        output == nullptr || output->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        output->layout != StorageLayout::RowScaleV1 ||
        std::get_if<ResourceDescriptor>(reader.find("frontend/tokenizer.json")) == nullptr) {
        throw std::runtime_error("v3 object signature mismatch");
    }
}

void test_v3_validation() {
    const Json directory = v3_fixture().at("directory");
    {
        auto broken = directory;
        broken["bindings"].erase("text/embedding");
        auto file = write_v3_fixture(broken, "unbound");
        expect_artifact_error([&] { Reader reader(file.path); }, "v3 object without binding");
    }
    {
        auto broken                   = directory;
        broken["objects"][3]["bytes"] = 543;
        auto file                     = write_v3_fixture(broken, "wrong_size");
        expect_artifact_error([&] { Reader reader(file.path); }, "v3 tensor with wrong encoded size");
    }
    {
        auto broken                    = directory;
        broken["objects"][1]["offset"] = 257;
        auto file                      = write_v3_fixture(broken, "misaligned");
        expect_artifact_error([&] { Reader reader(file.path); }, "misaligned v3 object");
    }
    {
        auto broken = directory;
        broken["files"].push_back(broken["files"][0]);
        auto file = write_v3_fixture(broken, "continuation");
        expect_artifact_error([&] { Reader reader(file.path); }, "v3 continuation files");
    }
    {
        auto broken                    = directory;
        broken["objects"][2]["format"] = "fp16";
        auto file                      = write_v3_fixture(broken, "unknown_format");
        expect_artifact_error([&] { Reader reader(file.path); }, "unknown v3 format");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_v3_fixture();
        test_v3_validation();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
