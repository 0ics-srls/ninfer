#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

// NInfer v3 container (upstream since 2026-09-15): 32-byte header (magic, JSON length, 16-byte artifact id), a JSON
// directory with opaque object ids, logical bindings, uses and components, payload at the next 4096 boundary.
// The stored bytes of every object are the same encodings v2 used, so the reader maps the v3 directory back onto the
// object names the model binders already consume (the inverse of upstream tools/upgrade_ninfer_v2_to_v3.py).
constexpr std::array<std::byte, 8> kMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kV3HeaderBytes = 32;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}


// ---- NInfer v3 directory -------------------------------------------------------------------------------------

NumericFormat parse_v3_format(std::string_view name) {
    if (name == "bf16") { return NumericFormat::BF16; }
    if (name == "fp32") { return NumericFormat::FP32; }
    if (name == "int32") { return NumericFormat::I32; }
    if (name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "fp8_e4m3fn_row_bf16") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown v3 tensor format: " + std::string(name));
}

StorageLayout parse_v3_layout(std::string_view name) {
    if (name == "contiguous_le_v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row_split_k128_v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "block_scale_k16_m128x4_v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown v3 tensor layout: " + std::string(name));
}

ResourceEncoding parse_v3_encoding(std::string_view name) {
    if (name == "raw_bytes_v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown v3 resource encoding: " + std::string(name));
}

std::string parent_path(std::string_view path) {
    const auto slash = path.rfind('/');
    if (slash == std::string_view::npos) { throw ArtifactError("v3 name without parent: " + std::string(path)); }
    return std::string(path.substr(0, slash));
}

std::string last_segment(std::string_view path) {
    const auto slash = path.rfind('/');
    return std::string(slash == std::string_view::npos ? path : path.substr(slash + 1));
}

// Logical parameter name (v3) -> stored object name (v2) for a parameter that owns a whole object.
std::string v3_single_name(std::string logical) {
    if (logical == "proposal/head") { return "text/draft_head"; }
    if (logical == "proposal/token_ids") { return "text/draft_head_token_ids"; }
    constexpr std::string_view kMtp = "mtp/layers/0/";
    if (logical.starts_with(kMtp)) { logical = "mtp/layer/" + logical.substr(kMtp.size()); }
    if (logical.starts_with("vision/layers/")) {
        const auto role = last_segment(logical);
        if (role.starts_with("norm1_") || role.starts_with("norm2_")) {
            return parent_path(logical) + "/" + role.substr(0, 5) + "/" + role.substr(6);
        }
    }
    if (logical == "vision/merger/norm_weight") { return "vision/merger/norm/weight"; }
    if (logical == "vision/merger/norm_bias") { return "vision/merger/norm/bias"; }
    return logical;
}

// Several logical parameters stored in one fused object (ranges in storage order) -> the fused v2 name.
std::string v3_fused_name(const std::vector<std::string>& logical) {
    const auto parent = parent_path(logical.front());
    std::vector<std::string> roles;
    for (const auto& name : logical) { roles.push_back(last_segment(name)); }
    const auto base = parent_path(v3_single_name(parent + "/x"));
    const auto has  = [&](std::string_view role) {
        return std::find(roles.begin(), roles.end(), role) != roles.end();
    };
    const auto is = [&](std::initializer_list<std::string_view> expected) {
        return roles.size() == expected.size() && std::equal(roles.begin(), roles.end(), expected.begin());
    };
    if (parent.find("/moe") != std::string::npos) {
        throw ArtifactError("v3 MoE artifacts are not supported by this tree yet");
    }
    if (has("context_key") || has("context_value")) { return base + "/query_key_value"; }  // DFlash views
    if (parent.starts_with("vision/")) {
        if (is({"query", "key", "value"})) { return base + "/qkv"; }
        if (is({"query_bias", "key_bias", "value_bias"})) { return base + "/qkv_bias"; }
    }
    if (parent.starts_with("dflash") && is({"query", "key", "value"})) { return base + "/query_key_value"; }
    if (is({"a_projection", "b_projection"})) { return base + "/a_b_projection"; }
    if (is({"router", "shared_score"})) { return base + "/router_shared_gate"; }
    std::string joined;
    for (const auto& role : roles) { joined += (joined.empty() ? "" : "_") + role; }
    return base + "/" + joined;
}

std::string v3_scalar_operation(std::string_view group, std::string_view role) {
    if (role == "up" || (role == "gate" && group.ends_with("/mlp"))) { return "gate_up_projection"; }
    if (role == "query" || role == "key" || role == "gate" || role == "value" || role == "z") {
        return "input_projection";
    }
    if (role == "output") { return "output_projection"; }
    if (role == "down") { return "down_projection"; }
    throw ArtifactError("v3 activation scalar on unknown role: " + std::string(role));
}

// Object id -> name the binders use, rebuilt from bindings, activation-scalar uses and component resources.
std::unordered_map<std::string, std::string> v3_object_names(const Json& directory) {
    std::unordered_map<std::string, std::string> names;
    if (const auto uses = directory.find("uses"); uses != directory.end()) {
        for (const auto& use : *uses) {
            const auto aux = use.find("auxiliaries");
            if (aux == use.end() || !aux->contains("activation_input_divisor")) { continue; }
            const auto& object    = require_string(aux->at("activation_input_divisor").at("object"), "scalar object");
            const auto& parameter = require_string(use.at("parameter"), "use parameter");
            const auto group      = parent_path(parameter);
            const auto name = parent_path(v3_single_name(group + "/x")) + "/" +
                              v3_scalar_operation(group, last_segment(parameter)) + "/input_scale_divisor";
            names.emplace(object, name);
        }
    }
    for (const auto& component : directory.at("components")) {
        const auto resources = component.find("resources");
        if (resources == component.end()) { continue; }
        for (const auto& [role, id] : resources->items()) {
            names.emplace(require_string(id, "resource id"), "frontend/" + role);
        }
    }
    struct Ref {
        std::uint64_t begin;
        std::string logical;
        bool whole;
        bool operator<(const Ref& other) const {
            return std::tie(begin, logical, whole) < std::tie(other.begin, other.logical, other.whole);
        }
    };
    std::unordered_map<std::string, std::vector<Ref>> refs;
    for (const auto& [logical, binding] : directory.at("bindings").items()) {
        if (const auto object = binding.find("object"); object != binding.end()) {
            refs[require_string(*object, "binding object")].push_back({0, logical, true});
            continue;
        }
        for (const auto& part : binding.at("parts")) {
            refs[require_string(part.at("object"), "binding part object")].push_back(
                {require_unsigned(part.at("range").at(0), "binding range", false), logical, false});
        }
    }
    for (auto& [id, list] : refs) {
        if (names.contains(id)) { continue; }
        std::sort(list.begin(), list.end());
        if (list.size() == 1 && list.front().whole) {
            names.emplace(id, v3_single_name(list.front().logical));
            continue;
        }
        std::vector<std::string> logical;
        for (const auto& ref : list) {
            if (std::find(logical.begin(), logical.end(), ref.logical) == logical.end()) {
                logical.push_back(ref.logical);
            }
        }
        names.emplace(id, v3_fused_name(logical));
    }
    return names;
}

ArtifactIdentity v3_identity(const Json& directory) {
    ArtifactIdentity identity;
    identity.model_id = require_string(directory.at("metadata").at("name"), "v3 metadata name");
    std::string recipe;
    if (const auto provenance = directory.find("provenance");
        provenance != directory.end() && provenance->contains("recipe")) {
        recipe = require_string(provenance->at("recipe"), "v3 recipe");
    }
    bool nvfp4 = recipe.ends_with("_nvfp4");
    if (recipe.empty()) {
        for (const auto& object : directory.at("objects")) {
            if (object.value("format", "") == "nvfp4") { nvfp4 = true; }
        }
    }
    identity.weights_id = nvfp4 ? "nvfp4" : "groupwise-int";
    return identity;
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        if (file.size() >= kV3HeaderBytes &&
            std::equal(kMagicV3.begin(), kMagicV3.end(), file.data())) {
            load_v3();
            return;
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not NInfer v2 or v3");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }


    void load_v3() {
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("v3 json_bytes must be positive"); }
        const auto metadata_end = checked_add(kV3HeaderBytes, json_bytes, "v3 JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "v3 payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared v3 JSON or payload start extends beyond the file");
        }
        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kV3HeaderBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid v3 JSON directory: ") + error.what());
        }
        const auto& files = directory.at("files");
        if (!files.is_array() || files.size() != 1) {
            throw ArtifactError("v3 artifacts with continuation files are not supported yet");
        }
        identity         = v3_identity(directory);
        const auto names = v3_object_names(directory);

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("v3 objects must be a nonempty array");
        }
        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw : raw_objects) {
            const auto& id     = require_string(raw.at("id"), "v3 object id");
            const auto& kind   = require_string(raw.at("kind"), "v3 object kind");
            const auto offset  = require_unsigned(raw.at("offset"), "v3 object offset", false);
            const auto bytes   = require_unsigned(raw.at("bytes"), "v3 object bytes", true);
            const auto named   = names.find(id);
            if (named == names.end()) { throw ArtifactError("v3 object " + id + " has no known binding"); }
            const auto& name = named->second;
            ObjectDescriptor object;
            std::uint64_t alignment = 0;
            if (kind == "tensor") {
                std::vector<std::uint64_t> shape;
                for (const auto& dim : raw.at("shape")) { shape.push_back(require_unsigned(dim, "v3 shape", true)); }
                const auto format = parse_v3_format(require_string(raw.at("format"), "v3 format"));
                const auto layout = parse_v3_layout(require_string(raw.at("layout"), "v3 layout"));
                const auto expected = tensor_encoded_size(layout, format, shape);
                if (bytes != expected) {
                    throw ArtifactError("v3 tensor " + id + " (" + name + ") stores " + std::to_string(bytes) +
                                        " bytes; layout requires " + std::to_string(expected));
                }
                alignment = tensor_alignment(layout);
                object    = TensorDescriptor{name, std::move(shape), format, layout, offset, bytes};
            } else if (kind == "resource") {
                const auto encoding = parse_v3_encoding(require_string(raw.at("encoding"), "v3 encoding"));
                alignment           = resource_alignment(encoding);
                object              = ResourceDescriptor{name, encoding, offset, bytes};
            } else {
                throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
            }
            if (offset < cursor) { throw ArtifactError("v3 object " + id + " overlaps or is out of order"); }
            if (offset % alignment != 0) { throw ArtifactError("v3 object " + id + " is misaligned"); }
            const auto end = checked_add(offset, bytes, "v3 object payload range");
            if (end > payload_bytes) { throw ArtifactError("v3 object " + id + " extends beyond the file"); }
            cursor = end;
            // v3 stores some activation scalars once per role (gate and up); the copies are identical and the
            // binders read one per projection: the first is served under the shared name.
            if (index.contains(name)) { continue; }
            index.emplace(name, entries.size());
            entries.push_back(std::move(object));
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
