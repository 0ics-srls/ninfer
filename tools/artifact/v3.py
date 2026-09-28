"""NInfer v3 containers for the Python tools: read into the object names the tools use, write derived v3 artifacts.

A v3 artifact stores the same payload encodings as v2 behind a different directory: opaque object ids, logical
bindings (whole objects or ranges of fused objects), activation-scalar uses and component resources. The reader
rebuilds the stored-object names (`text/layers/0/mlp/gate_up`, ...) and the identity the runtime binders use, the
same mapping as src/artifact/reader.cpp. The writer emits a v3 file whose directory is derived from a source v3
directory (e.g. one tensor-parallel rank), keeping ids, bindings and uses.
"""

from __future__ import annotations

import copy
import json
import re
import struct
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .layouts import align_up, encoded_size, get_layout

MAGIC_V3 = b"NINFER\x00\x03"
HEADER_V3 = struct.Struct("<8sQ16s")
HEADER_V3_BYTES = HEADER_V3.size
PAYLOAD_ALIGNMENT = 4096

FORMATS = {"bf16": "BF16", "fp32": "FP32", "int32": "I32", "q4_g64_fp16": "Q4G64_F16S",
           "q5_g64_fp16": "Q5G64_F16S", "q6_g64_fp16": "Q6G64_F16S", "q8_g32_fp16": "W8G32_F16S",
           "nvfp4": "NVFP4", "fp8_e4m3fn_row_bf16": "FP8_E4M3FN_ROW_BF16S"}
LAYOUTS = {"contiguous_le_v1": "contiguous-le-v1", "row_split_k128_v1": "row-split-k128-v1",
           "block_scale_k16_m128x4_v1": "blockscale-k16-m128x4-v1", "row_scale_v1": "row-scale-v1"}
ENCODINGS = {"raw_bytes_v1": "raw-bytes-v1"}
_FUSED = {("a_projection", "b_projection"): "a_b_projection",
          ("router", "shared_score"): "router_shared_gate"}
_VISION_FUSED = {("query", "key", "value"): "qkv", ("query_bias", "key_bias", "value_bias"): "qkv_bias"}


def single_name(logical: str) -> str:
    if logical == "proposal/head":
        return "text/draft_head"
    if logical == "proposal/token_ids":
        return "text/draft_head_token_ids"
    if logical.startswith("mtp/layers/0/"):
        logical = "mtp/layer/" + logical[len("mtp/layers/0/"):]
    m = re.fullmatch(r"(vision/layers/\d+/)(norm[12])_(weight|bias)", logical)
    if m:
        return f"{m.group(1)}{m.group(2)}/{m.group(3)}"
    m = re.fullmatch(r"vision/merger/norm_(weight|bias)", logical)
    if m:
        return f"vision/merger/norm/{m.group(1)}"
    return logical


def fused_name(logical: list[str]) -> str:
    parent = logical[0].rsplit("/", 1)[0]
    roles = tuple(name.rsplit("/", 1)[1] for name in logical)
    base = single_name(parent + "/x").rsplit("/", 1)[0]
    if "/moe" in parent:
        raise NotImplementedError("v3 MoE artifacts are not supported by this tree yet")
    if "context_key" in roles or "context_value" in roles:          # DFlash views over the same q/k/v bytes
        return base + "/query_key_value"
    if parent.startswith("vision/") and roles in _VISION_FUSED:
        return base + "/" + _VISION_FUSED[roles]
    if parent.startswith("dflash") and roles == ("query", "key", "value"):
        return base + "/query_key_value"
    return base + "/" + _FUSED.get(roles, "_".join(roles))


def _scalar_operation(group: str, role: str) -> str:
    if role == "up" or (role == "gate" and group.endswith("/mlp")):
        return "gate_up_projection"
    if role in ("query", "key", "gate", "value", "z"):
        return "input_projection"
    if role == "output":
        return "output_projection"
    if role == "down":
        return "down_projection"
    raise ValueError(f"v3 activation scalar on unknown role: {role}")


def object_names(directory: dict) -> dict[str, str]:
    """v3 object id -> stored-object name used by the binders."""
    names: dict[str, str] = {}
    for use in directory.get("uses", []):
        aux = (use.get("auxiliaries") or {}).get("activation_input_divisor")
        if aux:
            group, role = use["parameter"].rsplit("/", 1)
            base = single_name(group + "/x").rsplit("/", 1)[0]
            names.setdefault(aux["object"], f"{base}/{_scalar_operation(group, role)}/input_scale_divisor")
    for component in directory["components"].values():
        for role, oid in (component.get("resources") or {}).items():
            names.setdefault(oid, "frontend/" + role)
    refs: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
    for logical, binding in directory["bindings"].items():
        if "object" in binding:
            refs[binding["object"]].append((0, logical, True))
        else:
            for part in binding["parts"]:
                refs[part["object"]].append((part["range"][0], logical, False))
    for oid, found in refs.items():
        if oid in names:
            continue
        found.sort()
        if len(found) == 1 and found[0][2]:
            names[oid] = single_name(found[0][1])
            continue
        members: list[str] = []
        for _, name, _ in found:
            if name not in members:
                members.append(name)
        names[oid] = fused_name(members)
    return names


def identity_of(directory: dict) -> tuple[str, str]:
    recipe = (directory.get("provenance") or {}).get("recipe", "")
    nvfp4 = recipe.endswith("_nvfp4") if recipe else any(o.get("format") == "nvfp4" for o in directory["objects"])
    weights_id = "nvfp4" if nvfp4 else "groupwise-int"
    ranks = ((directory.get("metadata") or {}).get("tensor_parallel") or {}).get("ranks")
    if ranks:
        weights_id += f"-tp{ranks}"
    return directory["metadata"]["name"], weights_id


def read_directory(file) -> tuple[dict, int, bytes]:
    """Parse the header at the current start of `file`; returns (directory, payload_offset, artifact_id)."""
    file.seek(0)
    magic, json_bytes, artifact_id = HEADER_V3.unpack(file.read(HEADER_V3_BYTES))
    if magic != MAGIC_V3:
        raise ValueError("artifact magic is not NInfer v3")
    directory = json.loads(file.read(json_bytes).decode("utf-8"))
    if len(directory["files"]) != 1:
        raise ValueError("v3 artifacts with continuation files are not supported yet")
    return directory, align_up(HEADER_V3_BYTES + json_bytes, PAYLOAD_ALIGNMENT), artifact_id


def stored_view(raw: dict, name: str):
    """(kind, name, shape, format, layout | encoding) of one v3 object in the tools' vocabulary."""
    if raw["kind"] == "tensor":
        return ("tensor", name, tuple(raw["shape"]), FORMATS[raw["format"]], LAYOUTS[raw["layout"]])
    return ("resource", name, None, None, ENCODINGS[raw["encoding"]])


def derive_directory(source: dict, objects: list[dict], *, dropped: set[str], row_scale: dict[str, tuple[int, int]],
                     metadata: dict | None = None, provenance: dict | None = None,
                     drop_components: Iterable[str] = ()) -> dict:
    """Directory for a derived artifact: `objects` (in storage order, offsets assigned by V3Writer) replace the
    source objects; bindings and uses that reference dropped objects go away; ranges of fused objects whose rows
    were cut are scaled by row_scale[id] = (num, den) (every part starts on a segment boundary)."""
    bindings = {}
    for logical, binding in source["bindings"].items():
        if "object" in binding:
            if binding["object"] not in dropped:
                bindings[logical] = copy.deepcopy(binding)
            continue
        parts = [p for p in binding["parts"] if p["object"] not in dropped]
        if not parts:
            continue
        if len(parts) != len(binding["parts"]):
            raise ValueError(f"binding {logical} spans kept and dropped objects")
        new_parts = []
        for part in parts:
            begin, end = part["range"]
            if part["object"] in row_scale:
                num, den = row_scale[part["object"]]
                if (begin * num) % den or (end * num) % den:
                    raise ValueError(f"binding {logical} range {part['range']} does not scale by {num}/{den}")
                begin, end = begin * num // den, end * num // den
            new_parts.append({**part, "range": [begin, end]})
        bindings[logical] = {**binding, "parts": new_parts}
    uses = []
    for use in source.get("uses", []):
        aux = (use.get("auxiliaries") or {}).values()
        if use["parameter"] in bindings and not any(a.get("object") in dropped for a in aux):
            uses.append(copy.deepcopy(use))
    components = {k: copy.deepcopy(v) for k, v in source["components"].items() if k not in set(drop_components)}
    out = {key: copy.deepcopy(value) for key, value in source.items()}
    out.update(components=components, objects=objects, bindings=bindings, uses=uses)
    out["metadata"] = {**source.get("metadata", {}), **(metadata or {})}
    out["provenance"] = {**source.get("provenance", {}), **(provenance or {})}
    return out


class V3Writer:
    """Streaming writer: plans aligned offsets for `directory['objects']`, then takes payloads in that order."""

    def __init__(self, path: str | Path, directory: dict):
        self.path = Path(path)
        cursor = 0
        for raw in directory["objects"]:
            if raw["kind"] == "tensor":
                layout = get_layout(LAYOUTS[raw["layout"]])
                raw["bytes"] = encoded_size(layout, FORMATS[raw["format"]], tuple(raw["shape"]))
                alignment = layout.alignment
            else:
                ENCODINGS[raw["encoding"]]
                alignment = 1
            raw["offset"] = align_up(cursor, alignment)
            cursor = raw["offset"] + raw["bytes"]
        directory["files"] = [{"path": None, "payload_bytes": cursor}]
        self.objects = directory["objects"]
        data = json.dumps(directory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.payload_offset = align_up(HEADER_V3_BYTES + len(data), PAYLOAD_ALIGNMENT)
        self._file = self.path.open("wb")
        self._file.write(HEADER_V3.pack(MAGIC_V3, len(data), uuid.uuid4().bytes))
        self._file.write(data)
        self._file.write(b"\x00" * (self.payload_offset - HEADER_V3_BYTES - len(data)))
        self._next = 0
        self._cursor = 0

    def write(self, oid: str, payload) -> None:
        raw = self.objects[self._next]
        if oid != raw["id"]:
            raise ValueError(f"expected payload {raw['id']}, got {oid}")
        if raw["offset"] > self._cursor:
            self._file.write(b"\x00" * (raw["offset"] - self._cursor))
        view = memoryview(payload).cast("B")
        if len(view) != raw["bytes"]:
            raise ValueError(f"payload {oid} has {len(view)} bytes; expected {raw['bytes']}")
        self._file.write(view)
        self._cursor = raw["offset"] + raw["bytes"]
        self._next += 1

    def finish(self) -> None:
        if self._next != len(self.objects):
            raise ValueError(f"artifact is missing payload {self.objects[self._next]['id']}")
        self._file.truncate(self.payload_offset + self._cursor)
        self._file.close()
