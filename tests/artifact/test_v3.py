"""NInfer v3 in the Python tools (tools/artifact/v3.py, Artifact.open on v3): the directory fixture
tests/fixtures/artifact/v3_directory.json is shared with the C++ reader test (tests/test_ninfer_artifact_reader.cpp), so
both readers map the same v3 directory onto the same stored-object names and identity."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.artifact import v3
from tools.artifact.container import Artifact, ArtifactError, ResourceObject, TensorObject

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "artifact" / "v3_directory.json"


def _fixture() -> tuple[dict, dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["directory"], data["expected"]


def _planned(directory: dict) -> dict:
    """The directory without the writer-assigned fields, as a tool hands it to V3Writer."""
    out = copy.deepcopy(directory)
    out.pop("files", None)
    for raw in out["objects"]:
        raw.pop("offset", None)
        if raw["kind"] == "tensor":
            raw.pop("bytes", None)
    return out


def _write(path: Path, directory: dict) -> list[bytes]:
    """Write `directory` with V3Writer; object i gets payload bytes of value i + 1 (the C++ fixture's markers)."""
    writer = v3.V3Writer(path, directory)
    payloads = []
    for i, raw in enumerate(writer.objects):
        payloads.append(bytes([i + 1]) * raw["bytes"])
        writer.write(raw["id"], payloads[-1])
    writer.finish()
    return payloads


def test_writer_plans_the_fixture_offsets(tmp_path):
    directory, _ = _fixture()
    planned = _planned(directory)
    _write(tmp_path / "a.ninfer", planned)
    assert [(o["offset"], o["bytes"]) for o in planned["objects"]] == \
           [(o["offset"], o["bytes"]) for o in directory["objects"]]
    assert planned["files"] == directory["files"]


def test_reader_maps_names_identity_and_payloads(tmp_path):
    directory, expected = _fixture()
    path = tmp_path / "a.ninfer"
    payloads = _write(path, _planned(directory))
    art = Artifact.open(path)
    assert (art.identity.model_id, art.identity.weights_id) == \
           (expected["identity"]["model_id"], expected["identity"]["weights_id"])
    assert v3.object_names(art.v3) == expected["names"]
    distinct = list(dict.fromkeys(expected["names"].values()))
    assert [o.name for o in art.objects] == distinct           # duplicated activation scalars: first wins
    for i, raw in enumerate(directory["objects"]):
        obj = art.find(expected["names"][raw["id"]])
        if obj.name in [expected["names"][r["id"]] for r in directory["objects"][:i]]:
            continue
        assert bytes(art.payload(obj)) == payloads[i], obj.name
    gate_up = art.find("text/layers/3/mlp/gate_up")
    assert isinstance(gate_up, TensorObject) and gate_up.format == "W8G32_F16S" and gate_up.shape == (4, 32)
    assert isinstance(art.find("frontend/tokenizer.json"), ResourceObject)
    assert len(art.v3_entries) == len(directory["objects"])


def test_identity_without_recipe_or_tensor_parallel(tmp_path):
    directory, _ = _fixture()
    planned = _planned(directory)
    planned["provenance"] = {}
    planned["metadata"] = {"name": "plain"}
    _write(tmp_path / "a.ninfer", planned)
    assert v3.identity_of(Artifact.open(tmp_path / "a.ninfer").v3) == ("plain", "groupwise-int")


def test_unbound_object_is_rejected(tmp_path):
    directory, _ = _fixture()
    planned = _planned(directory)
    del planned["bindings"]["text/embedding"]
    _write(tmp_path / "a.ninfer", planned)
    with pytest.raises(ArtifactError):
        Artifact.open(tmp_path / "a.ninfer")


def test_continuation_files_are_rejected(tmp_path):
    directory, _ = _fixture()
    path = tmp_path / "a.ninfer"
    _write(path, _planned(directory))
    raw = path.read_bytes()
    _, json_bytes, artifact_id = v3.HEADER_V3.unpack(raw[:v3.HEADER_V3_BYTES])
    parsed = json.loads(raw[v3.HEADER_V3_BYTES:v3.HEADER_V3_BYTES + json_bytes])
    parsed["files"].append(parsed["files"][0])
    data = json.dumps(parsed, separators=(",", ":")).encode()
    assert len(data) < v3.PAYLOAD_ALIGNMENT - v3.HEADER_V3_BYTES
    path.write_bytes(v3.HEADER_V3.pack(v3.MAGIC_V3, len(data), artifact_id) + data)
    with path.open("rb") as file, pytest.raises(ValueError):
        v3.read_directory(file)


def test_writer_enforces_order_and_sizes(tmp_path):
    directory, _ = _fixture()
    writer = v3.V3Writer(tmp_path / "a.ninfer", _planned(directory))
    first, second = writer.objects[0], writer.objects[1]
    with pytest.raises(ValueError):
        writer.write(second["id"], bytes(second["bytes"]))           # out of order
    with pytest.raises(ValueError):
        writer.write(first["id"], bytes(first["bytes"] + 1))         # wrong size
    writer.write(first["id"], bytes(first["bytes"]))
    with pytest.raises(ValueError):
        writer.finish()                                              # payloads missing


def test_derive_directory_scales_fused_ranges_and_drops():
    directory, _ = _fixture()
    objects = [dict(o) for o in directory["objects"] if o["id"] != "w/proposal_head"]
    derived = v3.derive_directory(
        directory, objects, dropped={"w/proposal_head"}, row_scale={"w/mlp_gate_up": (1, 2)},
        metadata={"tensor_parallel": {"ranks": 2, "rank": 1}}, provenance={"split": "test"})
    assert "proposal/head" not in derived["bindings"]
    assert derived["bindings"]["text/layers/3/mlp/gate"]["parts"][0]["range"] == [0, 1]
    assert derived["bindings"]["text/layers/3/mlp/up"]["parts"][0]["range"] == [1, 2]
    assert derived["bindings"]["text/layers/0/gdn/b_projection"]["parts"][0]["range"] == [2, 4]   # not scaled
    assert len(derived["uses"]) == 2
    assert derived["metadata"]["name"] == "fixture-v3" and derived["metadata"]["tensor_parallel"]["rank"] == 1
    assert derived["provenance"] == {"recipe": "qwen3_8_27b_nvfp4", "split": "test"}
    # the source is left untouched
    assert directory["bindings"]["text/layers/3/mlp/gate"]["parts"][0]["range"] == [0, 2]


def test_derive_directory_drops_uses_of_dropped_scalars_and_components():
    directory, _ = _fixture()
    derived = v3.derive_directory(directory, list(directory["objects"]), dropped={"s/mlp_up_divisor"},
                                  row_scale={}, drop_components=("text",))
    assert [u["parameter"] for u in derived["uses"]] == ["text/layers/3/mlp/gate"]
    assert derived["components"] == {}


def test_derive_directory_rejects_inconsistent_requests():
    directory, _ = _fixture()
    with pytest.raises(ValueError):    # binding spans kept and dropped objects
        broken = copy.deepcopy(directory)
        broken["bindings"]["text/layers/3/mlp/gate"]["parts"].append({"object": "w/embedding", "range": [0, 1]})
        v3.derive_directory(broken, list(broken["objects"]), dropped={"w/embedding"}, row_scale={})
    with pytest.raises(ValueError):    # range that does not scale exactly
        v3.derive_directory(directory, list(directory["objects"]), dropped=set(), row_scale={"w/gdn_ab": (1, 4)})
