"""Derived-model patcher (tools/tp2/patch_from_gguf.py): Q8_0 blocks read bit for bit, the GDN value heads of ssm_out
gathered back from llama.cpp's tiled order to the checkpoint's grouped order, FP8 targets re-encoded with the
artifact's own row quantizer, W8 targets copied as Q8_0 blocks, and only the target objects rewritten."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from tools.artifact import layouts as L
from tools.artifact.container import Artifact, ArtifactIdentity, ResourceSpec, TensorSpec, write_artifact
from tools.convert.qwen3_8_27b.fp8_embedding import encode_bf16_rows
from tools.tp2 import patch_from_gguf as P


def q8_tensor(name: str, codes: torch.Tensor, scales: torch.Tensor):
    """A GGUF Q8_0 tensor as gguf.GGUFReader exposes it: uint8 rows of [fp16 scale | 32 int8 codes] blocks."""
    n, groups, _ = codes.shape
    blocks = torch.cat([scales.contiguous().view(torch.uint8).reshape(n, groups, 2),
                        codes.contiguous().view(torch.uint8)], dim=-1)
    return SimpleNamespace(name=name, tensor_type=SimpleNamespace(name="Q8_0"),
                           data=blocks.reshape(n, groups * 34).numpy())


def random_q8(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k // 32, 32), generator=g, dtype=torch.int8)
    scales = (torch.rand(n, k // 32, generator=g) * 0.01 + 1e-4).to(torch.float16)
    return codes, scales


def test_q8_blocks_are_read_bit_for_bit():
    codes, scales = random_q8(3, 96, 1)
    got_codes, got_scales = P.q8_codes_scales(q8_tensor("t", codes, scales))
    assert torch.equal(got_codes, codes) and torch.equal(got_scales.view(torch.int16), scales.view(torch.int16))
    value = P.q8_float(q8_tensor("t", codes, scales))
    assert torch.equal(value, (codes.float() * scales.float().unsqueeze(-1)).reshape(3, 96))


def test_non_q8_tensor_is_rejected():
    t = q8_tensor("t", *random_q8(1, 32, 2))
    t.tensor_type = SimpleNamespace(name="F16")
    with pytest.raises(AssertionError):
        P.q8_codes_scales(t)


def test_ssm_out_value_heads_are_regrouped():
    """llama.cpp tiles the 48 GDN value heads as r * 16 + k, the checkpoint groups them as k * 3 + r: artifact head h
    must come from GGUF head (h % 3) * 16 + h // 3 (measured cosine 1.000 on every layer of Qwen3.8-27B)."""
    heads, dim, rows = P.GDN_K_HEADS * P.GDN_V_PER_K, P.GDN_HEAD_DIM, 2
    codes = torch.zeros(rows, heads * dim // 32, 32, dtype=torch.int8)
    for gguf_head in range(heads):            # every column of GGUF head j carries the code j + 1
        codes.view(rows, heads, dim)[:, gguf_head, :] = gguf_head + 1
    scales = torch.ones(rows, heads * dim // 32, dtype=torch.float16)
    value = P.gguf_value("blk.0.ssm_out.weight", q8_tensor("blk.0.ssm_out.weight", codes, scales))
    got = value.view(rows, heads, dim)
    for h in range(heads):
        k, r = divmod(h, P.GDN_V_PER_K)
        assert torch.all(got[:, h, :] == (r * P.GDN_K_HEADS + k) + 1), h
    # other tensors keep the GGUF column order
    other = P.gguf_value("blk.0.attn_output.weight", q8_tensor("blk.0.attn_output.weight", codes, scales))
    assert torch.equal(other, P.q8_float(q8_tensor("x", codes, scales)))


def _artifact(path, n=8, k=64, kw8=128):
    """A small Qwen3.8-27B-shaped artifact: the five target kinds plus objects that must stay untouched (W8 objects
    use a whole 128-column row-split block: the Q8_0 source carries no padding groups)."""
    g = torch.Generator().manual_seed(9)
    fp8 = lambda rows, cols: encode_bf16_rows(torch.randn(rows, cols, generator=g).to(torch.bfloat16))
    w8 = lambda rows, cols: L.encode_row_split(*random_q8(rows, cols, rows + cols), "W8G32_F16S", (rows, cols))
    gdn_k = P.GDN_K_HEADS * P.GDN_V_PER_K * P.GDN_HEAD_DIM
    entries = [
        (ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 4), b"{}  "),
        (TensorSpec("text/token_embedding", (n, k), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"), fp8(n, k)),
        (TensorSpec("text/layers/0/gdn/output", (n, gdn_k), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"), fp8(n, gdn_k)),
        (TensorSpec("text/layers/3/attention/output", (n, k), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"), fp8(n, k)),
        (TensorSpec("text/layers/3/attention/query_key_gate_value", (n, k), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"),
         fp8(n, k)),
        (TensorSpec("mtp/layer/attention/output", (n, kw8), "W8G32_F16S", "row-split-k128-v1"), w8(n, kw8)),
        (TensorSpec("mtp/layer/mlp/down", (n, kw8), "W8G32_F16S", "row-split-k128-v1"), w8(n, kw8)),
        (TensorSpec("mtp/layer/mlp/gate_up", (n, kw8), "W8G32_F16S", "row-split-k128-v1"), w8(n, kw8)),
    ]
    write_artifact(path, ArtifactIdentity("qwen3_8_27b", "fixture"), entries)
    return {
        "token_embd.weight": q8_tensor("token_embd.weight", *random_q8(n, k, 21)),
        "blk.0.ssm_out.weight": q8_tensor("blk.0.ssm_out.weight", *random_q8(n, gdn_k, 22)),
        "blk.3.attn_output.weight": q8_tensor("blk.3.attn_output.weight", *random_q8(n, k, 23)),
        "blk.64.attn_output.weight": q8_tensor("blk.64.attn_output.weight", *random_q8(n, kw8, 24)),
        "blk.64.ffn_down.weight": q8_tensor("blk.64.ffn_down.weight", *random_q8(n, kw8, 25)),
    }


def test_targets_cover_exactly_the_residual_writers(tmp_path):
    _artifact(tmp_path / "base.ninfer")
    art = Artifact.open(tmp_path / "base.ninfer")
    assert P.targets(art) == [
        ("text/layers/0/gdn/output", "blk.0.ssm_out.weight", "fp8"),
        ("text/layers/3/attention/output", "blk.3.attn_output.weight", "fp8"),
        ("text/token_embedding", "token_embd.weight", "fp8"),
        ("mtp/layer/attention/output", "blk.64.attn_output.weight", "w8"),
        ("mtp/layer/mlp/down", "blk.64.ffn_down.weight", "w8"),
    ]


def test_patch_rewrites_only_the_targets(tmp_path, monkeypatch):
    base_path, out_path = tmp_path / "base.ninfer", tmp_path / "out.ninfer"
    gguf = _artifact(base_path)
    monkeypatch.setattr(P, "gguf_tensors", lambda path: gguf)
    monkeypatch.setattr(sys, "argv", ["patch_from_gguf", "--base", str(base_path), "--gguf", "derived.gguf",
                                      "--out", str(out_path)])
    assert P.main() == 0
    base, out = Artifact.open(base_path), Artifact.open(out_path)
    assert out.objects == base.objects and out.identity == base.identity          # directory untouched
    patched = {name: (tname, kind) for name, tname, kind in P.targets(base)}
    for obj in base.objects:
        got = bytes(out.payload(out.find(obj.name)))
        if obj.name not in patched:
            assert got == bytes(base.payload(obj)), obj.name
            continue
        tname, kind = patched[obj.name]
        if kind == "w8":
            codes, scales = P.q8_codes_scales(gguf[tname])
            assert got == L.encode_row_split(codes, scales, "W8G32_F16S", obj.shape), obj.name
        else:
            expected = encode_bf16_rows(P.gguf_value(tname, gguf[tname]).to(torch.bfloat16))
            assert got == expected, obj.name


def test_check_flags_a_permuted_source(tmp_path, monkeypatch, capsys):
    """--check: the artifact patched from a GGUF agrees with that GGUF (quantization-sized difference) and is
    rejected against a GGUF whose columns are permuted."""
    base_path, out_path = tmp_path / "base.ninfer", tmp_path / "out.ninfer"
    gguf = _artifact(base_path)
    monkeypatch.setattr(P, "gguf_tensors", lambda path: gguf)
    monkeypatch.setattr(sys, "argv", ["patch_from_gguf", "--base", str(base_path), "--gguf", "x", "--out",
                                      str(out_path)])
    assert P.main() == 0
    art = Artifact.open(out_path)
    assert P.check(art, "same.gguf", None) == 0
    permuted = dict(gguf)
    codes, scales = P.q8_codes_scales(gguf["blk.3.attn_output.weight"])
    flipped = codes.reshape(codes.shape[0], -1).flip(1).reshape(codes.shape)
    permuted["blk.3.attn_output.weight"] = q8_tensor("blk.3.attn_output.weight", flipped, scales.flip(1))
    monkeypatch.setattr(P, "gguf_tensors", lambda path: permuted)
    assert P.check(art, "permuted.gguf", None) == 1
    assert "ORDINE DIVERSO" in capsys.readouterr().out
