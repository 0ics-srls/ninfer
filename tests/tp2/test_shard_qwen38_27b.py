"""TP2 sharder (tools/tp2/shard_qwen38_27b.py): every split is done on stored words, so the rank pieces of each
segment must reassemble the source bit for bit, for every stored format, on both axes, even and uneven cuts."""

from __future__ import annotations

import pytest
import torch

from tools.artifact import layouts as L
from tools.artifact.container import TensorObject
from tools.tp2 import shard_qwen38_27b as S


def _obj(name, shape, fmt, layout):
    return TensorObject(name, tuple(shape), fmt, layout, 0, L.encoded_size(layout, fmt, tuple(shape)))


def _reassemble(pieces: list[torch.Tensor], rule: S.Rule, dim: int, num: int = 1, den: int = 1) -> torch.Tensor:
    """Inverse of take_frac: interleave each rank's part of every segment back into source order."""
    out, offset = [], [0] * S.TP
    for seg in rule.segments:
        for rank in range(S.TP):
            length = rule.piece(seg, rank)[1] * num // den
            out.append(pieces[rank].narrow(dim, offset[rank], length))
            offset[rank] += length
    return torch.cat(out, dim=dim)


def _fp8_words(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = (torch.randn(n, k, generator=g) * 2).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = (torch.rand(n, generator=g) + 0.5).to(torch.bfloat16)
    return codes, scales


def _w8_words(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k // 32, 32), generator=g, dtype=torch.int8)
    scales = (torch.rand(n, k // 32, generator=g) + 0.01).to(torch.float16)
    return codes, scales


RULES = [
    S.Rule("rows", (64, 128)),                 # two segments, halves (attention qkgv / GDN qkvz style)
    S.Rule("rows", (128, 128), 64),            # uneven cut of a fused gate|up pair (MLP gate_up)
    S.Rule("rows", (128, 128), 96),
]
COL_RULES = [S.Rule("cols", (128,)), S.Rule("cols", (256,), 128), S.Rule("cols", (256,), 64)]
# W8 rows are stored in 128-column blocks (row-split-k128-v1): a K split must cut on a block boundary, as the
# production cuts do (6144 / 2, 17408 at a multiple of 256).
W8_COL_RULES = [S.Rule("cols", (256,)), S.Rule("cols", (512,), 128), S.Rule("cols", (384,), 256)]


def test_rule_pieces_cover_every_segment():
    for rule in RULES + COL_RULES + [S.MLP_GATE_UP, S.K17408, S.ATTN_QKGV, S.GDN_QKVZ, S.GDN_CONV]:
        for seg in rule.segments:
            (b0, l0), (b1, l1) = rule.piece(seg, 0), rule.piece(seg, 1)
            assert b0 == 0 and b1 == l0 and l0 + l1 == seg
        assert S.rank_extent(rule, 0) + S.rank_extent(rule, 1) == sum(rule.segments)


def test_rule_table_maps_qwen38_27b_names():
    assert S.rule_for("text/layers/3/attention/query_key_gate_value") is S.ATTN_QKGV
    assert S.rule_for("text/layers/3/attention/output") is S.K6144
    assert S.rule_for("text/layers/0/gdn/query_key_value_z") is S.GDN_QKVZ
    assert S.rule_for("text/layers/0/gdn/a_b_projection") is S.GDN_AB
    assert S.rule_for("text/layers/0/gdn/a_log") is S.GDN_HEADS
    assert S.rule_for("text/layers/0/gdn/dt_bias") is S.GDN_HEADS
    assert S.rule_for("text/layers/0/gdn/convolution") is S.GDN_CONV
    assert S.rule_for("text/layers/0/gdn/output") is S.K6144
    assert S.rule_for("text/layers/7/mlp/gate_up") is S.MLP_GATE_UP
    assert S.rule_for("text/layers/7/mlp/down") is S.K17408
    assert S.rule_for("mtp/layer/mlp/gate_up") is S.MLP_GATE_UP
    for replicated in ("text/token_embedding", "text/lm_head", "text/layers/0/input_norm", "vision/layers/0/mlp/down",
                       "text/layers/7/mlp/gate_up_projection/input_scale_divisor", "frontend/tokenizer.json"):
        assert S.rule_for(replicated) is None, replicated


def test_default_split_is_even_and_shapes_follow_the_rule():
    assert S.MLP_RANK0 == 8704
    assert S.shard_shape((2 * 17408, 5120), S.MLP_GATE_UP, 0) == (17408, 5120)
    assert S.shard_shape((5120, 17408), S.K17408, 1) == (5120, 8704)
    assert S.shard_shape((14336, 5120), S.ATTN_QKGV, 0) == (7168, 5120)
    uneven = S.Rule("rows", (17408, 17408), 7680)
    assert S.shard_shape((34816, 5120), uneven, 0) == (15360, 5120)
    assert S.shard_shape((34816, 5120), uneven, 1) == (19456, 5120)
    with pytest.raises(AssertionError):
        S.shard_shape((100, 5120), S.MLP_GATE_UP, 0)


@pytest.mark.parametrize("rule", RULES + COL_RULES)
def test_bf16_and_fp32_split_reassembles(rule):
    n, k = (sum(rule.segments), 8) if rule.axis == "rows" else (4, sum(rule.segments))
    dim = 0 if rule.axis == "rows" else 1
    for fmt, dtype in (("BF16", torch.bfloat16), ("FP32", torch.float32)):
        source = torch.randn(n, k, generator=torch.Generator().manual_seed(n + k)).to(dtype)
        obj = _obj("x", (n, k), fmt, "contiguous-le-v1")
        pieces = []
        for rank in range(S.TP):
            payload = S.shard_payload(obj, memoryview(L.encode_direct(source, fmt)), rule, rank)
            pieces.append(L.decode_direct(payload, fmt, S.shard_shape((n, k), rule, rank)))
        assert torch.equal(_reassemble(pieces, rule, dim), source)


@pytest.mark.parametrize("rule", RULES + COL_RULES)
def test_fp8_row_split_reassembles_words(rule):
    n, k = (sum(rule.segments), 16) if rule.axis == "rows" else (8, sum(rule.segments))
    codes, scales = _fp8_words(n, k, 7)
    obj = _obj("x", (n, k), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1")
    raw = memoryview(L.encode_fp8_row_scaled(codes, scales, (n, k)))
    words = [L.decode_fp8_row_scaled_words(S.shard_payload(obj, raw, rule, r), S.shard_shape((n, k), rule, r))
             for r in range(S.TP)]
    if rule.axis == "rows":
        assert torch.equal(_reassemble([w[0] for w in words], rule, 0), codes)
        assert torch.equal(_reassemble([w[1] for w in words], rule, 0), scales)
    else:  # K split: each rank keeps every row scale (its partial sum is scaled by the same row multiplier)
        assert torch.equal(_reassemble([w[0] for w in words], rule, 1), codes)
        assert all(torch.equal(w[1], scales) for w in words)


@pytest.mark.parametrize("rule", RULES + W8_COL_RULES)
def test_w8_split_reassembles_q8_blocks(rule):
    n, k = (sum(rule.segments), 128) if rule.axis == "rows" else (4, sum(rule.segments))
    codes, scales = _w8_words(n, k, 11)
    obj = _obj("x", (n, k), "W8G32_F16S", "row-split-k128-v1")
    raw = memoryview(L.encode_row_split(codes, scales, "W8G32_F16S", (n, k)))
    words = [L.decode_row_split_codes(S.shard_payload(obj, raw, rule, r), "W8G32_F16S",
                                      S.shard_shape((n, k), rule, r)) for r in range(S.TP)]
    dim, den = (0, 1) if rule.axis == "rows" else (1, 32)
    assert torch.equal(_reassemble([w[1] for w in words], rule, dim, 1, den), codes)
    assert torch.equal(_reassemble([w[0] for w in words], rule, dim, 1, den), scales)


@pytest.mark.parametrize("rule", [S.Rule("rows", (128, 128)), S.Rule("rows", (256, 256), 128),
                                  S.Rule("cols", (256,)), S.Rule("cols", (512,), 128)])
def test_nvfp4_split_reassembles_words(rule):
    n, k = (sum(rule.segments), 128) if rule.axis == "rows" else (128, sum(rule.segments))
    g = torch.Generator().manual_seed(5)
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0, 0x7F, (n, k // 16), generator=g, dtype=torch.uint8)
    divisor = torch.tensor(1.75, dtype=torch.float32)
    obj = _obj("x", (n, k), "NVFP4", "blockscale-k16-m128x4-v1")
    raw = memoryview(L.encode_nvfp4(codes, scales, divisor, (n, k)))
    words = [L.decode_nvfp4_words(S.shard_payload(obj, raw, rule, r), S.shard_shape((n, k), rule, r))
             for r in range(S.TP)]
    if rule.axis == "rows":
        assert torch.equal(_reassemble([w[0] for w in words], rule, 0), codes)
        assert torch.equal(_reassemble([w[1] for w in words], rule, 0), scales)
    else:
        assert torch.equal(_reassemble([w[0] for w in words], rule, 1, 1, 2), codes)
        assert torch.equal(_reassemble([w[1] for w in words], rule, 1, 1, 16), scales)
    divisor_bytes = L.encode_direct(divisor.reshape(()), "FP32")
    for w in words:
        assert bytes(L.encode_direct(torch.as_tensor(w[2], dtype=torch.float32).reshape(()), "FP32")) == divisor_bytes


def test_w8_k_split_off_a_block_boundary_is_rejected():
    codes, scales = _w8_words(4, 256, 3)
    obj = _obj("x", (4, 256), "W8G32_F16S", "row-split-k128-v1")
    raw = memoryview(L.encode_row_split(codes, scales, "W8G32_F16S", (4, 256)))
    with pytest.raises(ValueError, match="128-column"):
        S.shard_payload(obj, raw, S.Rule("cols", (256,), 64), 0)


def test_unsupported_format_is_rejected():
    obj = _obj("x", (2, 64), "Q4G64_F16S", "row-split-k128-v1")
    with pytest.raises(ValueError):
        S.shard_payload(obj, memoryview(bytes(obj.bytes)), S.Rule("rows", (2,)), 0)


def test_q8_mlp_payload_takes_gguf_blocks_bit_for_bit(monkeypatch):
    """--mlp-q8: rank r's gate_up = [gate rows of its piece ; up rows of its piece], down = its K groups, as W8G32_F16S
    with the GGUF Q8_0 codes and fp16 scales unchanged (the production MLP of the NInfer ranks)."""
    full, hidden, cut = 768, 5120, 256
    monkeypatch.setattr(S, "MLP_FULL", full)
    blocks = {
        "blk.4.ffn_gate.weight": _w8_words(full, hidden, 1),
        "blk.4.ffn_up.weight": _w8_words(full, hidden, 2),
        "blk.4.ffn_down.weight": _w8_words(hidden, full, 3),
    }
    monkeypatch.setattr(S, "q8_blocks", lambda name: blocks[name])
    gate_up, down = S.Rule("rows", (full, full), cut), S.Rule("cols", (full,), cut)
    gate, up = blocks["blk.4.ffn_gate.weight"], blocks["blk.4.ffn_up.weight"]
    d_codes, d_scales = blocks["blk.4.ffn_down.weight"]
    for rank, (begin, length) in enumerate(((0, cut), (cut, full - cut))):
        payload, shape = S.q8_mlp_payload("text/layers/4/mlp/gate_up", gate_up, rank)
        assert shape == [2 * length, hidden]
        scales, codes = L.decode_row_split_codes(payload, "W8G32_F16S", shape)
        assert torch.equal(codes, torch.cat([gate[0][begin:begin + length], up[0][begin:begin + length]]))
        assert torch.equal(scales, torch.cat([gate[1][begin:begin + length], up[1][begin:begin + length]]))

        payload, shape = S.q8_mlp_payload("text/layers/4/mlp/down", down, rank)
        assert shape == [hidden, length]
        scales, codes = L.decode_row_split_codes(payload, "W8G32_F16S", shape)
        assert torch.equal(codes, d_codes[:, begin // 32:(begin + length) // 32])
        assert torch.equal(scales, d_scales[:, begin // 32:(begin + length) // 32])


def test_text_mlp_detection():
    assert S.is_text_mlp("text/layers/12/mlp/gate_up") and S.is_text_mlp("text/layers/12/mlp/down")
    assert not S.is_text_mlp("mtp/layer/mlp/down")           # the MTP layer keeps its artifact weights
    assert not S.is_text_mlp("text/layers/12/mlp/gate_up_projection/input_scale_divisor")
