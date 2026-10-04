"""Split a Qwen3.8-27B NInfer artifact into two tensor-parallel rank artifacts (a v3 source gives v3 ranks).

Rank r of 2 keeps half of every head / intermediate dimension:
  * column-parallel (split output rows N): attention qkgv, GDN qkvz, GDN a/b, a_log, dt_bias,
    GDN convolution channels, MLP gate_up, and the same for the MTP layer;
  * row-parallel (split input columns K): attention output, GDN output, MLP down (each rank then
    produces a partial sum that the runtime all-reduces);
  * everything else (embeddings, LM/draft heads, norms, divisors, vision, resources) is copied.

Every split is done on exact stored words (FP8 codes + row scales, NVFP4 packed codes + natural
scales + divisor, W8 groups + scales, BF16/FP32 values) and re-encoded with tools.artifact, so the
two halves re-concatenate bit-exactly to the source (see --verify).

usage: python -m tools.tp2.shard_qwen38_27b SRC OUT_PREFIX [--verify] [--drop-dflash2] [--mlp-rank0 N]
       --mlp-q8 GGUF takes the text MLP of every layer from a llama.cpp Q8_0 GGUF of the same checkpoint (blk.L.ffn_gate /
       ffn_up / ffn_down), stored as W8G32_F16S: the Q8_0 blocks bit for bit (v3 source only).
       --mlp-rank0 N gives rank 0 the first N of the 17408 MLP intermediate columns (multiple of 256; default 8704 =
       even split) and rank 1 the rest; each rank's binary must be built with that width (NINFER_TP2_INTERMEDIATE).
       --drop-dflash2 leaves the DFlash2 companion weights (optional component, not sharded here) out of both ranks.
       writes OUT_PREFIX.rank0.ninfer and OUT_PREFIX.rank1.ninfer
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import torch

from tools.artifact.container import (Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec,
                                      TensorObject, TensorSpec)
from tools.artifact import layouts as L
from tools.artifact.v3 import V3Writer, derive_directory

TP = 2


MLP_FULL = 17408
MLP_RANK0 = int(sys.argv[sys.argv.index("--mlp-rank0") + 1]) if "--mlp-rank0" in sys.argv else MLP_FULL // 2
assert 0 < MLP_RANK0 < MLP_FULL and MLP_RANK0 % 256 == 0, "--mlp-rank0 must be a multiple of 256 below 17408"


@dataclass(frozen=True)
class Rule:
    axis: str                  # "rows" (N) or "cols" (K)
    segments: tuple[int, ...]  # lengths of consecutive logical segments along that axis; each is split
    cut: int | None = None     # rank 0 keeps [0, cut) of every segment, rank 1 the rest; None = halves

    def piece(self, seg: int, rank: int) -> tuple[int, int]:
        """(start, length) of `rank`'s part of one segment, in logical elements."""
        if self.cut is None:
            return rank * (seg // TP), seg // TP
        return (0, self.cut) if rank == 0 else (self.cut, seg - self.cut)


ATTN_QKGV = Rule("rows", (6144, 1024, 6144, 1024))
GDN_QKVZ = Rule("rows", (2048, 2048, 6144, 6144))
GDN_AB = Rule("rows", (48, 48))
GDN_HEADS = Rule("rows", (48,))
GDN_CONV = Rule("cols", (2048, 2048, 6144))     # (4, 10240) taps x channels
MLP_GATE_UP = Rule("rows", (MLP_FULL, MLP_FULL), MLP_RANK0)
K6144 = Rule("cols", (6144,))
K17408 = Rule("cols", (MLP_FULL,), MLP_RANK0)


def rule_for(name: str) -> Rule | None:
    if name.startswith("vision/") or not (name.startswith("text/layers/") or name.startswith("mtp/layer/")):
        return None
    table = {
        "/attention/query_key_gate_value": ATTN_QKGV,
        "/attention/output": K6144,
        "/gdn/query_key_value_z": GDN_QKVZ,
        "/gdn/a_b_projection": GDN_AB,
        "/gdn/a_log": GDN_HEADS,
        "/gdn/dt_bias": GDN_HEADS,
        "/gdn/convolution": GDN_CONV,
        "/gdn/output": K6144,
        "/mlp/gate_up": MLP_GATE_UP,
        "/mlp/down": K17408,
    }
    for suffix, rule in table.items():
        if name.endswith(suffix):
            return rule
    return None


def rank_extent(rule: Rule, rank: int) -> int:
    return sum(rule.piece(seg, rank)[1] for seg in rule.segments)


def shard_shape(shape: tuple[int, ...], rule: Rule, rank: int = 0) -> tuple[int, ...]:
    if rule.axis == "rows":
        assert shape[0] == sum(rule.segments), (shape, rule)
        return (rank_extent(rule, rank),) + tuple(shape[1:])
    assert shape[-1] == sum(rule.segments), (shape, rule)
    return tuple(shape[:-1]) + (rank_extent(rule, rank),)


def take_frac(t: torch.Tensor, dim: int, rule: Rule, rank: int, num: int = 1, den: int = 1) -> torch.Tensor:
    """Concatenate rank's part of each logical segment along `dim`; storage elements per logical element
    along that dim = num/den (1/2 for packed NVFP4 codes, 1/16 for their scales, 1/32 for W8 groups)."""
    parts, start = [], 0
    for seg in rule.segments:
        begin, length = rule.piece(seg, rank)
        assert (seg * num) % den == 0 and (begin * num) % den == 0 and (length * num) % den == 0, (seg, begin, length)
        parts.append(t.narrow(dim, start + begin * num // den, length * num // den))
        start += seg * num // den
    assert start == t.shape[dim], (start, t.shape, dim)
    return torch.cat(parts, dim=dim).contiguous()


def take(t: torch.Tensor, dim: int, rule: Rule, rank: int) -> torch.Tensor:
    return take_frac(t, dim, rule, rank)


def shard_payload(obj: TensorObject, raw: memoryview, rule: Rule, rank: int) -> bytes:
    shape = tuple(obj.shape)
    new_shape = shard_shape(shape, rule, rank)
    dim = 0 if rule.axis == "rows" else len(shape) - 1
    fmt = obj.format
    if fmt == "FP8_E4M3FN_ROW_BF16S":
        codes, scales = L.decode_fp8_row_scaled_words(raw, shape)          # [N,K] u8, [N] bf16
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule, rank), take(scales, 0, rule, rank)
        else:
            codes = take(codes, 1, rule, rank)
        return L.encode_fp8_row_scaled(codes, scales, new_shape)
    if fmt == "NVFP4":
        codes, scales, divisor = L.decode_nvfp4_words(raw, shape)          # [N,K/2], [N,K/16], ()
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule, rank), take(scales, 0, rule, rank)
        else:
            codes = take_frac(codes, 1, rule, rank, 1, 2)
            scales = take_frac(scales, 1, rule, rank, 1, 16)
        return L.encode_nvfp4(codes, scales, divisor, new_shape)
    if fmt == "W8G32_F16S":
        scales, codes = L.decode_row_split_codes(raw, fmt, shape)          # [N,G], [N,G,32]
        geom = L.row_split_geometry(fmt, shape)
        if rule.axis == "rows":
            codes, scales = take(codes, 0, rule, rank), take(scales, 0, rule, rank)
        else:
            assert geom.groups_per_row * 32 == shape[-1], "padded K not supported for K split"
            if any(rule.piece(seg, r)[1] % 128 for seg in rule.segments for r in range(TP)):
                raise ValueError(f"W8 K split of {obj.name} must cut on 128-column row-split blocks: {rule}")
            codes = take_frac(codes, 1, rule, rank, 1, 32)
            scales = take_frac(scales, 1, rule, rank, 1, 32)
        return L.encode_row_split(codes, scales, fmt, new_shape)
    if fmt in ("BF16", "FP32"):
        t = L.decode_direct(raw, fmt, shape)
        t = take(t, dim, rule, rank)
        return L.encode_direct(t, fmt)
    raise ValueError(f"no shard rule for format {fmt} ({obj.name})")


DROP = ("dflash2/",) if "--drop-dflash2" in sys.argv else ()


def kept(name: str) -> bool:
    return not name.startswith(DROP) if DROP else True


MLP_Q8 = sys.argv[sys.argv.index("--mlp-q8") + 1] if "--mlp-q8" in sys.argv else None
_Q8_TENSORS = None


def q8_blocks(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes int8 [N, K/32, 32], scales fp16 [N, K/32]) of one Q8_0 GGUF tensor."""
    global _Q8_TENSORS
    if _Q8_TENSORS is None:
        from gguf import GGUFReader
        _Q8_TENSORS = {t.name: t for t in GGUFReader(MLP_Q8).tensors}
    t = _Q8_TENSORS[name]
    assert t.tensor_type.name == "Q8_0", (name, t.tensor_type.name)
    raw = torch.from_numpy(t.data.copy())                # [N, K/32 * 34] bytes
    blocks = raw.view(raw.shape[0], -1, 34)
    scales = blocks[:, :, 0:2].contiguous().view(torch.float16).squeeze(-1)
    codes = blocks[:, :, 2:34].contiguous().view(torch.int8)
    return codes, scales


def q8_mlp_payload(name: str, rule: Rule, rank: int) -> tuple[bytes, list[int]]:
    """W8G32_F16S payload and shape of this rank's MLP object, from the Q8_0 GGUF."""
    layer = int(name.split("/")[2])
    if name.endswith("/mlp/gate_up"):
        begin, length = rule.piece(MLP_FULL, rank)
        parts = [q8_blocks(f"blk.{layer}.ffn_{half}.weight") for half in ("gate", "up")]
        codes = torch.cat([c[begin:begin + length] for c, _ in parts]).contiguous()
        scales = torch.cat([sc[begin:begin + length] for _, sc in parts]).contiguous()
        shape = [2 * length, 5120]
    else:
        begin, length = rule.piece(MLP_FULL, rank)
        codes, scales = q8_blocks(f"blk.{layer}.ffn_down.weight")
        codes = codes[:, begin // 32:(begin + length) // 32].contiguous()
        scales = scales[:, begin // 32:(begin + length) // 32].contiguous()
        shape = [5120, length]
    return L.encode_row_split(codes, scales, "W8G32_F16S", shape), shape


def is_text_mlp(name: str) -> bool:
    return name.startswith("text/layers/") and (name.endswith("/mlp/gate_up") or name.endswith("/mlp/down"))


def build_v3(art: Artifact, prefix: str) -> None:
    """v3 source -> v3 ranks: same ids, bindings and uses; split objects get their rank shape, row-split fused
    objects get halved binding ranges, identity carries metadata.tensor_parallel."""
    # With --mlp-q8 the NVFP4 activation divisors of the MLP have no consumer any more: they leave with the NVFP4.
    q8_scalar = re.compile(r"^text/layers/\d+/mlp/(gate_up|down)_projection/input_scale_divisor$")
    def kept_here(name: str) -> bool:
        return kept(name) and not (MLP_Q8 and q8_scalar.match(name))
    keep = [(raw, obj) for raw, obj in art.v3_entries if kept_here(obj.name)]
    dropped = {raw["id"] for raw, obj in art.v3_entries if not kept_here(obj.name)}
    rules = {raw["id"]: rule_for(obj.name) if isinstance(obj, TensorObject) else None for raw, obj in keep}
    parted = {p["object"] for b in art.v3["bindings"].values() for p in b.get("parts", [])}
    for oid, rule in rules.items():
        assert not (rule and rule.axis == "cols" and oid in parted), f"K split of fused object {oid}"
    writers = []
    for r in range(TP):
        # Every fused part starts on a segment boundary, so its range scales with the rank's share of the segment.
        row_scale = {oid: (rank_extent(rule, r), sum(rule.segments)) for oid, rule in rules.items()
                     if rule and rule.axis == "rows"}
        objects = []
        for raw, obj in keep:
            rule = rules[raw["id"]]
            if MLP_Q8 and is_text_mlp(obj.name):
                shape = list(shard_shape(tuple(obj.shape), rule, r))
                objects.append({**raw, "shape": shape, "format": "q8_g32_fp16", "layout": "row_split_k128_v1"})
            else:
                objects.append({**raw, "shape": list(shard_shape(tuple(obj.shape), rule, r))} if rule else dict(raw))
        directory = derive_directory(
            art.v3, objects, dropped=dropped, row_scale=row_scale,
            metadata={"tensor_parallel": {"ranks": TP, "rank": r,
                                          "mlp_intermediate": [MLP_RANK0, MLP_FULL - MLP_RANK0],
                                          **({"mlp_source": "Q8_0 GGUF, W8G32_F16S"} if MLP_Q8 else {})}},
            provenance={"tensor_parallel_split": {"tool": "tools.tp2.shard_qwen38_27b",
                                                  "source_artifact_id": art.artifact_id.hex()}},
            drop_components=("dflash2",) if DROP else ())
        writers.append(V3Writer(f"{prefix}.rank{r}.ninfer", directory))
    for i, (raw, obj) in enumerate(keep):
        payload = art.payload(obj)
        rule = rules[raw["id"]]
        for r in range(TP):
            if MLP_Q8 and is_text_mlp(obj.name):
                writers[r].write(raw["id"], q8_mlp_payload(obj.name, rule, r)[0])
            else:
                writers[r].write(raw["id"], shard_payload(obj, payload, rule, r) if rule else payload)
        if rule and i % 40 == 0:
            print(f"[{i}/{len(keep)}] {obj.name} {tuple(obj.shape)} -> {shard_shape(tuple(obj.shape), rule, 0)}"
                  f" + {shard_shape(tuple(obj.shape), rule, 1)}", flush=True)
    for w in writers:
        w.finish()
    print("done", flush=True)


def build(src: str, prefix: str) -> None:
    art = Artifact.open(src)
    if art.v3 is not None:
        build_v3(art, prefix)
        return
    specs = []
    for obj in art.objects:
        if not kept(obj.name):
            continue
        if isinstance(obj, TensorObject):
            rule = rule_for(obj.name)
            assert rule is None or rule.cut is None or MLP_RANK0 == MLP_FULL // 2, "uneven split needs a v3 source"
            shape = shard_shape(tuple(obj.shape), rule) if rule else tuple(obj.shape)
            specs.append(TensorSpec(obj.name, shape, obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
    identity = ArtifactIdentity(art.identity.model_id, art.identity.weights_id + "-tp2")
    writers = [ArtifactWriter(f"{prefix}.rank{r}.ninfer", identity, specs) for r in range(TP)]
    for i, obj in enumerate(art.objects):
        if not kept(obj.name):
            continue
        raw = art.payload(obj)
        rule = rule_for(obj.name) if isinstance(obj, TensorObject) else None
        for r in range(TP):
            writers[r].write(obj.name, shard_payload(obj, raw, rule, r) if rule else raw)
        if rule and i % 40 == 0:
            print(f"[{i}/{len(art.objects)}] {obj.name} {tuple(obj.shape)} -> {shard_shape(tuple(obj.shape), rule)}", flush=True)
    for w in writers:
        w.finish()
    print("done", flush=True)


def verify(src: str, prefix: str, limit: int = 0) -> None:
    """Re-concatenate each split tensor from the two ranks and compare words with the source."""
    art = Artifact.open(src)
    ranks = [Artifact.open(f"{prefix}.rank{r}.ninfer") for r in range(TP)]
    checked = 0
    for obj in art.objects:
        if not kept(obj.name):
            continue
        if not isinstance(obj, TensorObject):
            assert all(bytes(rk.payload(obj.name)) == bytes(art.payload(obj)) for rk in ranks)
            continue
        rule = rule_for(obj.name)
        if rule is None:
            for rk in ranks:
                assert bytes(rk.payload(obj.name)) == bytes(art.payload(obj)), obj.name
            continue
        # Round trip: shard again from source and compare to the stored rank payloads.
        for r, rk in enumerate(ranks):
            assert bytes(rk.payload(obj.name)) == shard_payload(obj, art.payload(obj), rule, r), obj.name
        # Semantic check on dequantized values: rank halves of each segment reassemble the source.
        if obj.format in ("FP8_E4M3FN_ROW_BF16S",):
            full = L.dequantize_fp8_row_scaled(art.payload(obj), obj.shape)
            halves = [L.dequantize_fp8_row_scaled(rk.payload(obj.name), rk.find(obj.name).shape) for rk in ranks]
            dim = 0 if rule.axis == "rows" else 1
            rebuilt, off = [], [0, 0]
            for seg in rule.segments:
                for r in range(TP):
                    h = rule.piece(seg, r)[1]
                    rebuilt.append(halves[r].narrow(dim, off[r], h)); off[r] += h
            assert torch.equal(torch.cat(rebuilt, dim=dim), full), obj.name
        checked += 1
        if limit and checked >= limit:
            break
    print(f"verify ok ({checked} split tensors)", flush=True)


if __name__ == "__main__":
    src, prefix = sys.argv[1], sys.argv[2]
    if "--verify" in sys.argv:
        verify(src, prefix, limit=int(sys.argv[sys.argv.index("--verify") + 1]) if len(sys.argv) > sys.argv.index("--verify") + 1 else 0)
    else:
        build(src, prefix)
