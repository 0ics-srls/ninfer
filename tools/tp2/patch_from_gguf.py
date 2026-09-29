"""Patch a Qwen3.8-27B NInfer artifact with the tensors a derived model (abliteration) changed, taken from its Q8_0 GGUF.

An abliterated/uncensored Qwen3.8-27B differs from the base only in the matrices that write into the residual stream
(checked tensor by tensor on the GGUFs): attention output, GDN output, MLP down, token embedding, and the same two
matrices of the MTP layer. This tool copies the base artifact and rewrites, in place, the payload of those objects:

  text/layers/N/attention/output   <- blk.N.attn_output   Q8_0 -> FP8 E4M3 row-scaled (single rounding from ~BF16)
  text/layers/N/gdn/output         <- blk.N.ssm_out       Q8_0 -> FP8 E4M3 row-scaled (value heads re-grouped, see gguf_value)
  text/token_embedding             <- token_embd          Q8_0 -> FP8 E4M3 row-scaled
  mtp/layer/attention/output       <- blk.64.attn_output  Q8_0 bit for bit (W8G32_F16S)
  mtp/layer/mlp/down               <- blk.64.ffn_down     Q8_0 bit for bit (W8G32_F16S)

The text MLP (gate_up/down) is taken bit for bit by the TP2 sharder (`shard_qwen38_27b.py --mlp-q8 <same gguf>`).
Every payload keeps the base object's format, layout and size, so the directory is untouched.

  python3 -m tools.tp2.patch_from_gguf --base BASE.ninfer --gguf DERIVED-Q8_0.gguf --check [--gguf-base BASE-Q8_0.gguf]
  python3 -m tools.tp2.patch_from_gguf --base BASE.ninfer --gguf DERIVED-Q8_0.gguf --out OUT.ninfer

--check compares, before writing anything, each target object of the base artifact with the same tensor of a GGUF made
from the SAME weights (--gguf-base): an unchanged column order gives a quantization-sized difference (a few percent),
a permuted one a difference of the order of 1.4. Run it before trusting a new GGUF source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

from tools.artifact import layouts as L
from tools.artifact.container import Artifact, TensorObject
from tools.convert.qwen3_8_27b.fp8_embedding import encode_bf16_rows


def gguf_tensors(path: str):
    from gguf import GGUFReader
    return {t.name: t for t in GGUFReader(path).tensors}


def q8_codes_scales(t) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes int8 [N, K/32, 32], scales fp16 [N, K/32]) of one Q8_0 GGUF tensor."""
    assert t.tensor_type.name == "Q8_0", (t.name, t.tensor_type.name)
    raw = torch.from_numpy(t.data.copy())
    blocks = raw.view(raw.shape[0], -1, 34)
    scales = blocks[:, :, 0:2].contiguous().view(torch.float16).squeeze(-1)
    codes = blocks[:, :, 2:34].contiguous().view(torch.int8)
    return codes, scales


def q8_float(t) -> torch.Tensor:
    codes, scales = q8_codes_scales(t)
    return (codes.float() * scales.float().unsqueeze(-1)).reshape(codes.shape[0], -1)


# llama.cpp's converter stores the GDN value heads tiled (head r * K_HEADS + k), while the HF checkpoint and NInfer group
# them by key head (k * V_PER_K + r). Measured on Qwen3.8-27B: artifact head h matches GGUF head (h % 3) * 16 + h // 3 with
# cosine 1.000 on every layer. ssm_out's input columns follow the value heads, so they are gathered back here.
GDN_K_HEADS, GDN_V_PER_K, GDN_HEAD_DIM = 16, 3, 128


def gguf_value(tname: str, t) -> torch.Tensor:
    value = q8_float(t)
    if tname.endswith(".ssm_out.weight"):
        heads = GDN_K_HEADS * GDN_V_PER_K
        assert value.shape[1] == heads * GDN_HEAD_DIM, (tname, tuple(value.shape))
        order = [(h % GDN_V_PER_K) * GDN_K_HEADS + h // GDN_V_PER_K for h in range(heads)]
        value = value.view(value.shape[0], heads, GDN_HEAD_DIM)[:, order, :].reshape(value.shape[0], -1)
    return value




def targets(art: Artifact) -> list[tuple[str, str, str]]:
    """(object name, gguf tensor, kind) for every object to rewrite."""
    names = {o.name for o in art.objects if isinstance(o, TensorObject)}
    out = []
    for i in range(64):
        if f"text/layers/{i}/attention/output" in names:
            out.append((f"text/layers/{i}/attention/output", f"blk.{i}.attn_output.weight", "fp8"))
        if f"text/layers/{i}/gdn/output" in names:
            out.append((f"text/layers/{i}/gdn/output", f"blk.{i}.ssm_out.weight", "fp8"))
    out.append(("text/token_embedding", "token_embd.weight", "fp8"))
    out.append(("mtp/layer/attention/output", "blk.64.attn_output.weight", "w8"))
    out.append(("mtp/layer/mlp/down", "blk.64.ffn_down.weight", "w8"))
    return [t for t in out if t[0] in names]


def new_payload(obj: TensorObject, t, kind: str) -> bytes:
    shape = list(obj.shape)
    if kind == "fp8":
        assert obj.format == "FP8_E4M3FN_ROW_BF16S", (obj.name, obj.format)
        value = gguf_value(t.name, t)
        assert list(value.shape) == shape, (obj.name, list(value.shape), shape)
        return encode_bf16_rows(value.to(torch.bfloat16))
    assert obj.format == "W8G32_F16S", (obj.name, obj.format)
    codes, scales = q8_codes_scales(t)
    return L.encode_row_split(codes.contiguous(), scales.contiguous(), "W8G32_F16S", shape)


def check(art: Artifact, gguf_base: str, gguf_mtp: str | None) -> int:
    """FP8 targets: relative difference of the values. W8 (MTP) targets: share of identical payload bytes against the
    same tensor re-encoded from the base model's MTP GGUF (the official sidecar)."""
    tensors = gguf_tensors(gguf_base)
    mtp = gguf_tensors(gguf_mtp) if gguf_mtp else {}
    worst, checked = 0.0, 0
    for name, tname, kind in targets(art):
        obj = art.find(name)
        if kind == "w8":
            if tname not in mtp:
                print(f"  {name:36s} <- {tname}: senza --gguf-base-mtp, salto"); continue
            codes, scales = q8_codes_scales(mtp[tname])
            ref = L.encode_row_split(codes.contiguous(), scales.contiguous(), "W8G32_F16S", list(obj.shape))
            mine = bytes(art.payload(obj))
            same = sum(1 for x, y in zip(mine, ref) if x == y) / max(len(ref), 1) if len(mine) == len(ref) else 0.0
            print(f"  {name:36s} <- {tname:28s} byte identici {same:.4f}")
            continue
        if tname not in tensors:
            print(f"  {name:36s} <- {tname}: assente nel GGUF di riferimento, salto"); continue
        mine = L.dequantize_fp8_row_scaled(art.payload(obj), list(obj.shape))
        ref = gguf_value(tname, tensors[tname])
        rel = float((mine - ref).norm() / ref.norm())
        worst, checked = max(worst, rel), checked + 1
        if rel > 0.08 or name.endswith(("/3/attention/output", "/0/gdn/output")) or not name.startswith("text/layers/"):
            print(f"  {name:36s} <- {tname:28s} diff relativa {rel:.4f}")
    print(f"FP8: peggiore {worst:.4f} su {checked} oggetti ({'ordine coerente' if worst < 0.08 else 'ORDINE DIVERSO: non usare'})")
    return 0 if worst < 0.08 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--gguf", required=True, help="Q8_0 GGUF of the derived model")
    ap.add_argument("--gguf-base", help="Q8_0 GGUF of the base model (for --check)")
    ap.add_argument("--gguf-base-mtp", help="Q8_0 GGUF of the base model's MTP layer, e.g. the sidecar (for --check)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    art = Artifact(a.base)
    if a.check:
        return check(art, a.gguf_base or a.gguf, a.gguf_base_mtp)
    if not a.out:
        ap.error("--out is required unless --check")
    tensors = gguf_tensors(a.gguf)
    work = [(art.find(n), tensors[t], k) for n, t, k in targets(art)]
    plan = [(obj, new_payload(obj, t, k)) for obj, t, k in work]
    for obj, data in plan:
        assert len(data) == len(art.payload(obj)), (obj.name, len(data), len(art.payload(obj)))
    tmp = a.out + ".tmp"
    shutil.copyfile(a.base, tmp)
    with open(tmp, "r+b") as f:
        for obj, data in plan:
            f.seek(art.payload_offset + obj.offset)
            f.write(data)
    os.replace(tmp, a.out)
    print(f"{a.out}: {len(plan)} oggetti riscritti "
          f"({sum(1 for _, _, k in work if k == 'fp8')} FP8 per riga, {sum(1 for _, _, k in work if k == 'w8')} W8)")
    out = Artifact(a.out)
    for obj, data in plan[:3] + plan[-2:]:
        assert bytes(out.payload(out.find(obj.name))) == data, obj.name
    print("rilettura ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
