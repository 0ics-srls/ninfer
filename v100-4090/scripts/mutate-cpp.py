#!/usr/bin/env python3
"""RED proof for the C++ tests of our engine changes: each mutation must make its test fail on the V100.

  python3 v100-4090/scripts/mutate-cpp.py      (after TESTS=1 v100-4090/scripts/build.sh v100)

Every mutated file is restored, and its target rebuilt, before the next mutation. Last run 2026-10-05: 8/8 RED."""
import os, subprocess, sys

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
V100 = subprocess.run("nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/V100/{print $1}'",
                      shell=True, capture_output=True, text=True).stdout.strip()

MUTATIONS = [
    ("wire8 scale m/128", "src/targets/qwen3_6_27b_tp2/impl/tp2_wire8.cu",
     "if (lane == 0) { scales[block] = m / 127.0f; }", "if (lane == 0) { scales[block] = m / 128.0f; }",
     "ninfer_tp2_wire8_test", [], {}),
    ("frontend v3 pairing rejected", "src/targets/qwen3_6/impl/frontend/frontend.cpp",
     "same_semantics = fi::CompiledChatTemplate::resolve(config_template).semantics() ==",
     "same_semantics = false && fi::CompiledChatTemplate::resolve(config_template).semantics() ==",
     "ninfer_qwen3_6_frontend_test", [], {}),
    ("frontend vision budget env ignored", "src/targets/qwen3_6/impl/frontend/frontend.cpp",
     'std::getenv("NINFER_MAX_PROMPT_VISION_TOKENS")', 'std::getenv("NINFER_MAX_PROMPT_VISION_TOKENS_IGNORED")',
     "ninfer_qwen3_6_frontend_test", [], {}),
    ("v3 reader mtp rename", "src/artifact/reader.cpp",
     'logical = "mtp/layer/" + logical.substr(kMtp.size());', 'logical = "mtp/layers/" + logical.substr(kMtp.size());',
     "ninfer_artifact_reader_test", [], {}),
    ("vision wmma: wrong normalization", "src/ops/softmax_attention/dense/packed/volta_wmma.cuh",
     "sm.corr[warp][row] = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;",
     "sm.corr[warp][row] = row_sum > 0.0f ? 0.98f / row_sum : 0.0f;",
     "ninfer_softmax_attention_test", ["--packed-only"], {}),
    ("gdn fused: rms off by 1%", "src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu",
     "const float inv = rsqrtf(s / D + eps);", "const float inv = rsqrtf(s / D * 0.98f + eps);",
     "ninfer_gdn_gating_proj_test", [], {}),
    ("vision scalar (mode 0): no rescale", "src/ops/softmax_attention/dense/packed/volta.cuh",
     "acc[item] = acc[item] * old_scale + probability * value;", "acc[item] = acc[item] + probability * value;",
     "ninfer_softmax_attention_test", ["--packed-only"], {"NINFER_VISION_ATTN": "0"}),
    ("vision tiled (mode 1): no rescale", "src/ops/softmax_attention/dense/packed/volta.cuh",
     "float a0 = acc[i][0] * corr, a1 = acc[i][1] * corr, a2 = acc[i][2] * corr;",
     "float a0 = acc[i][0], a1 = acc[i][1], a2 = acc[i][2];",
     "ninfer_softmax_attention_test", ["--packed-only"], {"NINFER_VISION_ATTN": "1"}),
]


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=SRC, capture_output=True, text=True)


def build(target):
    r = sh(f"TESTS=1 v100-4090/scripts/build.sh v100 {target}")
    return r.returncode == 0


def run(target, args, env):
    envs = " ".join(f"-e {k}={v}" for k, v in env.items())
    r = sh(f"docker run --rm --user $(id -u):$(id -g) --gpus device={V100} {envs} -v {SRC}:/src "
           f"ninfer-v100-4090/build:cuda12.8 /src/build-v100/tests/{target} {' '.join(args)}")
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]


for label, path, old, new, target, args, env in MUTATIONS:
    full = os.path.join(SRC, path)
    original = open(full, encoding="utf-8").read()
    if original.count(old) != 1:
        print(f"SKIP  {label}: anchor found {original.count(old)} times"); continue
    try:
        open(full, "w", encoding="utf-8").write(original.replace(old, new))
        if not build(target):
            print(f"?     {label}: mutant does not build"); continue
        rc, tail = run(target, args, env)
        print(f"{'RED  ' if rc not in (0, 77) else 'GREEN!!'} {label}: exit {rc} {tail}")
    finally:
        open(full, "w", encoding="utf-8").write(original)
        os.utime(full)
        build(target)
sys.stdout.flush()
