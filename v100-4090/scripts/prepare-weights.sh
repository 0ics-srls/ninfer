#!/bin/bash
# prepare-weights.sh [--uncensored] [--verify]
#
# Builds the two tensor-parallel rank artifacts the engine loads (rank 0 = V100, rank 1 = RTX 4090) from public files.
# Nothing is retrained or re-quantized from scratch: the artifact is a repackaging of the official weights.
#
#   1. neroued/Qwen3.8-27B-nvfp4-NInfer : qwen3_8_27b_nvfp4.ninfer   (official Qwen3.8-27B as a NInfer v3 artifact;
#                                          attention and Gated DeltaNet are the official FP8 tensors)
#   2. ggml-org/Qwen3.8-27B-GGUF        : Qwen3.8-27B-Q8_0.gguf      (the text MLP is taken from here, Q8_0 blocks bit
#                                          for bit, instead of the NVFP4 MLP of the artifact)
#   3. tools/tp2/shard_qwen38_27b.py     splits every head / intermediate dimension in two: MLP columns 7680 (V100) +
#                                          9728 (4090), drops the DFlash2 companion (this setup uses MTP drafts)
#
# --uncensored also builds the orcarouter Qwen3.8-27B-Uncensored ranks: the abliteration only changes the matrices
# that write into the residual stream, so tools/tp2/patch_from_gguf.py copies the base artifact and rewrites those
# tensors from bartowski's Q8_0 GGUF of the Uncensored model; the MLP then comes from that same GGUF.
#
# Output (MODELS, default ~/models/ninfer):
#   tp2/qwen3_8_27b_q8mlp.mlp7680.rank{0,1}.ninfer             ~15.1 GB + ~14.6 GB
#   tp2/qwen3_8_27b_uncensored_q8mlp.mlp7680.rank{0,1}.ninfer  (with --uncensored)
# Disk: ~24 GB + ~29 GB of downloads, ~30 GB per model of ranks (the downloads can be deleted afterwards).
# Time: download, then ~15-20 min of CPU per model for the sharding (the patch adds ~15 min).
#
# Environment: MODELS (output and download directory), HF_TOKEN (only if Hugging Face rate-limits you),
#              IMAGE (tools image, default ninfer-v100-4090/tools), MLP_V100 (default 7680, must match build.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
MODELS="${MODELS:-$HOME/models/ninfer}"
IMAGE="${IMAGE:-ninfer-v100-4090/tools}"
MLP_V100="${MLP_V100:-7680}"
UNCENSORED=0; VERIFY=""
for a in "$@"; do
  case "$a" in
    --uncensored) UNCENSORED=1 ;;
    --verify) VERIFY="--verify" ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not found: docker build -f v100-4090/docker/Dockerfile.tools -t $IMAGE v100-4090/docker"; exit 2; }
mkdir -p "$MODELS/download" "$MODELS/tp2"

tools() {   # run a command in the tools container with the repo at /src and MODELS at /m
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp ${HF_TOKEN:+-e HF_TOKEN} \
    -v "$SRC":/src:ro -v "$MODELS":/m -w /src "$IMAGE" "$@"
}
fetch() {   # fetch <repo> <file>: download once into MODELS/download
  if [ -s "$MODELS/download/$2" ]; then echo "have $2"; return; fi
  echo "downloading $1 :: $2"
  tools hf download "$1" "$2" --local-dir /m/download
}

fetch neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer
fetch ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf

OUT=qwen3_8_27b_q8mlp.mlp$MLP_V100
if [ -s "$MODELS/tp2/$OUT.rank1.ninfer" ]; then
  echo "have tp2/$OUT.rank{0,1}.ninfer"
else
  echo "sharding the official model -> tp2/$OUT.rank{0,1}.ninfer"
  tools python -m tools.tp2.shard_qwen38_27b /m/download/qwen3_8_27b_nvfp4.ninfer /m/tp2/$OUT \
    --mlp-rank0 "$MLP_V100" --mlp-q8 /m/download/Qwen3.8-27B-Q8_0.gguf --drop-dflash2 $VERIFY
fi

if [ "$UNCENSORED" = 1 ]; then
  fetch bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf
  BASE_U=/m/download/qwen3_8_27b_uncensored.ninfer
  if [ ! -s "$MODELS/download/qwen3_8_27b_uncensored.ninfer" ]; then
    echo "checking that the Uncensored GGUF lines up with the base (expect a few % per tensor, not ~1.4)"
    tools python -m tools.tp2.patch_from_gguf --base /m/download/qwen3_8_27b_nvfp4.ninfer \
      --gguf /m/download/orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf \
      --gguf-base /m/download/Qwen3.8-27B-Q8_0.gguf --check
    echo "patching the abliterated tensors -> download/qwen3_8_27b_uncensored.ninfer"
    tools python -m tools.tp2.patch_from_gguf --base /m/download/qwen3_8_27b_nvfp4.ninfer \
      --gguf /m/download/orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf --out "$BASE_U"
  fi
  OUT_U=qwen3_8_27b_uncensored_q8mlp.mlp$MLP_V100
  if [ -s "$MODELS/tp2/$OUT_U.rank1.ninfer" ]; then
    echo "have tp2/$OUT_U.rank{0,1}.ninfer"
  else
    echo "sharding the Uncensored model -> tp2/$OUT_U.rank{0,1}.ninfer"
    tools python -m tools.tp2.shard_qwen38_27b "$BASE_U" /m/tp2/$OUT_U \
      --mlp-rank0 "$MLP_V100" --mlp-q8 /m/download/orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf --drop-dflash2 $VERIFY
  fi
fi

ls -la "$MODELS/tp2"
