#!/bin/bash
# build.sh <v100|4090|both> [cmake target...]
#
# Builds the engine for one tensor-parallel rank inside the build container, with a persistent ccache.
#   v100 -> build-v100        sm_70, rank 0, MLP width 7680
#   4090 -> build-ada-volta   sm_89 on the Volta code path (same kernels/layouts as rank 0, plus Ada-only kernels where
#                             they pay off), rank 1, MLP width 9728
# The MLP width of each rank is a compile-time constant and MUST match the weights: prepare-weights.sh shards with
# --mlp-rank0 7680, so rank 0 gets 7680 of the 17408 intermediate columns and rank 1 the other 9728.
#
# Environment:
#   SRC          repository to build (default: this repository)
#   IMAGE        build image (default: ninfer-v100-4090/build:cuda12.8, see v100-4090/docker/Dockerfile.build)
#   CCACHE_HOST  ccache directory on the host (default: ~/.cache/ninfer-ccache)
#   JOBS         parallel compile jobs (default: number of CPUs - 1)
#   MLP_V100     MLP width of rank 0 (default 7680); rank 1 gets 17408 - MLP_V100
#
# A full build takes ~25-40 min per rank on a 6-core CPU; with a warm ccache, a rebuild after a small change takes seconds.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
IMAGE="${IMAGE:-ninfer-v100-4090/build:cuda12.8}"
CCACHE_HOST="${CCACHE_HOST:-$HOME/.cache/ninfer-ccache}"
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
MLP_V100="${MLP_V100:-7680}"
MLP_4090=$(( 17408 - MLP_V100 ))
(( MLP_V100 % 256 == 0 && MLP_V100 > 0 && MLP_V100 < 17408 )) || { echo "MLP_V100 must be a multiple of 256 below 17408"; exit 2; }

TARGET="${1:?usage: build.sh <v100|4090|both> [cmake target...]}"; shift || true
if [ "$TARGET" = both ]; then "$0" v100 "$@" && exec "$0" 4090 "$@"; fi

case "$TARGET" in
  v100) DIR=build-v100;      ARCH=70; OPTS="-DNINFER_TP2_INTERMEDIATE=$MLP_V100" ;;
  4090) DIR=build-ada-volta; ARCH=89; OPTS="-DNINFER_VOLTA_PATH=ON -DNINFER_TP2_INTERMEDIATE=$MLP_4090" ;;
  *) echo "unknown target '$TARGET' (v100, 4090 or both)"; exit 2 ;;
esac

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not found: docker build -f v100-4090/docker/Dockerfile.build -t $IMAGE v100-4090/docker"; exit 2; }
mkdir -p "$CCACHE_HOST"
echo "building $DIR (sm_$ARCH, MLP width $([ "$TARGET" = v100 ] && echo $MLP_V100 || echo $MLP_4090)) from $SRC with $JOBS jobs"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$SRC":/src -v "$CCACHE_HOST":/ccache "$IMAGE" bash -c "
    cmake -S . -B $DIR -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_CUDA_ARCHITECTURES=$ARCH $OPTS \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache &&
    nice -n 15 cmake --build $DIR -j$JOBS ${*:+--target $*}"
ls -la "$SRC/$DIR/apps/ninfer-serve"
