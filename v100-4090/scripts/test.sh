#!/bin/bash
# test.sh <v100|4090|python> [arguments...]
#
# Runs the test suite in the containers, on the card of the rank (selected by name, never by index):
#   v100    ctest of build-v100 (TESTS=1 scripts/build.sh v100) on the Tesla V100
#   4090    ctest of build-ada-volta (TESTS=1 scripts/build.sh 4090) on the RTX 4090
#   python  pytest of the Python tools (artifact reader/writer, converters, tools/tp2) in the tools image, CPU only
# The C++ suite links the CUDA driver even for its host-only tests, so it always runs with the rank's GPU: a run
# without a GPU proves nothing. Extra arguments go to ctest / pytest. Examples:
#   v100-4090/scripts/test.sh v100                              # whole suite on the V100
#   v100-4090/scripts/test.sh 4090 -R softmax_attention         # attention tests on the RTX 4090
#   v100-4090/scripts/test.sh python tests/tp2                  # TP2 tool tests
# Optional inputs of the original suite, passed through when set:
#   NINFER_QWEN3_6_FRONTEND_RESOURCES   directory with the official Qwen3.6 frontend JSONs (frontend test)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
IMAGE="${IMAGE:-ninfer-v100-4090/build:cuda12.8}"
TOOLS_IMAGE="${TOOLS_IMAGE:-ninfer-v100-4090/tools}"
TARGET="${1:?usage: test.sh <v100|4090|python> [arguments...]}"; shift || true

if [ "$TARGET" = python ]; then
  exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC":/src -w /src "$TOOLS_IMAGE" \
    python -m pytest -q -p no:cacheprovider "${@:-tests}"
fi

case "$TARGET" in
  v100) DIR=build-v100;      NAME=V100 ;;
  4090) DIR=build-ada-volta; NAME=4090 ;;
  *) echo "unknown target '$TARGET' (v100, 4090 or python)"; exit 2 ;;
esac
[ -f "$SRC/$DIR/CTestTestfile.cmake" ] || { echo "no tests in $DIR: build with TESTS=1 v100-4090/scripts/build.sh $TARGET"; exit 2; }
UUID=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v n="$NAME" '$2 ~ n {print $1; exit}')
[ -n "$UUID" ] || { echo "no GPU matching $NAME: the $TARGET suite runs on its own card"; exit 2; }
MOUNTS=(-v "$SRC":/src)
ENVS=()
if [ -n "${NINFER_QWEN3_6_FRONTEND_RESOURCES:-}" ]; then
  MOUNTS+=(-v "$NINFER_QWEN3_6_FRONTEND_RESOURCES":/frontend-resources:ro)
  ENVS+=(-e NINFER_QWEN3_6_FRONTEND_RESOURCES=/frontend-resources)
fi
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --gpus "device=$UUID" "${MOUNTS[@]}" "${ENVS[@]}" "$IMAGE" \
  ctest --test-dir "/src/$DIR" --output-on-failure "$@"
