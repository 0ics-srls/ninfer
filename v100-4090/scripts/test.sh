#!/bin/bash
# test.sh <v100|4090> [ctest arguments...]
#
# Runs the test suite of one rank build (build with: TESTS=1 scripts/build.sh <v100|4090>) inside the build container.
#   GPU unset    no GPU in the container: CPU tests run, GPU tests report "skipped" (exit code 77)
#   GPU=v100     the container sees the Tesla V100 only (run with the v100 build)
#   GPU=4090     the container sees the RTX 4090 only (run with the 4090 build)
# The card is selected by name, never by index. Examples:
#   v100-4090/scripts/test.sh v100                                   # whole suite, CPU only
#   GPU=v100 v100-4090/scripts/test.sh v100 -R softmax_attention     # attention tests on the V100
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
IMAGE="${IMAGE:-ninfer-v100-4090/build:cuda12.8}"
TARGET="${1:?usage: test.sh <v100|4090> [ctest args...]}"; shift || true
case "$TARGET" in
  v100) DIR=build-v100 ;;
  4090) DIR=build-ada-volta ;;
  *) echo "unknown target '$TARGET' (v100 or 4090)"; exit 2 ;;
esac
[ -f "$SRC/$DIR/CTestTestfile.cmake" ] || { echo "no tests in $DIR: build with TESTS=1 v100-4090/scripts/build.sh $TARGET"; exit 2; }
GPUS=""
if [ -n "${GPU:-}" ]; then
  NAME=$([ "$GPU" = v100 ] && echo V100 || echo 4090)
  UUID=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v n="$NAME" '$2 ~ n {print $1; exit}')
  [ -n "$UUID" ] || { echo "no GPU matching $NAME"; exit 2; }
  GPUS="--gpus \"device=$UUID\""
fi
eval docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp $GPUS -v "$SRC":/src "$IMAGE" \
  ctest --test-dir "/src/$DIR" --output-on-failure "$@"
