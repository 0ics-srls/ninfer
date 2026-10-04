#!/bin/bash
# check-host.sh — verifies the host step by step and says what is missing. Safe to run any time (read-only).
# Every line starts with OK, WARN or FAIL. Run it after each section of the README.
set -u
ok()   { printf 'OK    %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; FAILED=1; }
FAILED=0

# --- OS
. /etc/os-release 2>/dev/null
[ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = "24.04" ] && ok "Ubuntu $VERSION_ID" || warn "tested on Ubuntu 24.04, found ${PRETTY_NAME:-unknown}"

# --- driver and GPUs
if ! command -v nvidia-smi >/dev/null; then fail "nvidia-smi not found: install nvidia-driver-580-server"; echo; exit 1; fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
case "$DRV" in 580.*) ok "driver $DRV (branch 580: the last one with Volta support)";;
  *) fail "driver $DRV: use branch 580 (590+ dropped the V100)";; esac
if apt-mark showhold 2>/dev/null | grep -q nvidia-driver-580; then ok "driver pinned (apt-mark hold)"; else warn "driver not pinned: sudo apt-mark hold nvidia-driver-580-server"; fi

V100_IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/V100/{print $1; exit}')
ADA_IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/4090/{print $1; exit}')
[ -n "$V100_IDX" ] && ok "Tesla V100 at nvidia-smi index $V100_IDX" || fail "no Tesla V100 visible (BIOS: Above 4G Decoding on? card seated? power cables?)"
[ -n "$ADA_IDX" ] && ok "RTX 4090 at nvidia-smi index $ADA_IDX" || fail "no RTX 4090 visible"

if [ -n "$V100_IDX" ]; then
  MEM=$(nvidia-smi -i "$V100_IDX" --query-gpu=memory.total --format=csv,noheader,nounits)
  [ "${MEM:-0}" -gt 30000 ] && ok "V100 has ${MEM} MiB" || fail "V100 reports ${MEM:-?} MiB: the 32 GB model is required"
  BAR1=$(nvidia-smi -i "$V100_IDX" -q -d MEMORY | awk '/BAR1 Memory Usage/{f=1} f && /Total/{print $3; exit}')
  [ "${BAR1:-0}" -ge 32768 ] && ok "V100 BAR1 = ${BAR1} MiB (full 32 GB window mapped)" \
    || fail "V100 BAR1 = ${BAR1:-?} MiB: the motherboard did not map the 32 GB BAR (see README, Hardware)"
  PL=$(nvidia-smi -i "$V100_IDX" --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  [ "${PL:-0}" -le 150 ] && ok "V100 power limit ${PL} W" || warn "V100 power limit ${PL} W (we run 150 W: host/gpu-powerlimit.sh)"
  T=$(nvidia-smi -i "$V100_IDX" --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  [ "${T:-0}" -lt 75 ] && ok "V100 at ${T} C" || warn "V100 at ${T} C: passive card, needs forced airflow (throttles at 83 C)"
fi
if [ -n "$ADA_IDX" ]; then
  PL=$(nvidia-smi -i "$ADA_IDX" --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  [ "${PL:-0}" -le 400 ] && ok "RTX 4090 power limit ${PL} W" || warn "RTX 4090 power limit ${PL} W (we run 400 W)"
fi

# --- Docker with GPU access
if command -v docker >/dev/null; then
  ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L >/tmp/check-host.$$ 2>&1; then
    ok "containers see $(grep -c '^GPU' /tmp/check-host.$$) GPUs (nvidia-container-toolkit works)"
  else
    fail "docker --gpus all failed: install nvidia-container-toolkit and run sudo nvidia-ctk runtime configure --runtime=docker"
  fi
  rm -f /tmp/check-host.$$
  for img in ninfer-v100-4090/build:cuda12.8 ninfer-v100-4090/tools; do
    docker image inspect "$img" >/dev/null 2>&1 && ok "image $img" || warn "image $img not built yet"
  done
else
  fail "docker not installed"
fi

# --- memory and disk
RAM=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
[ "$RAM" -ge 30 ] && ok "${RAM} GB RAM" || warn "${RAM} GB RAM: 32 GB or more recommended"
FREE=$(df -BG --output=avail "${MODELS:-$HOME}" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 90 ] && ok "${FREE} GB free for the weights" || warn "${FREE:-?} GB free: ~85 GB needed for downloads + ranks of one model"

# --- builds and weights
HERE="$(cd "$(dirname "$0")" && pwd)"; SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
for d in build-v100 build-ada-volta; do
  [ -x "$SRC/$d/apps/ninfer-serve" ] && ok "$d/apps/ninfer-serve built" || warn "$d not built yet (scripts/build.sh)"
done
M="${MODELS:-$HOME/models/ninfer}/tp2/${MODEL:-qwen3_8_27b_q8mlp.mlp7680}"
[ -s "$M.rank0.ninfer" ] && [ -s "$M.rank1.ninfer" ] && ok "weights $(basename "$M").rank{0,1}" || warn "weights not prepared yet (scripts/prepare-weights.sh)"

echo; [ "$FAILED" = 0 ] && echo "no blocking problem found" || { echo "fix the FAIL lines first"; exit 1; }
