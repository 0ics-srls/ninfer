#!/bin/bash
# gpu-powerlimit.sh — sets the power limit of each GPU BY NAME, never by index.
# nvidia-smi indices change when you move a card to another slot; a service with fixed indices would put the watts on
# the wrong card.
#
# Measured on our machine:
#   V100 -> 150 W : the V100 is passively cooled; at 150 W it keeps ~86% of its FP16 throughput. 150 -> 250 W gave only
#                   +4% prefill and +0.5% generation on this engine, for +6 C.
#   4090 -> 400 W : the curve is flat at the top; 450 -> 400 W costs ~3% and runs much cooler.
#
# Install:  sudo install -m 755 gpu-powerlimit.sh /usr/local/bin/ && sudo install -m 644 nvidia-powerlimit.service /etc/systemd/system/
#           sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-powerlimit.service
set -u
declare -A LIMITS=( ["V100"]=150 ["4090"]=400 )

nvidia-smi -pm 1 >/dev/null 2>&1
nvidia-smi --query-gpu=index,name --format=csv,noheader | while IFS=, read -r idx name; do
  idx="${idx// /}"
  for key in "${!LIMITS[@]}"; do
    case "$name" in
      *"$key"*)
        watt="${LIMITS[$key]}"
        max=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits -i "$idx" 2>/dev/null | cut -d. -f1)
        [ -n "${max:-}" ] && [ "$watt" -gt "$max" ] && watt="$max"
        if nvidia-smi -i "$idx" -pl "$watt" >/dev/null 2>&1; then
          logger -t gpu-powerlimit "GPU $idx ($name) -> ${watt} W"
        else
          logger -t gpu-powerlimit "GPU $idx ($name): setting ${watt} W FAILED"
        fi ;;
    esac
  done
done
