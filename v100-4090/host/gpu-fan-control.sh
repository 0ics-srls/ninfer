#!/bin/bash
# gpu-fan-control.sh — drives case fans from the GPU temperatures. The motherboard only looks at the CPU temperature and
# ignores the V100, which is a PASSIVE card: without forced airflow it reaches 83 C and throttles. With a dedicated fan
# on a shroud in front of the V100 and this curve it stays around 64-69 C under load.
#
# YOU MUST ADAPT THE THREE SETTINGS BELOW to your motherboard. Never write to the PWM channels of the CPU fans.
#   CHIP      name of the hwmon chip with the fan headers (cat /sys/class/hwmon/hwmon*/name; nct6779 on ASRock X570 Taichi)
#   PWM_V100  header of the fan blowing into the V100 (curve on the V100 temperature)
#   PWM_CASE  header of a case fan (gentle curve on the hottest GPU)
# To find a header: set one pwmN to 255 at a time (echo 1 > pwmN_enable; echo 255 > pwmN) and watch which fanN_input
# rises; put it back to automatic with echo 5 > pwmN_enable (value 5 = "smart fan" on nct67xx chips).
#
# Install:  sudo install -m 755 gpu-fan-control.sh /usr/local/bin/ && sudo install -m 644 gpu-fan.service /etc/systemd/system/
#           sudo systemctl daemon-reload && sudo systemctl enable --now gpu-fan.service
set -u
CHIP="${CHIP:-nct6779}"
PWM_V100="${PWM_V100:-4}"
PWM_CASE="${PWM_CASE:-1}"

HW=""
for d in /sys/class/hwmon/hwmon*; do [ "$(cat "$d/name" 2>/dev/null)" = "$CHIP" ] && HW="$d" && break; done
[ -z "$HW" ] && { echo "hwmon chip $CHIP not found (sudo modprobe nct6775?)" >&2; exit 1; }

pwm_v100() {   # V100 temperature -> PWM (0-255): full speed from 72 C, it throttles at 83 C
  local t=$1
  if   [ "$t" -ge 72 ]; then echo 255
  elif [ "$t" -ge 68 ]; then echo 225
  elif [ "$t" -ge 60 ]; then echo 190
  elif [ "$t" -ge 50 ]; then echo 150
  elif [ "$t" -ge 42 ]; then echo 110
  else                       echo 80
  fi
}
pwm_case() {   # hottest GPU -> PWM, gentler
  local t=$1
  if   [ "$t" -ge 78 ]; then echo 255
  elif [ "$t" -ge 68 ]; then echo 200
  elif [ "$t" -ge 55 ]; then echo 150
  else                       echo 110
  fi
}

for c in $PWM_CASE $PWM_V100; do echo 1 > "$HW/pwm${c}_enable" 2>/dev/null; done
trap "for c in $PWM_CASE $PWM_V100; do echo 5 > $HW/pwm\${c}_enable 2>/dev/null; done; exit 0" EXIT TERM INT

V100_IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F, '/V100/{gsub(/ /,"",$1); print $1; exit}')
[ -z "$V100_IDX" ] && { echo "V100 not found" >&2; exit 1; }
logger -t gpu-fan "V100 at index $V100_IDX, chip $CHIP, pwm${PWM_V100} (V100) pwm${PWM_CASE} (case)"

last_v=-1; last_c=-1
while true; do
  tv=$(timeout 3 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$V100_IDX" 2>/dev/null)
  tmax=$(timeout 3 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  if [ -z "${tv:-}" ] || [ -z "${tmax:-}" ]; then sleep 10 & wait $!; continue; fi
  pv=$(pwm_v100 "$tv"); pc=$(pwm_case "$tmax")
  [ "$pv" != "$last_v" ] && { echo "$pv" > "$HW/pwm$PWM_V100"; last_v=$pv; logger -t gpu-fan "V100 ${tv}C -> pwm $pv"; }
  [ "$pc" != "$last_c" ] && { echo "$pc" > "$HW/pwm$PWM_CASE"; last_c=$pc; logger -t gpu-fan "GPU max ${tmax}C -> case pwm $pc"; }
  sleep 10 & wait $!   # background + wait so that systemd's TERM interrupts the sleep at once
done
