#!/bin/bash
# start-tp.sh [max_context=262144] [port=8097]
#
# Starts the engine in tensor parallel on a Tesla V100 (rank 0, build-v100) + an RTX 4090 (rank 1, build-ada-volta):
# one container sees both cards (selected by UUID, never by index), runs one ninfer-serve per card and exposes a single
# OpenAI-compatible endpoint through tools/tp2/tp2_proxy.py, which keeps the two ranks in lockstep.
# Ready in ~75 s. Check: curl -s localhost:8097/health   ->  {"status":"ok"}
#
# Defaults = the configuration all published numbers were measured with:
#   262k context, int8 KV cache with explicit capacity 262144, MTP with 4 drafts and the light draft head, images on
#   (256k vision tokens per request), reasoning of previous turns preserved, 16k reasoning cap, 8-bit wire between the
#   cards during prefill.
#
# Environment:
#   MODEL        rank prefix in MODELS/tp2 (default qwen3_8_27b_q8mlp.mlp7680; Uncensored: qwen3_8_27b_uncensored_q8mlp.mlp7680)
#   MODELS       weights directory (default ~/models/ninfer)
#   SRC          repository with build-v100 / build-ada-volta (default: this repository)
#   LOGS         log directory (default ~/.local/state/ninfer-v100-4090)
#   IMAGE        build image (default ninfer-v100-4090/build:cuda12.8)
#   BIND         address to publish on (default 127.0.0.1; 0.0.0.0 to reach it from other machines)
#   KVCAP        KV cache capacity in tokens (default 262144; "auto" leaves 1 GiB free but does not fit with images)
#   EXTRA        extra ninfer-serve options (replaces the default --vision ... block below)
#   FOREGROUND=1 stay attached to the container (for llama-swap: it stops the model with docker stop)
#   Engine switches (all default on, set to 0 for A/B): NINFER_TP_WIRE8, NINFER_SM70_W8_KB2, NINFER_TP_PREFILL_ADD,
#   NINFER_W8_ADA_MMA, NINFER_W8_CUTLASS, NINFER_SM70_ATTN_V2D; NINFER_VISION_ATTN (2 tensor cores, 1 tiled, 0 scalar).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CTX="${1:-262144}"; PORT="${2:-8097}"
MODEL="${MODEL:-qwen3_8_27b_q8mlp.mlp7680}"
MODELS="${MODELS:-$HOME/models/ninfer}"
SRC="${SRC:-$(cd "$HERE/../.." && pwd)}"
LOGS="${LOGS:-$HOME/.local/state/ninfer-v100-4090}"
IMAGE="${IMAGE:-ninfer-v100-4090/build:cuda12.8}"
NAME=ninfer-v100-4090
EXTRA="${EXTRA:---vision --preserve-thinking --default-thinking-budget 16384 --media-live-mib 4096 --media-cache-mib 4096}"
mkdir -p "$LOGS"

V100=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/V100/{print $1; exit}')
ADA=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/4090/{print $1; exit}')
[ -n "$V100" ] && [ -n "$ADA" ] || { echo "need one Tesla V100 and one RTX 4090; nvidia-smi sees:"; nvidia-smi -L; exit 2; }
for f in "$MODELS/tp2/$MODEL.rank0.ninfer" "$MODELS/tp2/$MODEL.rank1.ninfer" \
         "$SRC/build-v100/apps/ninfer-serve" "$SRC/build-ada-volta/apps/ninfer-serve"; do
  [ -e "$f" ] || { echo "missing $f (run build.sh / prepare-weights.sh first)"; exit 2; }
done

free() { for u in "$V100" "$ADA"; do
  [ "$(nvidia-smi --id="$u" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || return 1; done; }
if [ -n "${FOREGROUND:-}" ]; then
  T0=$(date +%s); until free; do (( $(date +%s) - T0 > 120 )) && { echo "GPUs still busy after 120 s"; exit 3; }; sleep 2; done
else
  free || { echo "GPUs are busy (another model loaded?); stop it first"; exit 3; }
fi

[ -f "$LOGS/.tp-key" ] || (umask 077; head -c 24 /dev/urandom | base64 > "$LOGS/.tp-key")   # internal proxy<->rank key
LOG="tp-proxy-$(date +%m%d-%H%M).log"
# Fixed seed: at temperature > 0 the two ranks must sample the same token or the lockstep stops them.
SERVE="--model-id ninfer-27b --max-context $CTX --kv-capacity ${KVCAP:-262144} --max-concurrency 1 --prefill-chunk 2048
       --kv-dtype int8 --spec mtp --draft-tokens 4 --lm-head-draft --seed 42 $EXTRA"
SERVE=$(echo $SERVE)
[ -n "${FOREGROUND:-}" ] && DETACH= || DETACH=-d
ENVS=""
for v in NINFER_TP_WIRE8 NINFER_SM70_W8_KB2 NINFER_TP_PREFILL_ADD NINFER_W8_ADA_MMA NINFER_W8_CUTLASS NINFER_SM70_ATTN_V2D \
         NINFER_VISION_ATTN NINFER_TP_STATS NINFER_TP_DUMP_REQ; do
  [ -n "${!v:-}" ] && ENVS="$ENVS -e $v=${!v}"
done

# Rank coordination files live on a tmpfs inside the container: under /dev/shm, systemd (RemoveIPC) deletes them when
# the last session of the user closes, and every request longer than 180 s gets killed.
docker run $DETACH --rm --name "$NAME" --gpus "\"device=$V100,$ADA\"" --ipc=host --tmpfs /tp:rw,mode=1777 \
  --user "$(id -u):$(id -g)" \
  -e NINFER_TP_WIRE8="${NINFER_TP_WIRE8:-1}" -e NINFER_MAX_PROMPT_VISION_TOKENS="${NINFER_MAX_PROMPT_VISION_TOKENS:-262144}" $ENVS \
  -p "${BIND:-127.0.0.1}:$PORT:8080" \
  -v "$MODELS/tp2":/models:ro -v "$SRC":/src:ro -v "$SRC/build-v100":/v100:ro -v "$SRC/build-ada-volta":/ada:ro -v "$LOGS":/log \
  "$IMAGE" bash -c "exec python3 -u /src/tools/tp2/tp2_proxy.py --listen 0.0.0.0:8080 \
    --binary /v100/apps/ninfer-serve --binary-rank1 /ada/apps/ninfer-serve \
    --model-prefix /models/$MODEL --gpus $V100,$ADA --api-key-file /log/.tp-key --inject-key --log-dir /log \
    --lockstep-file /tp/lockstep --id-file /tp/tp.id -- $SERVE >> /log/$LOG 2>&1" >/dev/null
[ -n "${FOREGROUND:-}" ] && exit 0

T0=$(date +%s)
until curl -sf -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; do
  docker container inspect "$NAME" >/dev/null 2>&1 || { echo "the engine exited:"; tail -20 "$LOGS/$LOG"; tail -5 "$LOGS"/tp2_rank*.log; exit 1; }
  (( $(date +%s) - T0 > 900 )) && { echo "not ready after 15 min"; tail -20 "$LOGS/$LOG"; exit 1; }
  sleep 5
done
echo "ready in $(( $(date +%s) - T0 )) s on :$PORT · $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tr '\n' ' ') · logs $LOGS"
