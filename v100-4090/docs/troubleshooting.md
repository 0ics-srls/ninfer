# Troubleshooting

Run `v100-4090/scripts/check-host.sh` first: it checks driver, GPUs, BAR1, power limits, Docker GPU access, images,
builds and weights, and says which line is wrong.

## The machine does not boot with the V100 installed (no POST, no error code)
Above 4G Decoding is off — typically after a clear CMOS. Remove the V100, enable Above 4G Decoding and Re-Size BAR,
disable CSM, then put the card back.

## The V100 is not listed by `nvidia-smi`
- `sudo dmesg | grep -iE "BAR|nvidia"`: "no space for BAR" / "failed to assign" means the motherboard cannot map the
  32 GB BAR1. No setting fixes it on such a board; use a board that maps it (we use an ASRock X570 Taichi).
- Driver 590 or newer: Volta support ended with branch 580. `sudo apt install nvidia-driver-580-server` and pin it.
- Power: check the V100's 8-pin CPU/EPS-style power input and its adapter.

## `docker run --gpus all ...` fails
Install `nvidia-container-toolkit`, run `sudo nvidia-ctk runtime configure --runtime=docker`, restart Docker.

## The first request dies with "named symbol not found" in NCCL
NCCL without sm_70 kernels. Use the build image (`Dockerfile.build`), which ships NCCL 2.21.5.

## `start-tp.sh` says the GPUs are busy
Another model holds VRAM (for example a llama-swap entry). Stop it, or start with `FOREGROUND=1`, which waits up to
120 s for the GPUs to be released.

## The engine exits while loading
- `tail ~/.local/state/ninfer-v100-4090/tp2_rank*.log`.
- The rank artifacts and the builds must use the same MLP split: `build.sh` with `MLP_V100=7680` (default) and
  `prepare-weights.sh` with the same `MLP_V100`.
- Out of memory on the 4090: keep `KVCAP=262144` with images on; `--kv-capacity auto` reserves 1 GiB of headroom and does
  not fit together with the vision encoder.

## Requests longer than ~3 minutes are killed
The rank coordination files were deleted. `start-tp.sh` keeps them on a tmpfs inside the container (`/tp`); if you start
the engine another way, do not put them in `/dev/shm` (systemd `RemoveIPC` removes them when your last session closes).

## "media budget exceeded"
The conversation holds more images than the per-request vision budget. `start-tp.sh` sets
`NINFER_MAX_PROMPT_VISION_TOKENS=262144` and `--media-live-mib 4096`; with a smaller `--media-live-mib` the engine
refuses to start with that budget.

## Images take 30–40 s each
The vision attention is running the original scalar kernel: check that `NINFER_VISION_ATTN` is not set to 0 and that
the binaries were built from this branch. Expected: ~9–10 s per ~3,000-token image.

## The V100 is at 80+ °C and slows down
The airflow does not reach its fins. Check `journalctl -t gpu-fan` (the fan header must be the one you configured in
`gpu-fan-control.sh`) and the position of the fan or shroud.

## Generation is much slower than in the README
- Is MTP on (`--spec mtp --draft-tokens 4 --lm-head-draft`)? Without it generation roughly halves.
- Temperature and task matter: code accepts more drafts than prose. Compare with `bench/gen-speed.py` (~138 t/s).
- Another process on the GPUs (`nvidia-smi`).
- First measurement after startup: discard it.

## OpenCode stops at 32k output tokens or cuts long answers
`export OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=65536` in the shell that starts OpenCode.

## A run of the benchmark plan took much longer than another
Look at the reasoning tokens (`bench/session-times.py`): with the same engine, two runs of the same plan produced 165k
and 458k reasoning tokens. Pick the reasoning level with the OpenCode variants (`ctrl+t`), and compare engines by
replaying the same requests, not with single runs.
