# NInfer on a Tesla V100 + RTX 4090

**Qwen3.8-27B at 8-bit, ~100 tokens/s, 262k context, on two used GPUs working together in tensor parallel.**

This branch (`v100-4090`) of the [0ics-srls/ninfer](https://github.com/0ics-srls/ninfer) fork runs a dense
Qwen3.8-27B across a **Tesla V100 32 GB (Volta, 2017)** and a **GeForce RTX 4090 24 GB (Ada, 2022)**. Both cards
compute every token; each one runs kernels written for its own architecture. Everything you need to rebuild it is here:
the hardware, the BIOS, the OS, the build, the weights, the server, the agent setup, the benchmarks and the traps.

The guide is written so that a person **or a coding agent** can follow it top to bottom. Every step has a **Do** block
and a **Check** block with the output to expect; `v100-4090/scripts/check-host.sh` re-checks the whole host at any point.

| measured on real agentic coding sessions | |
|---|---|
| generation, up to 100k context | **100–106 tokens/s** |
| generation, past 160k context | **~91 tokens/s** |
| context | **262,144** tokens, int8 KV cache |
| cold prefill | **~1,160 tokens/s** (55k) |
| first token on a short follow-up turn at 120–200k | median **0.37 s** |
| image input (~3,000 vision tokens) | **~9 s** |
| code generation (MTP accepts more drafts on code) | 138–305 tokens/s |
| VRAM in use | ~23.3 GB per card |

A visual summary with comparisons against other home setups is in [`v100-4090/docs/turbo-rig.html`](v100-4090/docs/turbo-rig.html)
(download it and open it in a browser). Full numbers: [`v100-4090/docs/benchmarks.md`](v100-4090/docs/benchmarks.md).

---

## Contents

0. [How it works, in one page](#0-how-it-works-in-one-page)
1. [Hardware](#1-hardware)
2. [BIOS](#2-bios)
3. [Operating system and NVIDIA driver](#3-operating-system-and-nvidia-driver)
4. [Power limits and fans](#4-power-limits-and-fans)
5. [Docker and the build images](#5-docker-and-the-build-images)
6. [Build the engine](#6-build-the-engine)
7. [Prepare the weights](#7-prepare-the-weights)
8. [Start the server and verify it](#8-start-the-server-and-verify-it)
9. [Use it from a coding agent (llama-swap + OpenCode)](#9-use-it-from-a-coding-agent-llama-swap--opencode)
10. [Run the real-work benchmark](#10-run-the-real-work-benchmark)
11. [What we changed in the engine](#11-what-we-changed-in-the-engine)
12. [Traps](#12-traps)
13. [Credits and license](#13-credits-and-license)

---

## 0. How it works, in one page

- **The two cards are complementary.** The V100 has 32 GB of HBM2 at ~900 GB/s but no BF16, no FP8 and no INT8 on its
  tensor cores (FP16 only). The 4090 has ~2.2× the FP16 compute (measured: 149 vs 66.9 TFLOPS) and real FP8, but only
  24 GB. Together: 56 GB at almost equal bandwidth.
- **Tensor parallel, not layer split.** Every layer is split in two halves and both cards work on every token; after
  each row-parallel matrix the two partial sums are added across the PCIe bus (through host memory: no NVLink, no P2P).
- **The split follows compute, not memory.** The MLP gives 7,680 of its 17,408 columns to the V100 and 9,728 to the 4090,
  so the faster card does not wait for the slower one.
- **8-bit weights.** The MLP is the Q8_0 of the official GGUF copied bit for bit (one scale per 32 values); attention
  and Gated DeltaNet are the official FP8 tensors. Below 8 bits the model starts getting exact strings wrong in code.
- **Each card on its own code path.** The V100 runs kernels written for Volta (8-bit GEMV, int8-KV attention, flash
  attention on FP16 tensor cores); the 4090 runs the same layouts with Ada kernels where they pay off.
- **MTP speculative decoding** (4 drafts, light draft head) is the biggest single lever: each step verifies 2–3 tokens.
- **One endpoint.** A small proxy (`tools/tp2/tp2_proxy.py`) starts one `ninfer-serve` per card, keeps them in
  lockstep and exposes an OpenAI-compatible API (also Anthropic-compatible), with prefix cache, tool calls, reasoning
  and images.

---

## 1. Hardware

### 1.1 Parts

| part | what we use | requirement and why |
|---|---|---|
| GPU 0 | **Tesla V100 PCIe 32 GB** (used, from ~$670) | must be the **32 GB** model (the SXM2 version on an SXM2→PCIe adapter board should work too; we have not tested it) |
| GPU 1 | **GeForce RTX 4090 24 GB** (used, ~$2,800) | Ada, sm_89 |
| motherboard | ASRock X570 Taichi (BIOS P4.70) | **must map the V100's 32 GB BAR1** (see 1.2) and give each card at least PCIe 3.0 x8 |
| CPU | Ryzen 5 5600X | anything recent with enough PCIe lanes; all the work is on the GPUs |
| RAM | 64 GB DDR4 | 32 GB is enough for this model |
| PSU | 1,000 W | V100 at 150 W + 4090 at 400 W + the rest, with headroom |
| storage | ~1 TB SSD | ~85 GB per model during preparation (downloads + ranks) |
| cooling | a fan blowing straight into the V100 (shroud or bracket), driven by the GPU temperature (section 4) | the V100 is **passive**: without forced airflow it reaches 83 °C and throttles |

### 1.2 The motherboard must map a 32 GB BAR

The V100's BAR1 is fixed at 32 GB and needs a 64-bit prefetchable memory window that large. Many consumer boards do not
offer it. **We could not make the V100 work at all on a Gigabyte X670E AORUS MASTER** (the firmware offers at most
0.25 GB of window: tried CPU slot, 4090 removed, CSM off, Above 4G and ReBAR on, latest BIOS). On the **ASRock X570 Taichi**
the full 32 GB window is assigned. The 4090 has a resizable BAR (256 MB to 24 GB) and always fits in what is left.

**Check** (after section 3, with the driver installed):

```bash
nvidia-smi -q -d MEMORY | grep -A3 "BAR1"          # the V100 must show: Total : 32768 MiB
sudo dmesg | grep -iE "BAR 1|no space|failed to assign"   # no "no space" / "failed to assign" for the V100
```

### 1.3 Mounting

- Put each card in a slot wired to at least **PCIe 3.0 x8** (we run x8/x8). There is no P2P between the cards; that's fine.
- **V100 power connector:** the Tesla V100 PCIe uses an **8-pin CPU/EPS-style** power input, not a PCIe 8-pin. Check
  your card and use the proper adapter (two PCIe 8-pin to one EPS 8-pin) — never force a PCIe plug into it.
- **V100 airflow:** mount a fan (or a 3D-printed shroud with a 40–80 mm blower) that pushes air *through* the V100's fins,
  connected to a motherboard fan header you can control from Linux (section 4).
- Leave space between the cards: the 4090 dumps a lot of heat into the case.

---

## 2. BIOS

**Do** — before installing the V100:

1. **Above 4G Decoding: Enabled.** Without it the V100 stops the machine from booting, with no error code.
2. **Re-Size BAR Support: Enabled.**
3. **CSM: Disabled** (UEFI boot only).
4. Optional but simpler: **Secure Boot: Disabled** (otherwise you must enroll the key that signs the NVIDIA DKMS module).

> ⚠️ A **clear CMOS** resets all of the above. After a reset the V100 again prevents the machine from booting: set the
> options again *before* putting the card back in. On the X570 Taichi, a Dr. Debug code 66 after changing RAM or CPU
> means an old memory profile: clear CMOS, then redo this list.

**Check:** the machine boots with both cards installed and `lspci | grep -i nvidia` shows two devices.

---

## 3. Operating system and NVIDIA driver

Ubuntu Server 24.04 LTS. The host only needs the driver, Docker and the NVIDIA container toolkit: CUDA and every build
tool live in containers.

Two constraints you don't get to choose — the V100 is at the end of NVIDIA's support:

- **Driver branch 580, not the latest.** 580 is the last branch with Volta support (LTS until 2028); from 590 on the
  V100 disappears. It covers the 4090 fine.
- **CUDA 12.x, not 13.** CUDA 13 removed sm_70. The build image uses CUDA 12.8.

**Do:**

```bash
sudo apt update && sudo apt install -y nvidia-driver-580-server
sudo apt-mark hold nvidia-driver-580-server     # an upgrade must never replace it
sudo reboot
```

**Check:**

```bash
nvidia-smi                       # two GPUs, Driver Version: 580.x, CUDA Version: 13.0 (driver capability, fine)
nvidia-smi -q -d MEMORY | grep -A3 BAR1     # V100: Total : 32768 MiB (section 1.2)
```

> ⚠️ **GPU indices differ between tools.** `nvidia-smi` orders by PCI bus (on our board: 0 = V100, 1 = 4090); CUDA
> orders by speed (CUDA0 = 4090, CUDA1 = V100). Everything in this repository selects cards **by name or UUID**,
> never by index.

---

## 4. Power limits and fans

**Do** — power limits by name (V100 150 W, 4090 400 W):

```bash
cd v100-4090/host
sudo install -m 755 gpu-powerlimit.sh /usr/local/bin/
sudo install -m 644 nvidia-powerlimit.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-powerlimit.service
```

Why 150 W on the V100: it is passively cooled, and on this engine 150 → 250 W gave only +4% prefill and +0.5% generation
for 6 °C more. 450 → 400 W on the 4090 costs ~3% and runs much cooler.

**Do** — fans driven by the GPU temperature. **Edit the three settings at the top of `gpu-fan-control.sh` first**
(`CHIP`, `PWM_V100`, `PWM_CASE`): they depend on your motherboard. The script explains how to find the right fan header.
Never write to the PWM channels of the CPU fans.

```bash
sudo modprobe nct6775                               # sensor driver for Nuvoton chips (most ASRock/ASUS boards)
cat /sys/class/hwmon/hwmon*/name                    # find your chip name, e.g. nct6779
sudo install -m 755 gpu-fan-control.sh /usr/local/bin/
sudo install -m 644 gpu-fan.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-fan.service
```

**Check:**

```bash
nvidia-smi --query-gpu=name,power.limit,temperature.gpu --format=csv   # V100 150 W, 4090 400 W
journalctl -t gpu-fan -n 5                                            # "V100 xxC -> pwm yyy"
```

Under load the V100 should stay below ~70 °C. If it goes past 75 °C, the airflow is not reaching its fins.

---

## 5. Docker and the build images

**Do** — Docker and the NVIDIA container toolkit:

```bash
sudo apt install -y docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker "$USER"     # then log out and back in
```

**Do** — clone this branch with the benchmark submodule and build the two images:

```bash
git clone --recurse-submodules -b v100-4090 https://github.com/0ics-srls/ninfer.git
cd ninfer
docker build -f v100-4090/docker/Dockerfile.build -t ninfer-v100-4090/build:cuda12.8 v100-4090/docker
docker build -f v100-4090/docker/Dockerfile.tools -t ninfer-v100-4090/tools v100-4090/docker
```

The build image is CUDA 12.8 (the last line that compiles sm_70) with NCCL 2.21.5, which still has Volta kernels — the
NCCL in Ubuntu 24.04 does not ("named symbol not found" at the first all-reduce).

**Check:**

```bash
v100-4090/scripts/check-host.sh     # every line OK, except "not built yet" / "not prepared yet" warnings
```

---

## 6. Build the engine

**Do:**

```bash
v100-4090/scripts/build.sh both ninfer-serve
```

This builds two binaries from the same sources:

| directory | card | target | MLP columns |
|---|---|---|---|
| `build-v100` | V100 (rank 0) | sm_70 | 7,680 |
| `build-ada-volta` | RTX 4090 (rank 1) | sm_89 on the Volta code path (same layouts as rank 0, plus Ada kernels) | 9,728 |

The MLP width of each rank is a **compile-time constant** and must match the weights of section 7. A first build takes
~25–40 minutes per rank on a 6-core CPU; later rebuilds use ccache (`~/.cache/ninfer-ccache`) and take seconds.

**Check:**

```bash
ls -la build-v100/apps/ninfer-serve build-ada-volta/apps/ninfer-serve    # both exist
```

---

## 7. Prepare the weights

Nothing is retrained. The engine loads a repackaging of the official weights, split in two:

| source (Hugging Face) | file | used for |
|---|---|---|
| [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_nvfp4.ninfer` (23.7 GB) | official Qwen3.8-27B as a NInfer artifact: attention and Gated DeltaNet in the official FP8, embeddings, heads, vision encoder, MTP |
| [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-Q8_0.gguf` (28.6 GB) | the text MLP: its Q8_0 blocks are copied bit for bit, replacing the NVFP4 MLP of the artifact |
| optional: [`bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF`](https://huggingface.co/bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF) | `orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf` | the Uncensored variant (`--uncensored`): only the matrices the abliteration changed are rewritten |

All three are Apache-2.0.

**Do:**

```bash
v100-4090/scripts/prepare-weights.sh                 # official model
v100-4090/scripts/prepare-weights.sh --uncensored    # also the Uncensored variant (optional)
```

What it does (`MODELS`, default `~/models/ninfer`):
1. downloads the files into `MODELS/download` (once);
2. runs `tools/tp2/shard_qwen38_27b.py --mlp-rank0 7680 --mlp-q8 <Q8_0 GGUF> --drop-dflash2`, which splits every head and
   intermediate dimension on exact stored words (the two halves re-concatenate bit-exactly; add `--verify` to check);
3. with `--uncensored`: `tools/tp2/patch_from_gguf.py --check` (the GGUF must line up with the base: a few % difference
   per tensor, not ~1.4), then the patch, then the same sharding with the Uncensored GGUF.

Time: the downloads, then ~15–20 minutes of CPU per model. Disk: ~53 GB of downloads (deletable afterwards) + ~30 GB of
ranks per model.

**Check:**

```bash
ls -la ~/models/ninfer/tp2/
# qwen3_8_27b_q8mlp.mlp7680.rank0.ninfer   15,103,820,544 bytes
# qwen3_8_27b_q8mlp.mlp7680.rank1.ninfer   17,276,338,944 bytes
```

We rebuilt the ranks from scratch with this script and compared them with the files we run in production: they are
identical except for bytes 17–32 of the header, a random `artifact_id` (`uuid4`) written every time a file is created.
To compare two builds yourself: `cmp -l a.ninfer b.ninfer | wc -l` must print 16 or less, all at offsets 17–32.

---

## 8. Start the server and verify it

**Do:**

```bash
v100-4090/scripts/start-tp.sh                       # official model, 262k context, port 8097
# or: MODEL=qwen3_8_27b_uncensored_q8mlp.mlp7680 v100-4090/scripts/start-tp.sh
```

It refuses to start if a GPU is busy, waits until both ranks are ready (~75 s) and prints the VRAM in use. The defaults
are exactly the configuration all the numbers in this README were measured with:

```
--max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 1 --prefill-chunk 2048
--spec mtp --draft-tokens 4 --lm-head-draft --seed 42
--vision --preserve-thinking --default-thinking-budget 16384 --media-live-mib 4096 --media-cache-mib 4096
NINFER_TP_WIRE8=1  NINFER_MAX_PROMPT_VISION_TOKENS=262144
```

`v100-4090/scripts/stop.sh` stops it. Logs: `~/.local/state/ninfer-v100-4090/`.

**Check** — each command prints what to expect:

```bash
curl -s localhost:8097/health                                   # {"status":"ok"}
python3 v100-4090/bench/gen-speed.py                           # ~138 t/s, same text hash on every repeat
python3 v100-4090/bench/needle.py http://127.0.0.1:8097 120000 # 3/3, cold prefill ~1,000+ t/s
ffmpeg -f lavfi -i "mandelbrot=s=2048x1536" -frames:v 1 /tmp/test.jpg
python3 v100-4090/bench/image.py /tmp/test.jpg                 # ~9-10 s, ~3,100 prompt tokens
```

If a number is far off, see [section 12](#12-traps) and [`v100-4090/docs/troubleshooting.md`](v100-4090/docs/troubleshooting.md).

---

## 9. Use it from a coding agent (llama-swap + OpenCode)

**llama-swap** puts one OpenAI-compatible port (`:8080`) in front of all your models and starts the one the client asks
for. Example config: [`v100-4090/config/llama-swap.yaml`](v100-4090/config/llama-swap.yaml) (replace `/path/to/ninfer`).

**OpenCode** example: [`v100-4090/config/opencode.json`](v100-4090/config/opencode.json). Two things matter:

- **Reasoning level from the UI.** The model has the variants `xhigh`, `medium`, `low` and `none`, sent to the engine as
  `reasoning_effort`; switch them with `ctrl+t`. `xhigh` (the engine default) adds "think carefully, validate key
  assumptions…" to the system prompt — good for planning; `medium`/`low` think less.
- **64k output tokens.** OpenCode caps the output at 32k unless you set
  `export OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=65536` in the shell that starts it.

Keep context compaction on (`"compaction": { "auto": true }`).

---

## 10. Run the real-work benchmark

[`v100-4090/test-project`](v100-4090/test-project) is the
[`0ics-srls/ninfer-code-test`](https://github.com/0ics-srls/ninfer-code-test) submodule: a full-stack todo app
(.NET 10 + Angular 21) and a 6-block TDD plan that OpenCode executes on its own through the CVM plan executor — migrations,
API changes, regenerated client, UI, Playwright e2e. Its README explains how to run it and lists our four runs:
6/6 blocks every time, e2e 25/25 twice, code review 20–21/25 from three independent reviewers, 1 h 23 – 2 h 28 of
active work.

---

## 11. What we changed in the engine

All changes are behind environment switches (default on) so each one can be A/B-tested; details, measurements and
commits in [`v100-4090/docs/engine-changes.md`](v100-4090/docs/engine-changes.md).

| change | effect (measured) |
|---|---|
| tensor parallel V100 + 4090 with per-rank MLP width (7,680 / 9,728) | the 4090 no longer waits for the V100 |
| MLP from the Q8_0 GGUF bit for bit, 8-bit GEMV for the V100 (660–675 GB/s, 77% of peak) | 8-bit quality at V100 bandwidth |
| int8 KV cache + our own V100 attention kernel (2 CTAs/SM, 470–480 GB/s) | ~92 t/s at 261k |
| MTP with 4 drafts and a light draft head | code 87 → 305 t/s |
| prefill on tensor cores (CUTLASS 8-bit MLP, flash attention for Volta) | 1,030 t/s at 65k (llama.cpp: 560–730) |
| short reads (17–63 new tokens) on the fast kernel | first token 2–6 s → 0.3 s |
| proxy closes the response at once (was on a 2 s grid) | −2 s per turn |
| 8-bit wire between the cards during prefill (`NINFER_TP_WIRE8`) | prefill +10% (1,056 → 1,163 t/s at 55k) |
| V100 GEMV at smaller K blocks for T ≤ 8 (`NINFER_SM70_W8_KB2`) | generation +4.6% |
| per-request image budget 32k → 256k vision tokens (`NINFER_MAX_PROMPT_VISION_TOKENS`) | no "media budget exceeded" at the 11th photo |
| vision-encoder attention: scalar → tiled → FP16 tensor cores (`NINFER_VISION_ATTN`) | 28.5 → 9.0 s per photo |

---

## 12. Traps

The ones that cost us hours (more in [`v100-4090/docs/troubleshooting.md`](v100-4090/docs/troubleshooting.md)):

- **The motherboard can't map the V100's BAR** → the card never appears. No setting fixes it; change the board (1.2).
- **Clear CMOS turns off Above 4G** → the V100 stops the machine from booting.
- **Driver ≥ 590 or CUDA 13** → no V100.
- **BF16 on the V100** silently runs emulated (~7× slower) in PyTorch-style code. This engine never uses BF16 math on Volta.
- **FP8, NVFP4 and AWQ on the V100 buy memory, not speed**: it unpacks them to FP16 for every multiplication.
- **GPU indices differ** between `nvidia-smi` and CUDA: select cards by name or UUID.
- **Rank coordination files in `/dev/shm`** are deleted by systemd (`RemoveIPC`) when your last session closes, and
  every request longer than 180 s dies: `start-tp.sh` keeps them on a tmpfs inside the container.
- **Reasoning length varies 3× between runs of the same task.** To compare engines, replay the same requests; don't
  compare two single runs.
- **The date in OpenCode's system prompt changes at midnight** and invalidates the prefix cache: the whole context is
  read again once (147k tokens in 3 min 20 s).
- **Measure, don't guess.** More than once a "predicted" 3% was really 25%, or the other way around.

---

## 13. Credits and license

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the NInfer engine (RTX 5090).
- [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) — the V100 port.
- [huangserva/ninfer-v100-tpx](https://github.com/huangserva/ninfer-v100-tpx) — sm70 attention kernels and tensor
  parallel across V100s; its README is kept in [`README-upstream.md`](README-upstream.md).
- memoriaru — the sm_89 port.
- This branch — tensor parallel across V100 + RTX 4090, 8-bit MLP, the kernels and switches in section 11, the scripts
  and this guide.
- Weights: [Qwen](https://huggingface.co/Qwen) (Qwen3.8-27B), neroued (NInfer artifact), ggml-org (GGUF),
  orcarouter (Uncensored) and bartowski (its GGUF). All Apache-2.0.

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
