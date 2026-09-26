# deploy/

- `Dockerfile.tp2`: runtime image with `ninfer-serve`, `tp2_proxy.py`, and an NCCL that has sm_70 kernels.
  Put `libnccl.so.2` (from the `nvidia-nccl-cu12==2.21.5` wheel) and `libcudart.so.12` under `deploy/lib/` before building.
- `start_tp2.sh`: starts the container with both GPUs. Rollback is `docker stop <name>` and start whatever you ran before;
  the two services share the port and GPU 0, so never run both at once.

Operational notes from the first deployment (2× V100 32G PCIe, 2026-09-26):

- switching from the single-GPU container interrupted the service for about 24 s; each restart of the pair takes about 25 s;
- with `--max-context 262144 --vision` each GPU uses about 20.4 GB;
- the lockstep check fired once during a 300-request evaluation run; the proxy restarted both ranks in 23 s and the request was retried by the client, no wrong output was served.
