#!/bin/bash
# mutate-py.sh — RED proof for the Python tests of our tools: each mutation must make its tests fail.
# Runs on a throwaway copy of tools/ + tests/ in /tmp/mut-py (never on the working tree), with the tools image
# (docker build -f v100-4090/docker/Dockerfile.tools -t ninfer-v100-4090/tools v100-4090/docker). Last run 2026-10-05: 13/13 RED.
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
M=/tmp/mut-py
run() {  # run <label> <file> <python-replace-old> <python-replace-new> <pytest args...>
  local label=$1 file=$2 old=$3 new=$4; shift 4
  rm -rf $M; mkdir -p $M; cp -r $SRC/tools $SRC/tests $M/
  python3 - "$M/$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:4]
t = open(p).read()
assert old in t, ("mutation anchor not found", old)
open(p, "w").write(t.replace(old, new, 1))
PY
  out=$(docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $M:/src -w /src ninfer-v100-4090/tools \
        python -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -1)
  case "$out" in *failed*) echo "RED   $label: $out" ;; *) echo "GREEN!! $label: $out (mutation not caught)" ;; esac
}
run "gdn value heads not regrouped" tools/tp2/patch_from_gguf.py \
  "order = [(h % GDN_V_PER_K) * GDN_K_HEADS + h // GDN_V_PER_K for h in range(heads)]" \
  "order = list(range(heads))" tests/tp2/test_patch_from_gguf.py
run "patch writes at wrong offset" tools/tp2/patch_from_gguf.py \
  "f.seek(art.payload_offset + obj.offset)" "f.seek(obj.offset)" tests/tp2/test_patch_from_gguf.py
run "uneven cut ignored" tools/tp2/shard_qwen38_27b.py \
  "return (0, self.cut) if rank == 0 else (self.cut, seg - self.cut)" \
  "return rank * (seg // TP), seg // TP" tests/tp2/test_shard_qwen38_27b.py
run "q8 down takes wrong groups" tools/tp2/shard_qwen38_27b.py \
  "codes = codes[:, begin // 32:(begin + length) // 32].contiguous()" \
  "codes = codes[:, :length // 32].contiguous()" tests/tp2/test_shard_qwen38_27b.py
run "w8 K split scales not split" tools/tp2/shard_qwen38_27b.py \
  "            scales = take_frac(scales, 1, rule, rank, 1, 32)" "            pass" tests/tp2/test_shard_qwen38_27b.py
run "v3 mtp rename lost" tools/artifact/v3.py \
  'logical = "mtp/layer/" + logical[len("mtp/layers/0/"):]' "pass" tests/artifact/test_v3.py
run "v3 tp suffix lost" tools/artifact/v3.py 'weights_id += f"-tp{ranks}"' "pass" tests/artifact/test_v3.py
run "v3 ranges not scaled" tools/artifact/v3.py \
  "begin, end = begin * num // den, end * num // den" "pass" tests/artifact/test_v3.py
run "proxy seed per rank" tools/tp2/tp2_proxy.py \
  'req["seed"] = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF' \
  'return body' tests/tp2/test_tp2_proxy.py
run "proxy watchdog sleeps" tools/tp2/tp2_proxy.py "if finished.wait(2):" "if time.sleep(2):" tests/tp2/test_tp2_proxy.py
run "proxy maps by name" tools/tp2/tp2_proxy.py "if self.units_map is None or self.units_map[0] != gen:" \
  "if True:" tests/tp2/test_tp2_proxy.py
run "proxy keeps client key" tools/tp2/tp2_proxy.py "        if self.sup.args.inject_key:" "        if False:" \
  tests/tp2/test_tp2_proxy.py
run "proxy rank1 binary ignored" tools/tp2/tp2_proxy.py \
  "binary = a.binary_rank1 if (r == 1 and a.binary_rank1) else a.binary" "binary = a.binary" tests/tp2/test_tp2_proxy.py
rm -rf $M
