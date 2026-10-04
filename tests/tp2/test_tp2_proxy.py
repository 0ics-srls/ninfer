"""TP2 front proxy (tools/tp2/tp2_proxy.py) against two fake ranks: same body (same per-request seed) to both ranks,
client seed kept, key injection, rank-1 binary, lockstep block mapped once per generation, and a streamed response
that ends for the client as soon as both ranks are drained (no watchdog sleep in the tail)."""

from __future__ import annotations

import http.client
import http.server
import json
import mmap
import os
import socketserver
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from tools.tp2 import tp2_proxy as T


class FakeRank(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).received.append((self.server.server_address[1], dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f"data: {{\"i\": {i}}}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(handler):
    server = Threaded(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def proxy(tmp_path):
    received = []
    rank_handler = type("Rank", (FakeRank,), {"received": received})
    ranks = [serve(rank_handler), serve(rank_handler)]
    args = SimpleNamespace(
        rank_ports=[r.server_address[1] for r in ranks], probe_key="rank-key", inject_key=True,
        ready_wait=5, max_waiting=4, pending_timeout=5, stall_seconds=60, binary="/bin/ninfer-v100",
        binary_rank1="/bin/ninfer-4090", model_prefix="/models/m", serve_args=["--ctx", "8"],
        lockstep_file=str(tmp_path / "lockstep"))
    sup = T.Supervisor(args)
    alive = SimpleNamespace(poll=lambda: None, pid=0)
    sup.procs = [alive, alive]
    sup.generation = 1
    sup.ready.set()
    handler = type("Handler", (T.Handler,), {"sup": sup})
    front = serve(handler)
    yield SimpleNamespace(port=front.server_address[1], sup=sup, received=received, ranks=args.rank_ports)
    for server in ranks + [front]:
        server.shutdown()


def post(port, body, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    start = time.monotonic()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json", "Authorization": "Bearer client", **(headers or {})})
    response = conn.getresponse()
    data = response.read()
    return response.status, data, time.monotonic() - start


def test_both_ranks_get_the_same_seeded_body_and_the_rank_key(proxy):
    status, data, _ = post(proxy.port, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and data.endswith(b"data: [DONE]\n\n")
    assert sorted(port for port, _, _ in proxy.received) == sorted(proxy.ranks)
    (_, h0, b0), (_, h1, b1) = proxy.received
    assert b0 == b1
    seed = json.loads(b0)["seed"]
    assert isinstance(seed, int) and 0 <= seed <= 0x7FFFFFFF
    for headers in (h0, h1):
        assert headers["Authorization"] == "Bearer rank-key"
    proxy.received.clear()                                  # a new request draws a new seed (2^-31 collision odds)
    post(proxy.port, {"messages": [{"role": "user", "content": "hi"}]})
    seeds = {json.loads(body)["seed"] for _, _, body in proxy.received}
    assert len(seeds) == 1 and seeds != {seed}


def test_client_seed_is_kept(proxy):
    post(proxy.port, {"messages": [], "seed": 1234})
    assert [json.loads(body)["seed"] for _, _, body in proxy.received] == [1234, 1234]


def test_stream_ends_without_the_watchdog_sleep(proxy):
    post(proxy.port, {"messages": []})                     # warm up the connections
    _, _, elapsed = post(proxy.port, {"messages": []})
    assert elapsed < 1.0, f"response held {elapsed:.2f}s after both ranks finished"


def test_request_dump(proxy, tmp_path, monkeypatch):
    monkeypatch.setenv("NINFER_TP_DUMP_REQ", str(tmp_path / "dump"))
    post(proxy.port, {"messages": [], "seed": 7})
    files = sorted(os.listdir(tmp_path / "dump"))
    assert len(files) == 1 and json.loads((tmp_path / "dump" / files[0]).read_bytes())["seed"] == 7


def test_seed_is_only_added_to_chat_completions():
    handler = object.__new__(T.Handler)
    handler.path = "/v1/completions"
    assert handler._with_seed(b'{"prompt": "x"}') == b'{"prompt": "x"}'
    handler.path = "/v1/chat/completions"
    assert handler._with_seed(b"not json") == b"not json"
    assert json.loads(handler._with_seed(b"{}"))["seed"] >= 0


def test_rank_command_uses_the_rank1_binary():
    args = SimpleNamespace(binary="/b/v100", binary_rank1="/b/4090", model_prefix="/m/q", rank_ports=[1, 2],
                           probe_key="k", serve_args=["--x"])
    sup = T.Supervisor(args)
    assert sup.rank_cmd(0)[:2] == ["/b/v100", "/m/q.rank0.ninfer"]
    assert sup.rank_cmd(1)[:2] == ["/b/4090", "/m/q.rank1.ninfer"]
    args.binary_rank1 = None
    assert sup.rank_cmd(1)[0] == "/b/v100"


def test_lockstep_block_stays_mapped_when_its_name_disappears(tmp_path):
    path = tmp_path / "lockstep"
    path.write_bytes(bytes(64) + struct.pack("<QQ", 3, 4) + bytes(48))
    sup = T.Supervisor(SimpleNamespace(lockstep_file=str(path)))
    sup.generation = 1
    assert sup.units() == (3, 4)
    with open(path, "r+b") as f:                            # the ranks advance the counters in place
        mm = mmap.mmap(f.fileno(), 0)
        struct.pack_into("<QQ", mm, 64, 5, 6)
        mm.close()
    path.unlink()                                           # systemd RemoveIPC empties /dev/shm
    assert sup.units() == (5, 6)
    sup.generation = 2                                      # a restart maps the new block
    assert sup.units() is None
    path.write_bytes(bytes(64) + struct.pack("<QQ", 9, 9) + bytes(48))
    assert sup.units() == (9, 9)
