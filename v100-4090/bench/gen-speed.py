#!/usr/bin/env python3
"""Generation speed on a code task: the same request (3000 tokens, temperature 0, thinking off) repeated, first run discarded.

  gen-speed.py [url=http://127.0.0.1:8097] [repeats=3] [max_tokens=3000]

Prints tokens/s, accepted MTP drafts and a hash of the text (it must be identical across repeats at temperature 0).
Reference on V100 + 4090 (default start-tp.sh): ~138 t/s on this prompt; short-context code rewrites reach ~300 t/s."""
import hashlib, json, sys, time, urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8097"
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 3000


def call():
    body = {"model": "ninfer-27b", "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": "Write a long Python module implementing a thread-safe LRU cache with "
                                                     "TTL, with docstrings and full unit tests."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"}), timeout=3600))
    return time.time() - t0, r


call()  # warm-up, discarded
for _ in range(repeats):
    dt, r = call()
    tokens = r["usage"]["completion_tokens"]
    timings = r.get("timings", {})
    drafts = f" · drafts {timings.get('draft_n_accepted')}/{timings.get('draft_n')}" if "draft_n" in timings else ""
    digest = hashlib.sha1(r["choices"][0]["message"]["content"].encode()).hexdigest()[:10]
    print(f"{tokens} tokens in {dt:.2f} s = {tokens / dt:.1f} t/s{drafts} · text {digest}", flush=True)
