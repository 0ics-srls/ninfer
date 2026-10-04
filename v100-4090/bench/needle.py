#!/usr/bin/env python3
"""Long-context needles with SHORT follow-up questions, temperature 0.

  needle.py [url=http://127.0.0.1:8097] [context_tokens=120000]

Hides three codes (start, middle, end) in ~N tokens of this repository's C++ headers. The first request loads the
context (its time = cold prefill); then each question arrives as a short turn at the end of the conversation and must
return the exact code. Expected: 3/3, ~1,000-1,160 t/s cold prefill, follow-up answers in well under a second."""
import glob, json, os, sys, time, urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8097"
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
needles = {"of the harbour": "7391-QX", "of the tower": "4826-LM", "of the bridge": "5058-RT"}

chunks, n = [], 0
for f in sorted(glob.glob(os.path.join(repo, "src", "**", "*.h"), recursive=True)):
    s = open(f, encoding="utf-8", errors="replace").read()
    chunks.append(f"// {os.path.relpath(f, repo)}\n{s}")
    n += len(s) / 3.2
    if n > ctx:
        break
k = len(chunks)
for (name, code), i in zip(needles.items(), [2, k // 2, k - 2]):
    chunks.insert(i, f"// NOTE: the secret code {name} is {code}.\n")
msgs = [{"role": "system", "content": "You are an assistant. Answer briefly.\n\n" + "\n".join(chunks)},
        {"role": "user", "content": "Answer only: ready"}]


def ask(m):
    body = {"model": "ninfer-27b", "messages": m, "max_tokens": 40, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"}), timeout=3600))
    return r["choices"][0]["message"]["content"].strip(), time.time() - t0, r["usage"]["prompt_tokens"]


answer, dt, pt = ask(msgs)
print(f"loaded {pt} tokens in {dt:.0f} s = {pt / dt:.0f} t/s cold prefill: {answer!r}")
msgs.append({"role": "assistant", "content": answer})
ok = 0
for name, code in needles.items():
    msgs.append({"role": "user", "content": f"What is the secret code {name}? Answer only with the code."})
    answer, dt, pt = ask(msgs)
    ok += code in answer
    print(f"{name}: {answer!r} (expected {code}) {'OK' if code in answer else 'WRONG'} · {dt:.2f} s · prompt {pt}")
    msgs.append({"role": "assistant", "content": answer})
print(f"{ok}/3")
