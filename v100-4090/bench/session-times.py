#!/usr/bin/env python3
"""Timing of a whole agent session from the engine's request log (start-tp.sh with
EXTRA="... --request-log-jsonl /log/requests.jsonl"): active work time without idle gaps, time to first token on short
turns, generation speed by context size, long cold reads.

  session-times.py <requests.jsonl> [--gap 10] [--from HH:MM] [--to HH:MM]   (times in UTC, as in the log)

This is how the "active work" and "generation <100k / 100-160k / beyond" numbers in the README were measured."""
import json, statistics as st, sys
from datetime import datetime, timezone

argv = sys.argv[1:]


def opt(name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


gap = float(opt("--gap", 10)) * 60
t_from, t_to = opt("--from", None), opt("--to", None)
seen, rows = set(), []
for line in open(argv[0]):
    d = json.loads(line)
    if d.get("event") != "request_done":
        continue
    q = d["request"]
    key = (q["request_id"], q["sampling"]["seed"])
    if key in seen:          # both ranks log every request
        continue
    seen.add(key)
    end = d["timestamp_unix_ms"] / 1000
    t, r = d["timings_seconds"], d["result"]
    hhmm = datetime.fromtimestamp(end, timezone.utc).strftime("%H:%M")
    if (t_from and hhmm < t_from) or (t_to and hhmm > t_to):
        continue
    rows.append(dict(start=end - t["total"], end=end, ttft=t["ttft"], out=r["completion_tokens"], dec=t["decode"],
                     prompt=r["prompt_tokens"], new=r["prompt_tokens"] - r["prefix_cache_hit_tokens"],
                     msgs=q["message_count"], thinking=r.get("model_thinking_tokens") or 0))
rows.sort(key=lambda x: x["start"])
hm = lambda s: datetime.fromtimestamp(s, timezone.utc).strftime("%H:%M")
total = rows[-1]["end"] - rows[0]["start"]
gaps = [(rows[i - 1]["end"], rows[i]["start"]) for i in range(1, len(rows)) if rows[i]["start"] - rows[i - 1]["end"] > gap]
idle = sum(b - a for a, b in gaps)
print(f"requests {len(rows)} · {hm(rows[0]['start'])} -> {hm(rows[-1]['end'])} UTC · total {total / 3600:.2f} h · "
      f"active {(total - idle) / 3600:.2f} h · idle gaps (> {gap / 60:.0f} min) {len(gaps)} = {idle / 3600:.2f} h")
for a, b in gaps:
    print(f"  idle {hm(a)} -> {hm(b)} ({(b - a) / 60:.0f} min)")
short = sorted(r["ttft"] for r in rows if r["new"] <= 200)
if short:
    print(f"first token (<=200 new tokens, n={len(short)}): median {st.median(short):.2f} s · "
          f"p90 {short[int(len(short) * .9)]:.2f} · max {max(short):.2f}")
for lo, hi in [(0, 100e3), (100e3, 160e3), (160e3, 300e3)]:
    g = [r["out"] / r["dec"] for r in rows if r["dec"] > 2 and r["out"] > 100 and lo <= r["prompt"] < hi]
    if g:
        print(f"generation {lo / 1e3:.0f}k-{hi / 1e3:.0f}k: median {st.median(g):.0f} t/s (n={len(g)})")
print(f"generated {sum(r['out'] for r in rows):,} tokens, of which reasoning {sum(r['thinking'] for r in rows):,}")
cold = [r for r in rows if r["new"] > 20000]
print(f"long cold reads (>20k new): {len(cold)} · " + ", ".join(f"{r['prompt'] // 1000}k {r['ttft']:.0f}s" for r in cold))
