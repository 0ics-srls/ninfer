#!/usr/bin/env python3
"""Time to describe one image (vision encoder + prefill + a short answer), temperature 0, thinking off.

  image.py <image.jpg|png> [url=http://127.0.0.1:8097]

A 2048x1536 photo is ~3,000 vision tokens. Reference on V100 + 4090 with the tensor-core vision attention: ~9-10 s per
new image. No test image at hand? ffmpeg -f lavfi -i "mandelbrot=s=2048x1536" -frames:v 1 test.jpg"""
import base64, json, mimetypes, sys, time, urllib.request

path = sys.argv[1]
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8097"
mime = mimetypes.guess_type(path)[0] or "image/jpeg"
data = base64.b64encode(open(path, "rb").read()).decode()
body = {"model": "ninfer-27b", "max_tokens": 80, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": "Describe this image in two sentences."}]}]}
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    url + "/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer local"}), timeout=3600))
print(f"{time.time() - t0:.1f} s · prompt {r['usage']['prompt_tokens']} tokens · {r['choices'][0]['message']['content']!r}")
