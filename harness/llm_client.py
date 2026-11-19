"""
llm_client.py - Minimal client for llama-server's /completion endpoint.

Baseline settings (project spec): stateless (we send the full prompt every
turn), prompt caching OFF (cache_prompt=false; llama-server reuses cached
prefixes by default, so the baseline must switch it off), unconstrained
sampling (no grammar). Greedy decoding (temperature 0) for reproducibility.

Returns the generated text plus llama-server's own timings:
prompt tokens processed, prefill ms, generated tokens, decode ms.
"""

import json
import time
import urllib.request


class LlamaServerClient:
    def __init__(self, url="http://127.0.0.1:8080", cache_prompt=False,
                 n_predict=512, temperature=0.0, timeout=7200):
        self.url = url.rstrip("/")
        self.cache_prompt = cache_prompt
        self.n_predict = n_predict
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt):
        body = {
            "prompt": prompt,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "cache_prompt": self.cache_prompt,
            "stop": ["<|im_end|>"],
        }
        req = urllib.request.Request(self.url + "/completion",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        wall = time.time() - t0
        tm = out.get("timings", {})
        return {
            "text": out.get("content", ""),
            "prompt_tokens": tm.get("prompt_n"),
            "cached_tokens": tm.get("cache_n", 0),
            "prefill_ms": tm.get("prompt_ms"),
            "gen_tokens": tm.get("predicted_n"),
            "decode_ms": tm.get("predicted_ms"),
            "wall_s": round(wall, 3),
        }

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            return json.loads(r.read())
