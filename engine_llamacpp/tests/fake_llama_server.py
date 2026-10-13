#!/usr/bin/env python3
"""A stand-in for llama.cpp's llama-server, built from its source (tools/server/server-task.cpp, server-context.cpp):
503 on /health while 'loading', /props with n_ctx and build_info, POST /completion streaming SSE `data: {...}` events
with prompt_progress (return_progress), tokens per event (return_tokens, 1-3 per event like MTP drafts), the
end-of-turn token kept out of the stream, a final event with stop/stop_type/timings and empty tokens, and a cancel
when the client goes. Every request body is appended to $FAKE_LOG.

Used by test_llamacpp_engine.py; also handy for working on the web app without a GPU (see docs/LITE_EDITION.md)."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


args = sys.argv[1:]
if "--help" in args:
    # FAKE_HELP: what this build's --help lists (e.g. only "--no-webui" for an older llama.cpp, "--no-ui" for a newer one)
    print(os.environ.get("FAKE_HELP", "--moe-cache-mib N   GPU cache size in MiB for the MoE experts kept in the CPU\n"
                                      "--spec-type  [draft-mtp,...]\n--webui, --no-webui\n--ui, --no-ui"))
    sys.exit(0)
opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
API_KEY = opt("--api-key")      # like llama-server: everything but /health needs "Authorization: Bearer <key>"
if os.environ.get("FAKE_ARGS"):             # the command line this server was started with (the test reads it)
    with open(os.environ["FAKE_ARGS"], "w") as f:
        f.write(json.dumps(args))
port, n_ctx = int(opt("--port")), int(opt("-c", "4096"))
# with FAKE_TOKENIZER (a tokenizer/ folder) the reply is real text; without it, plain ids (enough for protocol tests)
tok = None
if os.environ.get("FAKE_TOKENIZER"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
    import jedai_tokenizer as ST  # noqa: E402
    tokdir = os.environ["FAKE_TOKENIZER"]
    vocab = json.load(open(f"{tokdir}/vocab.json", encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    tok = ST.Tokenizer(tokens, open(f"{tokdir}/merges.txt", encoding="utf-8").read().split("\n"),
                       json.load(open(f"{tokdir}/token_type.json")))
REPLY = tok.encode("<think>\nThe user greets me.\n</think>\n\nHello from Qwen3.6 through llama.cpp!",
                   parse_special=True) if tok else list(range(100, 124))
LONG = tok.encode(" counting" * 400) if tok else [7] * 2000
LONG_TRIGGER = 99999            # a prompt that ends with this id gets the long reply (cancel tests)
started = time.time()


def log_request(obj):
    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(json.dumps(obj) + "\n")
last_prompt: list[int] = []

# the startup lines the engine shim reads (formats from llama-model.cpp and llama-moe-cache.cpp)
print("load_tensors: offloaded 41/41 layers to GPU", file=sys.stderr, flush=True)
if "--moe-cache-mib" in args:
    mib = float(opt("--moe-cache-mib"))
    print(f"impl:      CUDA0 MoE cache size = {mib:8.2f} MiB for 19098.00 MiB of host experts", file=sys.stderr, flush=True)
    print(f"impl: 41 layers, q4_K: {int(mib / 1.82):5d} slots (18.1%)", file=sys.stderr, flush=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authorized(self):
        return not API_KEY or self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def do_GET(self):
        if self.path == "/health":
            if time.time() - started < 1.5:
                return self.send_json(503, {"error": {"code": 503, "message": "Loading model"}})
            return self.send_json(200, {"status": "ok"})
        if not self.authorized():
            return self.send_json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
        if self.path == "/props":
            return self.send_json(200, {"default_generation_settings": {"n_ctx": n_ctx}, "build_info": "b9301-86a2835"})
        self.send_json(404, {})

    def do_POST(self):
        global last_prompt
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.authorized():
            return self.send_json(401, {"error": {"code": 401, "message": "Invalid API Key"}})
        log_request({k: v for k, v in body.items() if k != "prompt"} | {"n_prompt": len(body["prompt"])})
        prompt = body["prompt"]
        cache = 0
        while cache < min(len(prompt), len(last_prompt)) and prompt[cache] == last_prompt[cache]:
            cache += 1
        cache = min(cache, len(prompt) - 1)
        last_prompt = list(prompt)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def ev(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        try:
            t0 = time.time()
            total = len(prompt)
            for done in range(cache, total + 1, max(1, (total - cache) // 3 or 1)):
                ev({"prompt_progress": {"total": total, "cache": cache, "processed": done,
                                        "time_ms": (time.time() - t0) * 1000}, "tokens": [], "content": "", "stop": False})
                time.sleep(0.05)
            ev({"prompt_progress": {"total": total, "cache": cache, "processed": total,
                                    "time_ms": (time.time() - t0) * 1000}, "tokens": [], "content": "", "stop": False})
            t1 = time.time()
            long = prompt[-1] == LONG_TRIGGER or (tok is not None and "count forever" in tok.decode(prompt[-40:]))
            out = LONG if long else REPLY
            n_max = body.get("n_predict", -1)
            n_max = len(out) if n_max is None or n_max < 0 else n_max
            gen, i, drafts, accepted = [], 0, 0, 0
            while i < len(out) and len(gen) < n_max:
                k = min(1 + (i % 3), len(out) - i, n_max - len(gen))      # 1-3 tokens per step, as MTP gives
                chunk = out[i:i + k]
                drafts += 2
                accepted += k - 1
                ev({"index": 0, "content": tok.decode(chunk) if tok else "", "tokens": chunk, "stop": False, "id_slot": 0,
                    "tokens_predicted": len(gen) + k, "tokens_evaluated": total})
                gen += chunk
                i += k
                time.sleep(0.03)
            stop_type = "limit" if len(gen) >= n_max and i < len(out) else "eos"
            ev({"index": 0, "content": "", "tokens": [], "id_slot": 0, "stop": True, "stop_type": stop_type,
                "tokens_predicted": len(gen), "tokens_evaluated": total, "tokens_cached": total + len(gen),
                "timings": {"cache_n": cache, "prompt_n": total - cache, "prompt_ms": (t1 - t0) * 1000,
                            "predicted_n": len(gen), "predicted_ms": (time.time() - t1) * 1000,
                            "draft_n": drafts, "draft_n_accepted": accepted}})
        except (BrokenPipeError, ConnectionResetError):
            log_request({"cancelled": True})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
