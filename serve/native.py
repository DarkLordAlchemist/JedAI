#!/usr/bin/env python3
"""serve/native.py - JedAI's own app (Chat, Monitor, About) in front of a model llama-server runs by itself.

serve/server.py tokenizes and renders chats itself, so it only fits one tokenizer family. A model like
gpt-oss-20b is better left to llama-server: `--jinja` does its chat template, its tool calls and its Anthropic
`/v1/messages`. This module starts llama-server on a private port (locked with a key only this process knows) and puts
JedAI's page in front of it:

  * the page and its files (serve/web) are JedAI's own, the same ones serve/server.py serves;
  * /health, /metrics, /v1/status, /settings, /mcp are answered here, from the requests that pass through and from
    the hardware telemetry (serve/telemetry.py), in the shapes the page reads;
  * everything else (/v1/chat/completions, /v1/messages, /v1/responses, /v1/models, ...) is forwarded untouched.

    python -m serve.native --config jedai-gpt-oss-20b.json [--port 8080]

Python standard library only (psutil, if installed, gives the CPU / RAM / disk figures).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_llamacpp.llamacpp_engine import free_port, ui_off_args  # noqa: E402
from serve.telemetry import Telemetry  # noqa: E402

WEB = ROOT / "serve" / "web"
API_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")   # counted in the Monitor
TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
         ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8"}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "content-length", "host"}
KEEP = 100          # requests the Monitor keeps


def log(msg: str) -> None:
    sys.stderr.write(f"[jedai] {msg}\n")
    sys.stderr.flush()


class Stats:
    """What the Monitor shows, built from the requests that pass through."""

    def __init__(self, model: str, ctx: int, extra: dict):
        self.model, self.ctx, self.extra = model, ctx, extra
        self.lock = threading.Lock()
        self.live: dict[int, dict] = {}
        self.history: list[dict] = []
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0}
        self.started = time.time()
        self.last_tok_s = 0.0
        self.loaded = False
        self.telemetry = Telemetry(extra=self._series)
        self._n = 0

    def _series(self) -> dict:
        with self.lock:
            cur = [r for r in self.live.values() if r.get("first")]
            rate = self._rate(cur[-1]) if cur else 0.0
        return {"tok_s": round(rate, 2) if rate else 0.0}

    @staticmethod
    def _rate(r: dict) -> float:
        dt = time.time() - r["first"]
        return r["generated"] / dt if dt > 0.25 and r["generated"] > 1 else 0.0

    def begin(self, path: str) -> int:
        with self.lock:
            self._n += 1
            self.live[self._n] = {"path": path, "started": time.time(), "first": None, "generated": 0,
                                  "prompt_tokens": None, "reused": None, "prompt_ms": None, "decode_ms": None}
            return self._n

    def token(self, rid: int, n: int = 1) -> None:
        with self.lock:
            r = self.live.get(rid)
            if r is not None:
                if r["first"] is None:
                    r["first"] = time.time()
                r["generated"] += n

    def end(self, rid: int, finish: str, usage: dict) -> None:
        with self.lock:
            r = self.live.pop(rid, None)
            if r is None:
                return
            now = time.time()
            n = usage.get("output_tokens")
            n = r["generated"] if n is None else n
            prompt = usage.get("prompt_tokens")
            reused = usage.get("reused")
            p_ms, d_ms = usage.get("prompt_ms"), usage.get("decode_ms")
            if d_ms is None and r["first"] and n and now - r["first"] > 0.05:     # a whole answer in one piece has no decode time
                d_ms = (now - r["first"]) * 1000
            if p_ms is None and r["first"]:
                p_ms = (r["first"] - r["started"]) * 1000
            tok_s = round(n / (d_ms / 1000), 1) if n and d_ms else None
            read = max(0, (prompt or 0) - (reused or 0))
            self.history.append({
                "projection": None, "time": r["started"], "duration_s": round(now - r["started"], 1), "finish": finish,
                "prompt_tokens": prompt or 0, "reused": reused, "output_tokens": n, "prompt_total": prompt,
                "prompt_read": read, "prompt_ms": p_ms, "decode_ms": d_ms, "decode_tok_s": tok_s, "hit_rate": None,
                "pcie_share": None, "drafts_offered": None, "drafts_accepted": None, "path": r["path"]})
            del self.history[:-KEEP]
            t = self.totals
            t["requests"] += 1
            t["prompt_tokens"] += prompt or 0
            t["reused"] += reused or 0
            t["output_tokens"] += n or 0
            t["prompt_ms"] += p_ms or 0.0
            t["decode_ms"] += d_ms or 0.0

    def metrics(self, all_requests: bool = False) -> dict:
        with self.lock:
            hist, totals = list(self.history), dict(self.totals)
            cur = list(self.live.values())[-1] if self.live else None
            queued = max(0, len(self.live) - 1)
            loaded = self.loaded
            if cur and cur["first"] is None:
                state = "reading"
            elif cur:
                state = "generating"
            else:
                state = "idle" if loaded else "unloaded"
            rate = round(self._rate(cur), 1) if cur and state == "generating" else None
            live = {"state": state, "queued": queued, "phase": None,
                    "prompt_tokens": cur and cur["prompt_tokens"], "prompt_read": None, "prompt_total": None,
                    "generated": cur and cur["generated"], "max_tokens": None,
                    "elapsed_s": round(time.time() - cur["started"], 1) if cur else None,
                    "tok_s": rate, "tok_s_mean": rate, "prefill_tok_s_mean": None, "tok_s_window_s": None}
        tel = self.telemetry.snapshot()
        engine = {"model": self.model, "max_context": self.ctx, "images": False, **self.extra}
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "conversation_cache": None, "requests_kept": len(hist), "totals": totals,
                "hardware": tel["now"], "hardware_static": tel["static"], "history": tel["history"], "time": time.time()}

    def v1_status(self) -> dict:
        with self.lock:
            totals, busy = dict(self.totals), len(self.live)
            last = self.history[-1] if self.history else None
        hw = self.telemetry.snapshot()
        now, static = hw["now"], hw["static"]
        return {"service": "jedai", "model": self.model, "loaded": self.loaded, "auto_load": False,
                "engine": self.extra.get("version"), "started": int(self.started),
                "uptime_s": int(time.time() - self.started), "cache_max_tokens": self.ctx,
                "context": {"native": self.ctx, "max_positions": self.ctx},
                "concurrency": {"serving": 1, "requested": 1},
                "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
                "vision": {"enabled": False, "available": False, "error": None},
                "activity": {"requests": totals["requests"] + busy, "in_flight": busy,
                             "last_request_at": int(last["time"]) if last else None},
                "last_timings": None,
                "machine": {"at": int(time.time()),
                            "gpu": {"name": static.get("gpu_name"),
                                    "used_mib": round(now["gpu_mem_used"] / 2 ** 20) if now.get("gpu_mem_used") else None,
                                    "total_mib": round(now["gpu_mem_total"] / 2 ** 20) if now.get("gpu_mem_total") else None,
                                    "util_pct": now.get("gpu_util"), "temp_c": now.get("gpu_temp"),
                                    "power_w": now.get("gpu_power")} if static.get("gpu_name") else None}}


def usage_of(obj: dict, acc: dict) -> None:
    """Fold the token counts and timings one llama-server JSON (a whole answer, or one stream event) carries into acc."""
    if not isinstance(obj, dict):
        return
    msg = obj.get("message")
    for u in (obj.get("usage"), msg.get("usage") if isinstance(msg, dict) else None):
        if isinstance(u, dict):
            for src, dst in (("prompt_tokens", "prompt_tokens"), ("input_tokens", "prompt_tokens"),
                             ("completion_tokens", "output_tokens"), ("output_tokens", "output_tokens"),
                             ("cache_read_input_tokens", "reused")):
                if isinstance(u.get(src), int) and u[src] >= acc.get(dst, 0):
                    acc[dst] = u[src]
    t = obj.get("timings")
    if isinstance(t, dict):
        if isinstance(t.get("cache_n"), int):
            acc["reused"] = t["cache_n"]
            if isinstance(t.get("prompt_n"), int):
                acc["prompt_tokens"] = t["prompt_n"] + t["cache_n"]
        if isinstance(t.get("prompt_ms"), (int, float)):
            acc["prompt_ms"] = t["prompt_ms"]
        if isinstance(t.get("predicted_ms"), (int, float)):
            acc["decode_ms"] = t["predicted_ms"]
        if isinstance(t.get("predicted_n"), int):
            acc["output_tokens"] = t["predicted_n"]


class Handler(BaseHTTPRequestHandler):
    server_version = "JedAI"
    stats: Stats
    upstream: tuple[str, int]
    upstream_key: str
    api_key: str
    model: str
    ctx: int
    ready: threading.Event

    def log_message(self, *a):          # the console stays quiet
        pass

    # ---------------------------------------------------------------- helpers
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        got = self.headers.get("Authorization", "")
        got = got[7:] if got.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
        if secrets.compare_digest(got.encode(), self.api_key.encode()):
            return True
        self._json(401, {"error": {"type": "authentication_error", "message": "an API key is needed"}})
        return False

    def _body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            out = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if not size:
                    while self.rfile.readline().strip():
                        pass
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        if path == "":
            return self._file(WEB / "index.html", "text/html; charset=utf-8")
        if path.startswith("/web/") or path.startswith("/fonts/"):
            rel = path[len("/web/"):] if path.startswith("/web/") else path[1:]
            f = (WEB / rel).resolve()
            if WEB.resolve() not in f.parents or f.suffix not in TYPES or not f.is_file():
                return self._json(404, {"error": {"message": "not found"}})
            return self._file(f, TYPES[f.suffix])
        if path in ("/health", "/api/health"):
            return self._json(200, {"status": "ok" if self.ready.is_set() else "loading", "max_context": self.ctx,
                                    "model": self.model, "images": False, "api_key": bool(self.api_key),
                                    "loaded": self.ready.is_set(), "service": "jedai"})
        if path == "/metrics":
            if self._authorized():
                self._json(200, self.stats.metrics("requests=all" in self.path))
            return
        if path == "/v1/status":
            if self._authorized():
                self._json(200, self.stats.v1_status())
            return
        if path == "/settings":
            if self._authorized():
                self._json(200, {"shared": False, "defaults": {}})
            return
        if path == "/mcp":
            if self._authorized():
                self._json(200, {"servers": [], "tools": 0})
            return
        if path == "/config":           # JedAI's run-config file: this model has none, the About tab hides its card
            return self._json(404, {"error": {"message": "not found"}})
        if path.startswith("/v1/") or path in ("/props", "/slots", "/models"):
            if self._authorized():
                self._forward("GET", None)
            return
        self._json(404, {"error": {"message": "not found"}})

    def _file(self, f: Path, ctype: str) -> None:
        try:
            body = f.read_bytes()
        except OSError:
            return self._json(404, {"error": {"message": "not found"}})
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        if not self._authorized():
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/settings":
            self._body()
            return self._json(200, {"shared": False, "defaults": {}})
        if path in ("/config", "/load", "/unload"):
            self._body()
            return self._json(404, {"error": {"message": "this model is managed by llama-server, not by a run config"}})
        self._forward("POST", self._body())

    def do_DELETE(self):
        self._json(404, {"error": {"message": "not found"}})

    # ---------------------------------------------------------------- the forward
    def _forward(self, method: str, body: bytes | None) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        counted = method == "POST" and path in API_PATHS
        rid = self.stats.begin(path) if counted else None
        acc: dict = {}
        finish = "stop"
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP and k.lower() not in ("authorization", "x-api-key")}
        headers["Authorization"] = f"Bearer {self.upstream_key}"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            conn = http.client.HTTPConnection(*self.upstream, timeout=3600)
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            if rid:
                self.stats.end(rid, "error", {})
            return self._json(502, {"error": {"type": "api_error",
                                              "message": f"the model server is not reachable ({e.__class__.__name__}): "
                                                         "it may still be loading or has stopped"}})
        try:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            streaming = "text/event-stream" in (resp.getheader("Content-Type") or "")
            buf = b""
            while True:
                chunk = resp.read1(8192) if streaming else resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if not rid:
                    continue
                buf += chunk
                if streaming:
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.startswith(b"data:"):
                            self._event(line[5:].strip(), rid, acc)
                elif len(buf) > 8 << 20:
                    buf = buf[-(1 << 20):]
            if rid and not streaming:
                try:
                    usage_of(json.loads(buf), acc)
                    self.stats.token(rid, acc.get("output_tokens", 0))
                except ValueError:
                    pass
            if resp.status >= 400:
                finish = "error"
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            finish = "disconnect"
        except OSError:
            finish = "error"
        finally:
            conn.close()
            if rid:
                self.stats.end(rid, finish, acc)

    def _event(self, raw: bytes, rid: int, acc: dict) -> None:
        if not raw or raw == b"[DONE]":
            return
        try:
            obj = json.loads(raw)
        except ValueError:
            return
        usage_of(obj, acc)
        delta = ""
        if isinstance(obj.get("choices"), list) and obj["choices"]:
            d = obj["choices"][0].get("delta") or {}
            delta = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if d.get("tool_calls"):
                delta += "x"
        elif obj.get("type") == "content_block_delta":
            d = obj.get("delta") or {}
            delta = d.get("text") or d.get("thinking") or d.get("partial_json") or ""
        elif obj.get("type") in ("response.output_text.delta", "response.reasoning_text.delta"):
            delta = obj.get("delta") or ""
        if delta:
            self.stats.token(rid)


def wait_ready(host: str, port: int, key: str, proc: subprocess.Popen, ready: threading.Event, stats: Stats) -> None:
    while proc.poll() is None:
        try:
            c = http.client.HTTPConnection(host, port, timeout=5)
            c.request("GET", "/health", headers={"Authorization": f"Bearer {key}"})
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                ready.set()
                stats.loaded = True
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="JedAI's page in front of a model llama-server runs on its own")
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--open", action="store_true", help="open the page in the browser when the model is ready")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    host, port = cfg.get("host", "127.0.0.1"), a.port or int(cfg.get("port", 8080))
    exe = str((ROOT / cfg["llama_server"]).resolve() if not os.path.isabs(cfg["llama_server"]) else cfg["llama_server"])
    model_path = str((ROOT / cfg["model"]).resolve() if not os.path.isabs(cfg["model"]) else cfg["model"])
    ctx = int(cfg["max_context"])
    lport, key = free_port(), secrets.token_urlsafe(24)
    cmd = [exe, "-m", model_path, "--host", "127.0.0.1", "--port", str(lport), "-c", str(ctx), "-np", "1",
           *ui_off_args(exe), "--api-key", key, *cfg.get("llama_args", [])]
    log("starting llama-server: " + " ".join(c if c != key else "<private key>" for c in cmd))
    logf = open(ROOT / cfg.get("log", "jedai-native.log"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    version = ""
    try:
        version = (Path(exe).resolve().parents[1] / "VERSION").read_text().strip()
    except OSError:
        pass
    stats = Stats(cfg.get("model_name", "gpt-oss-20b"), ctx,
                  {"version": f"llama.cpp {version}".strip(), "quant": cfg.get("quant"),
                   "gpu_expert_layers": cfg.get("gpu_expert_layers")})
    ready = threading.Event()
    Handler.stats, Handler.upstream, Handler.upstream_key = stats, ("127.0.0.1", lport), key
    Handler.api_key, Handler.model, Handler.ctx, Handler.ready = cfg.get("api_key", ""), stats.model, ctx, ready
    threading.Thread(target=wait_ready, args=("127.0.0.1", lport, key, proc, ready, stats), daemon=True).start()
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    log(f"JedAI is on http://{shown}:{port}/ (Chat, Monitor, About). The model is loading; the page says so until it is ready.")

    def announce():
        while not ready.is_set() and proc.poll() is None:
            time.sleep(0.5)
        if ready.is_set():
            log(f"ready: {stats.model}, {ctx:,} tokens of context. Open http://{shown}:{port}/")
            if a.open:
                webbrowser.open(f"http://{shown}:{port}/")
        else:
            log(f"llama-server stopped (exit {proc.returncode}); see {cfg.get('log', 'jedai-native.log')}")
            os._exit(1)
    threading.Thread(target=announce, daemon=True).start()

    def stop(*_):
        proc.terminate()
        os._exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        srv.serve_forever()
    finally:
        proc.terminate()
        stats.telemetry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
