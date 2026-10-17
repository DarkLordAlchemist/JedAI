#!/usr/bin/env python3
"""engine_llamacpp/llamacpp_engine.py - a JedAI-protocol engine backed by llama.cpp's llama-server.

The web app, the OpenAI/Anthropic API and the Monitor (serve/) talk to an engine process over stdin/stdout with a
small line protocol. This file speaks that protocol and does the model work through a private llama-server, so the
whole front end runs unchanged on any model llama.cpp can load (here: gpt-oss-20b).

    server -> engine                          engine -> server
    GEN <max_new> [k=v ...] <id,id,...>       INFO k=v ...         (once, before READY: the Monitor's facts)
    GENI ...           (images: refused)      READY <ctx> stop     (loaded; honours STOP)
    STOP               (cancel the request)   RESUME <n>           (prompt tokens reused from the cache)
    VRAM [mib]         (refused)              PP <read> <total> <ms> <tok/s>
    SAVE / RESTORE     (refused)              T <id>               (one per generated token)
    QUIT                                      BEAT                 (keep-alive while llama-server is busy)
                                              DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish>
                                                   <drafts_ok> <drafts> <reused>
                                              ERR <reason>

Arguments (the config's "args"; serve/ adds --serve):
    --llama-server PATH    the llama-server executable
    --model PATH           the GGUF
    --max-context N        context size (also read by serve/ before READY)
    --llama-port N         private port for llama-server, on 127.0.0.1 (default: a free one, chosen at each start)
    --eos-ids A,B          end-of-turn ids, <|im_end|> first (default 248046,248044); setup reads them from the GGUF
    --llama-args "..."     llama-server's own options, one string (-ngl 99 --cpu-moe --moe-cache-mib 3500 ...)
Other options (the ones serve/ adds for JedAI's own engine) are logged and ignored.

Only the Python standard library is used.
"""
from __future__ import annotations

import http.client
import json
import os
import queue
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ENGINE_NAME = "llama.cpp"
BEAT_S = 10.0               # a BEAT line after this much silence during a request (serve/ ends silent engines)
DEFAULT_EOS = [248046, 248044]       # <|im_end|>, <|endoftext|> in the vocabulary


_OUT = threading.Lock()


def say(line: str) -> None:
    """One protocol line to serve/. stdout is the protocol; everything else goes to stderr (the log). The request
    thread and the main loop both write, so whole lines go out under a lock."""
    with _OUT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[llamacpp-engine] {msg}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------ arguments
def parse_args(argv: list[str]) -> dict:
    a = {"serve": False, "llama_server": None, "model": None, "max_context": 32768, "port": 0,
         "eos": list(DEFAULT_EOS), "extra": []}
    takes_value = ("--llama-server", "--model", "--max-context", "--llama-port", "--eos-ids", "--llama-args")
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--serve":
            i += 1
            continue
        if k in takes_value:
            if i + 1 >= len(argv):
                raise SystemExit(f"{k} needs a value")
            v = argv[i + 1]
            i += 2
            if k == "--llama-server":
                a["llama_server"] = v
            elif k == "--model":
                a["model"] = v
            elif k == "--max-context":
                a["max_context"] = int(v)
            elif k == "--llama-port":
                a["port"] = int(v)
            elif k == "--eos-ids":
                a["eos"] = [int(x) for x in v.split(",") if x.strip()]
            else:
                # non-POSIX keeps Windows backslashes; the quotes it leaves around a token are taken off here
                a["extra"] += [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t
                               for t in shlex.split(v, posix=False)]
            continue
        # an option for JedAI's own engine (serve/ adds some): skip it and its value, if it has one
        has_value = i + 1 < len(argv) and not argv[i + 1].startswith("--")
        log(f"ignoring {k}{' ' + argv[i + 1] if has_value else ''} (an option of JedAI's own engine)")
        i += 2 if has_value else 1
    a["serve"] = "--serve" in argv
    if not a["llama_server"] or not a["model"]:
        raise SystemExit("--llama-server and --model are required")
    return a


# ------------------------------------------------------------------------------------------------ llama-server
def free_port() -> int:
    """A port nothing listens on now, so a llama-server left over from elsewhere is never the one we talk to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ui_off_args(llama_server: str) -> list[str]:
    """llama-server's own web page must never be what a user ends up in front of (the app is JedAI's page on its
    port: Chat, Monitor, About). The switch that turns that page off was renamed between llama.cpp builds (--no-webui,
    then --no-ui), so ask this build which one it knows. If it cannot be asked, keep the old one."""
    try:
        r = subprocess.run([llama_server, "--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
        text = (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ["--no-webui"]
    if "--no-ui" in text:
        return ["--no-ui"]
    if "--no-webui" in text:
        return ["--no-webui"]
    return []           # neither is known to this build: the private API key below still locks its page


class LlamaServer:
    """The private llama-server process: started once, its log copied to our stderr and read for the facts the
    Monitor shows (the MoE cache's slots and size, the layers on the GPU)."""

    CACHE_SIZE = re.compile(r"MoE cache size =\s*([\d.]+) MiB for ([\d.]+) MiB of host experts")
    CACHE_SLOTS = re.compile(r"(\d+) layers, \S+: *(\d+) slots \(")      # llama-moe-cache.cpp: one line per group
    OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")

    def __init__(self, a: dict):
        self.port = a["port"] or free_port()
        self.facts: dict = {}
        extra = a["extra"]
        # a key only this engine knows: a browser pointed at llama-server's private port gets nothing usable, whatever
        # build of llama.cpp is installed and whichever page it ships
        if "--api-key" in extra and extra.index("--api-key") + 1 < len(extra):
            self.key = extra[extra.index("--api-key") + 1]
            key_args = []
        else:
            self.key = secrets.token_urlsafe(24)
            key_args = ["--api-key", self.key]
        cmd = [a["llama_server"], "-m", a["model"], "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(a["max_context"]), "-np", "1", *ui_off_args(a["llama_server"]), *key_args, *extra]
        shown = [("<private key>" if i and cmd[i - 1] == "--api-key" else c) for i, c in enumerate(cmd)]
        log("starting: " + " ".join(shown))
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        threading.Thread(target=self._copy_log, daemon=True).start()

    def tokenize(self, text: str) -> list[int]:
        """llama-server's own ids for `text` ([] if it cannot be asked)."""
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
            c.request("POST", "/tokenize", body=json.dumps({"content": text, "add_special": False,
                                                            "parse_special": False}),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            r = c.getresponse()
            data = json.loads(r.read() or b"{}")
            c.close()
            return [int(t) for t in data.get("tokens") or []] if r.status == 200 else []
        except (OSError, http.client.HTTPException, ValueError):
            return []

    def _copy_log(self):
        slots = 0
        for line in self.proc.stdout:
            if "listening on http://" in line:       # not an address to open: the app is JedAI's own page
                line = "llama-server is ready on its private engine port (not a web page - open JedAI's address)\n"
            sys.stderr.write(line)
            sys.stderr.flush()
            m = self.CACHE_SIZE.search(line)
            if m:
                self.facts["expert_cache_mib"] = int(float(m.group(1)))
                self.facts["host_expert_mib"] = int(float(m.group(2)))
            m = self.CACHE_SLOTS.search(line)
            if m:
                slots += int(m.group(2))
                self.facts["expert_slots"] = slots
            m = self.OFFLOADED.search(line)
            if m:
                self.facts["gpu_layers"] = f"{m.group(1)}/{m.group(2)}"

    def alive(self) -> bool:
        return self.proc.poll() is None

    def get(self, path: str, timeout: float = 5.0):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def wait_ready(self) -> dict:
        """Until /health says ok (503 while the model loads). -> /props."""
        while True:
            if not self.alive():
                raise RuntimeError(f"llama-server exited while loading (exit code {self.proc.returncode})")
            try:
                if self.get("/health", 2.0).get("status") == "ok":
                    props = self.get("/props", 10.0)
                    time.sleep(0.3)
                    if not self.alive():      # another server answered on our port; ours has exited
                        raise RuntimeError(f"llama-server exited while loading (exit code {self.proc.returncode})")
                    return props
            except urllib.error.HTTPError as e:
                if e.code != 503:
                    raise
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError):
                pass
            time.sleep(0.5)

    def stop(self):
        if not self.alive():
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
            except OSError:
                pass


# ------------------------------------------------------------------------------------------------ requests
SAMPLING_KEYS = {   # JedAI's spelling -> llama-server's
    "temperature": ("temperature", float), "top_p": ("top_p", float), "top_k": ("top_k", int),
    "min_p": ("min_p", float), "penalty_repeat": ("repeat_penalty", float),
    "penalty_freq": ("frequency_penalty", float), "penalty_present": ("presence_penalty", float),
    "penalty_last_n": ("repeat_last_n", int), "seed": ("seed", int),
}


def parse_gen(line: str) -> tuple[int, dict, list[int]]:
    """`GEN <max_new> [k=v ...] <ids>` -> (max_new, llama-server sampling fields, prompt ids)."""
    f = line.split()
    if len(f) < 3:
        raise ValueError("GEN needs <max_new> and the prompt ids")
    max_new = int(f[1])
    ids = [int(x) for x in f[-1].split(",") if x]
    # JedAI's default is greedy: an absent temperature means 0 (serve/ only forwards a temperature above 0)
    body = {"temperature": 0.0}
    for kv in f[2:-1]:
        k, _, v = kv.partition("=")
        if k in SAMPLING_KEYS:
            name, cast = SAMPLING_KEYS[k]
            body[name] = cast(v)
        # pcie_frac, spec_min_p, ckpt, prefix ...: JedAI-engine tuning with no llama.cpp counterpart
    return max_new, body, ids


class Request:
    """One GEN, streamed from llama-server's /completion. Runs on its own thread so STOP can end it at any point,
    including in the middle of a long prompt."""

    def __init__(self, srv: LlamaServer, max_new: int, sampling: dict, ids: list[int], eos: list[int]):
        self.srv, self.ids, self.eos, self.im_end = srv, ids, set(eos), eos[0]
        self.body = {"prompt": ids, "n_predict": max_new if max_new > 0 else -1, "stream": True,
                     "return_tokens": True, "return_progress": True, "cache_prompt": True, "id_slot": 0,
                     "timings_per_token": False, **sampling}
        self.cancel = threading.Event()
        self.conn: http.client.HTTPConnection | None = None
        self.last_line = time.monotonic()

    def emit(self, line: str):
        self.last_line = time.monotonic()
        say(line)

    def stop(self):
        self.cancel.set()
        c = self.conn
        if c is not None and c.sock is not None:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)   # wakes the blocked read; llama-server sees the client go
            except OSError:
                pass

    def run(self):
        produced, last_tok, final, resumed, predicted = 0, None, None, False, 0
        t0 = time.monotonic()
        try:
            self.conn = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=None)
            self.conn.request("POST", "/completion", body=json.dumps(self.body),
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {self.srv.key}"})
            resp = self.conn.getresponse()
            if resp.status != 200:
                detail = resp.read().decode("utf-8", "replace")
                try:
                    detail = json.loads(detail).get("error", {}).get("message", detail)
                except (ValueError, AttributeError):
                    pass
                self.emit(f"ERR llama-server answered {resp.status}: {str(detail).strip()[:300]}")
                return
            for raw in resp:
                if self.cancel.is_set():
                    break
                raw = raw.strip()
                if not raw.startswith(b"data:"):
                    continue
                ev = json.loads(raw[5:].strip() or b"{}")
                prog = ev.get("prompt_progress")
                if prog:
                    cache, total = int(prog.get("cache", 0)), int(prog.get("total", len(self.ids)))
                    if not resumed:
                        resumed = True
                        if cache > 0:
                            self.emit(f"RESUME {cache}")
                    done, ms = int(prog.get("processed", 0)), float(prog.get("time_ms", 0.0))
                    rate = (done - cache) / (ms / 1000.0) if ms > 0 and done > cache else 0.0
                    self.emit(f"PP {done} {total} {ms:.0f} {rate:.1f}")
                # newer llama.cpp builds put a placeholder token 0 ("!") into every prompt-progress event; it is not
                # a generated token, and used to show up as "!!!" at the start of every answer
                toks = [] if prog else list(ev.get("tokens") or [])
                # a character split across tokens (an emoji): llama-server holds the first token back and sends the
                # whole text with only the LAST token's id in `tokens` (tokens_predicted jumps by more than the ids
                # sent). Without the held token the answer shows U+FFFD, so take the ids for that text from llama-server.
                n_pred = ev.get("tokens_predicted")
                if not prog and isinstance(n_pred, int):
                    gap = n_pred - predicted - len(toks)
                    predicted = n_pred
                    if gap > 0 and ev.get("content"):
                        toks = self.srv.tokenize(ev["content"]) or toks
                for tok in toks:
                    self.emit(f"T {int(tok)}")
                    produced += 1
                    last_tok = int(tok)
                if ev.get("stop"):
                    final = ev
                    break
        except (OSError, http.client.HTTPException, ValueError) as e:
            if not self.cancel.is_set():
                if not self.srv.alive():
                    self.emit(f"ERR llama-server stopped (exit code {self.srv.proc.returncode})")
                else:
                    self.emit(f"ERR llama-server connection failed: {e}")
                return
        finally:
            try:
                if self.conn is not None:
                    self.conn.close()
            except OSError:
                pass

        if self.cancel.is_set() or final is None:
            finish = "cancel"
        else:
            st = final.get("stop_type")
            finish = "length" if st == "limit" else "stop"
            if st == "eos" and last_tok not in self.eos:
                # llama-server keeps the end-of-turn token out of the stream; JedAI's engine sends it, and
                # serve/ reads it as the end of the turn
                self.emit(f"T {self.im_end}")
                produced += 1
        tm = (final or {}).get("timings") or {}
        cache_n = int(tm.get("cache_n", 0) or 0)
        prompt_ms = float(tm.get("prompt_ms", 0.0) or 0.0)
        decode_ms = float(tm.get("predicted_ms", 0.0) or 0.0)
        if final is None:
            decode_ms = (time.monotonic() - t0) * 1000.0
        self.emit(f"DONE {produced} {len(self.ids)} {prompt_ms:.1f} {decode_ms:.1f} {finish} "
                  f"{int(tm.get('draft_n_accepted', 0) or 0)} {int(tm.get('draft_n', 0) or 0)} {cache_n}")


# ------------------------------------------------------------------------------------------------ main loop
def main(argv: list[str]) -> int:
    a = parse_args(argv)
    if not a["serve"]:
        print(__doc__)
        return 2
    srv = LlamaServer(a)
    try:
        props = srv.wait_ready()
    except Exception as e:  # noqa: BLE001 - any start failure is reported the same way
        say(f"ERR {e}")
        srv.stop()
        return 1
    n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or a["max_context"])
    time.sleep(0.2)   # the last startup lines of the log reach the reader thread
    build = str(props.get("build_info") or "").strip().replace(" ", "-")
    info = {"context": n_ctx, "engine": f"{ENGINE_NAME}-{build}" if build else ENGINE_NAME,
            "model_file": os.path.basename(a["model"])}
    for k in ("expert_slots", "expert_cache_mib", "gpu_layers"):
        if k in srv.facts:
            info[k] = srv.facts[k]
    say("INFO " + " ".join(f"{k}={v}" for k, v in info.items()))
    say(f"READY {n_ctx} stop")

    commands: queue.Queue = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.rstrip("\r\n"))
        commands.put(None)

    threading.Thread(target=read_stdin, daemon=True).start()
    current: Request | None = None
    worker: threading.Thread | None = None
    try:
        while True:
            try:
                line = commands.get(timeout=1.0)
            except queue.Empty:
                if current is not None and worker is not None:
                    if not worker.is_alive():
                        current, worker = None, None
                    elif time.monotonic() - current.last_line >= BEAT_S:
                        current.emit("BEAT")
                if not srv.alive():
                    say(f"ERR llama-server stopped (exit code {srv.proc.returncode})")
                    return 1
                continue
            if line is None or line == "QUIT":
                if current is not None:
                    current.stop()
                break
            if not line.strip():
                continue
            cmd = line.split(maxsplit=1)[0]
            if cmd == "STOP":
                if current is not None:
                    current.stop()
                continue
            if worker is not None and worker.is_alive():
                worker.join()       # serve/ sends one request at a time; wait out a stopped one before the next
            current, worker = None, None
            if cmd == "GEN":
                try:
                    max_new, sampling, ids = parse_gen(line)
                except ValueError as e:
                    say(f"ERR {e}")
                    continue
                current = Request(srv, max_new, sampling, ids, a["eos"])
                worker = threading.Thread(target=current.run, daemon=True)
                worker.start()
            elif cmd == "GENI":
                say("ERR pictures are not supported by the llama.cpp engine yet")
            elif cmd == "VRAM":
                say("ERR the llama.cpp engine cannot resize its expert cache while it runs")
            elif cmd in ("SAVE", "RESTORE"):
                say("SERR invalid 0 disk sessions are not supported by the llama.cpp engine")
            else:
                say(f"ERR unknown command {cmd}")
    finally:
        if worker is not None:
            worker.join(timeout=10)
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
