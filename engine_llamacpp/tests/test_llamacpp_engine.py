"""engine_llamacpp/tests/test_llamacpp_engine.py - the engine's line protocol against a stand-in llama-server.

    python -m pytest engine_llamacpp/tests -q        (or: python engine_llamacpp/tests/test_llamacpp_engine.py)

No GPU, no model, no llama.cpp: fake_llama_server.py answers the way llama-server does. What is checked is what
serve/ relies on: INFO + READY, PP/RESUME, one T per token, the end-of-turn token at the end, DONE's fields, STOP,
the refusals, JedAI-only options ignored, and a llama-server that dies while loading."""
import os
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "llamacpp_engine.py"
FAKE = HERE / "fake_llama_server.py"
IM_END = 151645


def fake_exe(tmp: Path) -> str:
    """The stand-in as something the engine can start directly (a .cmd on Windows)."""
    if os.name != "nt":
        return str(FAKE)
    cmd = tmp / "fake_llama_server.cmd"
    cmd.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n', encoding="utf-8")
    return str(cmd)


class Engine:
    def __init__(self, llama_server: str, extra=()):
        env = {**os.environ, "FAKE_LOG": ""}
        env.pop("FAKE_TOKENIZER", None)
        self.p = subprocess.Popen([sys.executable, str(ENGINE), "--serve", "--llama-server", llama_server,
                                   "--model", "m.gguf", "--max-context", "8192", "--eos-ids", f"{IM_END},151643",
                                   "--llama-args", "-ngl 999 --cpu-moe --moe-cache-mib 1820", *extra],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, env=env)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def until(self, prefix: str, timeout: float = 20.0) -> list[str]:
        out, end = [], time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            out.append(line)
            if line.startswith(prefix):
                return out
        raise AssertionError(f"no {prefix} line; got {out[-10:]}")

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = Path(tempfile.mkdtemp())
        cls.e = Engine(fake_exe(cls.tmp), extra=["--expert-profile", "x.bin", "--vision"])
        cls.start = cls.e.until("READY")

    @classmethod
    def tearDownClass(cls):
        cls.e.close()

    def test_ready_and_info(self):
        info = dict(kv.split("=", 1) for kv in self.start[0].split()[1:])
        self.assertEqual(self.start[-1], "READY 8192 stop")
        self.assertEqual(info["expert_slots"], str(int(1820 / 1.82)))
        self.assertEqual(info["expert_cache_mib"], "1820")
        self.assertTrue(info["engine"].startswith("llama.cpp-b"))

    def test_generate_and_reuse(self):
        self.e.send("GEN 200 temperature=0.7 top_k=20 ckpt=0 1,2,3,4,5,6,7,8")
        lines = self.e.until("DONE")
        toks = [int(x[2:]) for x in lines if x.startswith("T ")]
        self.assertEqual(toks[-1], IM_END)                     # added: llama-server keeps it out of the stream
        self.assertEqual(len(toks), 25)
        self.assertTrue(any(x.startswith("PP ") for x in lines))
        done = lines[-1].split()
        self.assertEqual(done[1:3], ["25", "8"])
        self.assertEqual(done[5], "stop")
        self.assertEqual(len(done), 9)
        # same prefix again: the cache is reused and said with RESUME
        self.e.send("GEN 3 1,2,3,4,5,6,7,8,9")
        lines = self.e.until("DONE")
        self.assertIn("RESUME 8", lines)
        done = lines[-1].split()
        self.assertEqual((done[1], done[5], done[8]), ("3", "length", "8"))

    def test_stop_mid_answer(self):
        self.e.send("GEN 2000 1,2,99999")
        self.e.until("T ")
        self.e.send("STOP")
        done = self.e.until("DONE")[-1].split()
        self.assertEqual(done[5], "cancel")
        self.assertLess(int(done[1]), 2000)
        self.e.send("GEN 2 1,2")                               # and the next request runs
        self.assertEqual(self.e.until("DONE")[-1].split()[5], "length")

    def test_refusals(self):
        self.e.send("GENI 5 x 1,2")
        self.assertTrue(self.e.until("ERR")[-1].startswith("ERR pictures"))
        self.e.send("VRAM 100")
        self.assertTrue(self.e.until("ERR")[-1].startswith("ERR"))
        self.e.send("SAVE x")
        self.assertTrue(self.e.until("SERR")[-1].startswith("SERR invalid"))


class TestStartFailure(unittest.TestCase):
    def test_server_that_exits(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        if os.name == "nt":
            bad = tmp / "bad.cmd"
            bad.write_text("@exit /b 3\r\n")
        else:
            bad = tmp / "bad.sh"
            bad.write_text("#!/bin/sh\nexit 3\n")
            bad.chmod(0o755)
        e = Engine(str(bad))
        lines = e.until("ERR")
        self.assertIn("exited while loading", lines[-1])
        self.assertEqual(e.p.wait(timeout=15), 1)


if __name__ == "__main__":
    unittest.main()
