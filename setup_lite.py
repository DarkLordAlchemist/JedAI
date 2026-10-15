#!/usr/bin/env python3
"""setup_lite.py - install and start the small-PC edition: gpt-oss-20b (default) or Qwen3.6-35B-A3B through llama.cpp,
behind the same web app, API and Monitor.

    START-HERE.bat / ./setup.sh             first run: everything; later runs: start
    ... --setup                              choose again (quant, context, placement)
    ... --gguf D:\\models\\gpt-oss-20b-mxfp4.gguf     use a GGUF you already have (no download)
    ... --model qwen3.6-35b-a3b              the other model
    ... --llama-server C:\\llama\\llama-server.exe        use a llama.cpp you already have
    ... --quant IQ4_XS --ctx 16384 --mode cache|layers --moe-cache-mib 3000 --yes --no-start

What it does, in order: checks the PC (GPU memory, RAM), picks a GGUF size, downloads llama.cpp's release and the GGUF
(both resumable), writes the tokenizer next to it, decides where the experts go, writes `<slug>-<model>.json` and starts
`python -m serve.server` with it.

Where the experts go ("--mode"):
  cache   (default) every routed expert stays in RAM; the GPU keeps the most recently used ones in a VRAM cache
          (llama.cpp --cpu-moe --moe-cache-mib). The idea of JedAI's VRAM expert tier.
  layers  the experts of the last N layers live on the GPU, the rest in RAM (llama.cpp --n-cpu-moe).
The attention / DeltaNet weights, the shared experts, the router and the KV cache are always on the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
WINDOWS = os.name == "nt"

# ------------------------------------------------------------------------------------------------ what we run
BRAND = {"name": "JedAI Lite", "slug": "jedai-lite"}
try:
    BRAND.update(json.loads((ROOT / "brand.json").read_text(encoding="utf-8")))
except (OSError, ValueError):
    pass

# The models this edition offers. Any model whose GGUF carries the Qwen3.5-family tokenizer (pre "qwen35") works with
# the web app as it is; add one here. Repos are tried in order for the size you pick; an MTP repo first, because its
# file carries the draft layer that makes answers ~1.2-1.3x faster (llama.cpp --spec-type draft-mtp).
# A model marked "native" is not run through serve/server.py (which tokenizes in the Qwen family's way) but by
# llama-server itself (`--jinja`: its own chat template and tool calls), with JedAI's page in front: serve/native.py.
MODELS = {
    "gpt-oss-20b": {
        "title": "gpt-oss-20b",
        "repos": ["ggml-org/gpt-oss-20b-GGUF"],
        "default_quant": "MXFP4",
        "license": "Apache-2.0 (OpenAI)",
        "native": True,
        "default_ctx": 131072,
    },
    "qwen3.6-35b-a3b": {
        "title": "Qwen3.6-35B-A3B",
        "repos": ["ggml-org/Qwen3.6-35B-A3B-MTP-GGUF", "unsloth/Qwen3.6-35B-A3B-GGUF",
                  "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"],
        "default_quant": "Q4_K_M",
        "license": "Apache-2.0 (Qwen)",
    },
}
DEFAULT_MODEL = "gpt-oss-20b"
QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d_K(?:_[A-Z0-9]+)?|Q\d_\d|MXFP4(?:_MOE)?|BF16|F16|Q8_0))"
                      r"(?=[-_.])")
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
EXPERT_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")

GB, MIB = 1e9, 1024 * 1024


def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(question: str, default: str, yes: bool) -> str:
    if yes or not sys.stdin.isatty():
        return default
    a = input(f"{question} [{default}]: ").strip()
    return a or default


# ------------------------------------------------------------------------------------------------ Python packages
def ensure_deps() -> None:
    """The web app's three packages, pinned as requirements.txt pins them, into this (the .venv's) Python."""
    missing = []
    for mod in ("jinja2", "regex", "psutil"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(jinja2|regex|psutil|markupsafe)==\S+", line.strip())
        if m:
            pins[m.group(1)] = m.group(0)
    want = [pins.get(m, m) for m in missing] + ([pins["markupsafe"]] if "jinja2" in missing and "markupsafe" in pins else [])
    say(f"Installing {', '.join(missing)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *want])


# ------------------------------------------------------------------------------------------------ the PC
def gpu_info() -> dict:
    """nvidia-smi's first GPU: name, total and free MiB, the driver's CUDA version. {} without an NVIDIA GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        q = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, total, free = [x.strip() for x in q.split(",")]
        head = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", head)   # newer drivers print "CUDA UMD Version"
        return {"vendor": "nvidia", "name": name, "total_mib": int(total), "free_mib": int(free),
                "cuda": (int(m.group(1)), int(m.group(2))) if m else (12, 0)}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {}


def ram_gb() -> tuple[float, float]:
    """(total, available) GB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except ImportError:
        pass
    if Path("/proc/meminfo").exists():
        mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()}
        return mi["MemTotal"] / GB, mi.get("MemAvailable", mi["MemFree"]) / GB
    return 16.0, 8.0


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return max(1, (os.cpu_count() or 8) // 2)


# ------------------------------------------------------------------------------------------------ downloads
def http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": BRAND["slug"], **hf_auth(url)})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_auth(url: str) -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok and "huggingface.co" in url else {}


def download(url: str, dest: Path, size: int | None = None) -> Path:
    """Resumable: a `.part` file continues where it stopped. A finished file of the right size is kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": BRAND["slug"], **hf_auth(url)}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and size and have == size:      # the .part is already complete
            part.replace(dest)
            return dest
        raise
    with r:
        if have and r.status != 206:                     # the server ignored the range: start over
            have = 0
        total = size or (have + int(r.headers.get("Content-Length") or 0))
        t0, last, got = time.time(), 0.0, 0
        with part.open("ab" if have else "wb") as f:
            while True:
                chunk = r.read(4 * MIB)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last >= 2 or (total and have + got >= total):
                    last = now
                    rate = got / max(1e-6, now - t0) / 1e6
                    pct = f"{100 * (have + got) / total:5.1f}%" if total else ""
                    print(f"\r  {dest.name}: {(have + got) / GB:6.2f} / {total / GB:.2f} GB {pct}  {rate:6.1f} MB/s ",
                          end="", flush=True)
        print()
    if size and part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}; run setup again to resume")
    part.replace(dest)
    return dest


# ------------------------------------------------------------------------------------------------ llama.cpp
NEEDED_FLAGS = ("--moe-cache-mib", "draft-mtp")


def llama_features(exe: Path, flags: tuple = ()) -> set[str]:
    try:
        out = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace")
        text = out.stdout + out.stderr
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {f for f in (flags or NEEDED_FLAGS) if f in text}


def release_assets(gpu: dict) -> list[list[str]]:
    """Asset-name patterns for this PC, best first: each entry is [the build, its CUDA runtime if any]. A driver runs
    builds for its own CUDA major and older ones, so every major from the driver's down is tried, then Vulkan (names
    follow llama.cpp's .github/workflows/release.yml)."""
    sysname = "win" if WINDOWS else "ubuntu" if platform.system() == "Linux" else None
    if sysname is None:
        raise SystemExit("this setup downloads llama.cpp for Windows and Linux; on another system pass --llama-server")
    ext = r"\.zip" if WINDOWS else r"\.tar\.gz"
    out = []
    if gpu.get("vendor") == "nvidia":
        for cu in range(max(12, gpu["cuda"][0]), 11, -1):
            out.append([rf"^llama-.*bin-{sysname}-cuda-{cu}\.\d+-x64{ext}$",
                        rf"^cudart-llama-.*bin-{sysname}-cuda-{cu}\.\d+-x64{ext}$"])
    out.append([rf"^llama-.*bin-{sysname}-vulkan-x64{ext}$"])
    return out


def pick_release() -> dict:
    """The newest llama.cpp release that actually carries binaries. The version tags (v0.6.0) are placeholders with
    no assets; the nightly b… prereleases hold the builds, so asking for "latest" finds nothing to download."""
    for rel in http_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"):
        if rel.get("assets"):
            return rel
    raise SystemExit("no llama.cpp release with downloads was found; download one yourself and pass --llama-server")


def install_llama(gpu: dict, dest: Path, needed: tuple = ()) -> Path:
    exe_name = "llama-server.exe" if WINDOWS else "llama-server"
    found = next(iter(sorted(dest.rglob(exe_name))), None) if dest.exists() else None
    if found and set(needed or NEEDED_FLAGS) <= llama_features(found, needed):
        return found
    say("Downloading llama.cpp (the newest release with builds) ...")
    rel = pick_release()
    names = {a["name"]: a for a in rel.get("assets", [])}
    picked = []
    for option in release_assets(gpu):
        hits = [sorted((n for n in names if re.match(pat, n)), reverse=True) for pat in option]
        if hits[0]:
            picked = [names[h[0]] for h in hits if h]
            break
    if not picked:
        raise SystemExit(f"llama.cpp {rel.get('tag_name')} has no build for this PC; download one yourself and pass "
                         "--llama-server")
    say("  " + ", ".join(a["name"] for a in picked))
    dl = dest / "downloads"
    for a in picked:
        f = download(a["browser_download_url"], dl / a["name"], a.get("size"))
        say(f"  unpacking {f.name}")
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                z.extractall(dest / "bin")
        else:
            with tarfile.open(f) as t:
                t.extractall(dest / "bin", filter="data") if sys.version_info >= (3, 12) else t.extractall(dest / "bin")
    exe = next(iter(sorted((dest / "bin").rglob(exe_name))), None)
    if not exe:
        raise SystemExit(f"{exe_name} is not in the llama.cpp download")
    if not WINDOWS:
        exe.chmod(0o755)
        if exe.parent != dest / "bin":                    # cudart's libraries next to the executable
            for lib in (dest / "bin").rglob("libcu*.so*"):
                if lib.parent != exe.parent and not (exe.parent / lib.name).exists():
                    shutil.copy2(lib, exe.parent / lib.name)
    else:
        for dll in (dest / "bin").glob("*.dll"):          # cudart zip unpacks next to, not into, the build folder
            if exe.parent != dll.parent and not (exe.parent / dll.name).exists():
                shutil.copy2(dll, exe.parent / dll.name)
    (dest / "VERSION").write_text(str(rel.get("tag_name")), encoding="utf-8")
    return exe


# ------------------------------------------------------------------------------------------------ the GGUF
def quant_of(filename: str) -> str | None:
    m = QUANT_RE.search(Path(filename).name)
    return m.group(1).upper() if m else None


def repo_files(repo: str) -> dict[str, dict]:
    """{quant: {"repo", "files": [(path, size)], "size"}} for a Hugging Face repo's GGUFs (mmproj left out)."""
    try:
        items = http_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"  ({repo}: not reachable: {e})")
        return {}
    out: dict[str, dict] = {}
    for it in items:
        p = it.get("path", "")
        if it.get("type") != "file" or not p.endswith(".gguf") or "mmproj" in p.lower():
            continue
        q = quant_of(p)
        if not q:
            continue
        size = int((it.get("lfs") or {}).get("size") or it.get("size") or 0)
        e = out.setdefault(q, {"repo": repo, "files": [], "size": 0})
        e["files"].append((p, size))
        e["size"] += size
    for e in out.values():
        e["files"].sort()
    return out


def fetch_gguf(model: dict, quant: str, dest: Path, catalog: dict) -> Path:
    e = catalog.get(quant)
    if not e:
        raise SystemExit(f"no {quant} file for {model['title']} in {', '.join(model['repos'])}")
    say(f"Downloading {model['title']} {quant} from {e['repo']} ({e['size'] / GB:.1f} GB) ...")
    paths = [download(f"https://huggingface.co/{e['repo']}/resolve/main/{p}", dest / Path(p).name, s)
             for p, s in e["files"]]
    return paths[0]                                       # llama.cpp finds the other shards from the first


def gguf_facts(path: Path) -> dict:
    """What placement needs, from the GGUF header alone: GPU-side and expert bytes, layers, the KV geometry, MTP."""
    from gguf_reader import GGUFFile
    shards = [path]
    m = SHARD_RE.search(path.name)
    if m:
        shards = [path.with_name(SHARD_RE.sub(f"-{i:05d}-of-{m.group(2)}.gguf", path.name))
                  for i in range(1, int(m.group(2)) + 1)]
    md, expert, gpu_side, embd, per_layer = {}, 0, 0, 0, {}
    for sp in shards:
        g = GGUFFile(sp)
        md.update(g.metadata)
        end = sp.stat().st_size - g.data_start
        ts = sorted(g.tensors, key=lambda t: t.offset)
        for i, t in enumerate(ts):
            nbytes = (ts[i + 1].offset if i + 1 < len(ts) else end) - t.offset
            if EXPERT_RE.search(t.name):
                expert += nbytes
                lm = re.match(r"blk\.(\d+)\.", t.name)
                if lm:
                    per_layer[int(lm.group(1))] = per_layer.get(int(lm.group(1)), 0) + nbytes
            elif t.name.startswith("token_embd"):
                embd += nbytes                            # looked up on the CPU, not held on the GPU
            else:
                gpu_side += nbytes
    arch = md.get("general.architecture", "")
    a = lambda k, d=None: md.get(f"{arch}.{k}", d)
    n_layer = int(a("block_count", 0))
    interval = a("full_attention_interval")
    n_attn = n_layer // int(interval) if interval else n_layer
    nextn = int(a("nextn_predict_layers", 0) or 0)
    return {"arch": arch, "n_layer": n_layer, "n_attn": n_attn + nextn, "mtp": nextn > 0,
            "kv_heads": int(a("attention.head_count_kv", 2) or 2),
            "k_len": int(a("attention.key_length", 256) or 256), "v_len": int(a("attention.value_length", 256) or 256),
            "expert_bytes": expert, "gpu_bytes": gpu_side, "embd_bytes": embd,
            "layer_expert_bytes": max(per_layer.values()) if per_layer else 0,
            "n_expert": int(a("expert_count", 0) or 0),
            "pre": md.get("tokenizer.ggml.pre"), "vocab": len(md.get("tokenizer.ggml.tokens") or []),
            "eos_ids": eos_ids(md)}


def eos_ids(md: dict) -> list[int]:
    """End-of-turn ids, <|im_end|> first (the engine sends it when llama.cpp ends a turn), from the GGUF's vocabulary."""
    toks = md.get("tokenizer.ggml.tokens") or []
    want = ["<|im_end|>", "<|endoftext|>"]
    found = {t: i for i, t in enumerate(toks) if t in want}
    ids = [found[t] for t in want if t in found]
    eos = md.get("tokenizer.ggml.eos_token_id")
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    return ids


# ------------------------------------------------------------------------------------------------ placement
def plan(f: dict, gpu: dict, ctx: int, mode: str, cache_mib: int | None, reserve_mib: int) -> dict:
    """The llama-server arguments for this PC, and the numbers behind them."""
    kv_mib = f["n_attn"] * ctx * f["kv_heads"] * (f["k_len"] + f["v_len"]) * 1.0625 / MIB    # q8_0 K and V
    dense_mib = f["gpu_bytes"] / MIB
    free_mib = gpu.get("free_mib", 0)
    room = free_mib - dense_mib - kv_mib - reserve_mib     # what is left for experts on the GPU
    args = ["-ngl", "999", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-t", str(physical_cores())]
    out = {"kv_mib": kv_mib, "dense_mib": dense_mib, "room_mib": room, "mode": mode}
    per_expert = f["expert_bytes"] / max(1, f["n_layer"] * max(1, f["n_expert"])) / MIB
    if mode == "cache":
        mib = cache_mib if cache_mib is not None else int(max(0, room))
        args += ["--cpu-moe"] + (["--moe-cache-mib", str(mib)] if mib >= 256 else [])
        out.update(cache_mib=mib, gpu_experts=int(mib / per_expert) if per_expert else 0)
    else:
        layer_mib = f["layer_expert_bytes"] / MIB
        on_gpu = max(0, min(f["n_layer"], int(room // layer_mib))) if layer_mib else 0
        args += ["--n-cpu-moe", str(f["n_layer"] - on_gpu)]
        out.update(gpu_expert_layers=on_gpu, gpu_experts=on_gpu * f["n_expert"])
    if f["mtp"]:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    out["args"] = args
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=f"{BRAND['name']} setup")
    ap.add_argument("--setup", action="store_true", help="choose again, even if a config exists")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--quant", help="GGUF size, e.g. Q4_K_M, IQ4_XS, UD-Q3_K_XL")
    ap.add_argument("--gguf", help="a GGUF you already have (the first shard of a split one)")
    ap.add_argument("--llama-server", help="a llama-server you already have (needs --moe-cache-mib for --mode cache)")
    ap.add_argument("--ctx", type=int, default=None, help="context size (default: 32768, 65536 from 16 GB of VRAM)")
    ap.add_argument("--mode", choices=["cache", "layers"], default=None)
    ap.add_argument("--moe-cache-mib", type=int, default=None, help="the VRAM expert cache's size (default: all that fits)")
    ap.add_argument("--vram-gb", type=float, default=None,
                    help="the GPU's memory, when it cannot be read (AMD/Intel cards have no nvidia-smi)")
    ap.add_argument("--vram-reserve-mib", type=int, default=1600,
                    help="VRAM kept free for llama.cpp's buffers and the desktop (default 1600)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--data", default=None, help="where downloads go (default: <this folder>/data-lite)")
    ap.add_argument("--yes", action="store_true", help="take every recommended answer")
    ap.add_argument("--no-start", action="store_true", help="write the config, do not start")
    a = ap.parse_args()
    ensure_deps()
    model = MODELS[a.model]
    cfg_path = ROOT / f"{BRAND['slug']}-{a.model}.json"

    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        raise SystemExit("--host other than 127.0.0.1 needs --api-key: anyone on your network could use the model")
    if cfg_path.exists() and not a.setup:
        return start(cfg_path, a)

    say(f"\n{BRAND['name']} setup - {model['title']}\n")
    data = Path(a.data) if a.data else ROOT / "data-lite"
    gpu = gpu_info()
    if a.vram_gb:
        mib = int(a.vram_gb * 1024)
        gpu = {**gpu, "vendor": gpu.get("vendor", "other"), "name": gpu.get("name", "GPU (given)"), "total_mib": mib,
               "free_mib": min(gpu.get("free_mib", mib), mib) if gpu else mib - 400, "cuda": gpu.get("cuda", (12, 0))}
    ram_total, ram_avail = ram_gb()
    if gpu:
        say(f"GPU: {gpu['name']}, {gpu['total_mib'] / 1024:.1f} GB ({gpu['free_mib'] / 1024:.1f} GB free)")
    else:
        say("GPU: no NVIDIA GPU found (nvidia-smi). AMD/Intel cards use llama.cpp's Vulkan build; the expert cache is "
            "sized from --moe-cache-mib (default: none).")
    say(f"RAM: {ram_total:.1f} GB ({ram_avail:.1f} GB available now)")
    os_reserve = 4.5 if WINDOWS else 2.0
    fits_gb = max(0.0, ram_total - os_reserve - 1.0) + max(0.0, gpu.get("free_mib", 0) / 1024 - 1.6)

    # 1. the GGUF
    if a.gguf:
        gguf = Path(a.gguf).resolve()
        if not gguf.exists():
            raise SystemExit(f"{gguf} does not exist")
    else:
        say("Looking up the available sizes ...")
        catalog: dict = {}
        for repo in model["repos"]:
            for q, e in repo_files(repo).items():
                catalog.setdefault(q, e)                   # the first repo (MTP) wins for a size it has
        if not catalog:
            raise SystemExit("Hugging Face could not be reached. Download a GGUF yourself and pass --gguf")
        sizes = sorted(catalog.items(), key=lambda kv: -kv[1]["size"])
        fitting = [q for q, e in sizes if e["size"] / GB <= fits_gb]
        rec = a.quant or (model["default_quant"] if model["default_quant"] in catalog else (fitting or [sizes[-1][0]])[0])
        say(f"\nAbout {fits_gb:.0f} GB of RAM + VRAM can hold the model on this PC. Sizes:")
        for q, e in sizes:
            mark = "fits" if e["size"] / GB <= fits_gb else "partly read from the SSD while it answers"
            mtp = ", MTP draft" if "MTP" in e["repo"] else ""
            say(f"  {q:<12} {e['size'] / GB:5.1f} GB  ({mark}{mtp})")
        quant = ask("Which size?", rec, a.yes).upper()
        if quant not in catalog:
            raise SystemExit(f"{quant} is not one of the sizes above")
        if catalog[quant]["size"] / GB > fits_gb:
            say(f"  Note: {quant} is bigger than RAM + VRAM. It runs, but the experts that do not fit are read from the "
                "disk while it answers: use an NVMe SSD, close other programs, or pick a smaller size for more speed.")
        gguf = fetch_gguf(model, quant, data / "models", catalog)

    facts = gguf_facts(gguf)
    if model.get("native"):
        return setup_native(a, model, cfg_path, gguf, facts, gpu, data, ram_total, os_reserve)
    if facts["pre"] != "qwen35":
        raise SystemExit(f"{gguf.name}: tokenizer '{facts['pre']}'. The web app speaks the Qwen3.5-family tokenizer "
                         "(pre 'qwen35') only; pick a Qwen3.5 / Qwen3.6 model.")
    if not facts["eos_ids"]:
        raise SystemExit(f"{gguf.name}: no <|im_end|> token in its vocabulary")
    say(f"\n{gguf.name}: {facts['n_layer']} layers, {facts['n_expert']} experts each; experts "
        f"{facts['expert_bytes'] / GB:.1f} GB, GPU part {facts['gpu_bytes'] / GB:.2f} GB, "
        f"MTP draft layer: {'yes' if facts['mtp'] else 'no'}")

    # 2. llama.cpp
    exe = Path(a.llama_server).resolve() if a.llama_server else install_llama(gpu, data / "llama.cpp")
    feats = llama_features(exe)
    if "draft-mtp" not in feats and facts["mtp"]:
        say("  this llama.cpp has no MTP drafting (--spec-type draft-mtp): answers come without it")
        facts["mtp"] = False

    # 3. tokenizer, read by the web app
    pack = data / "packs" / a.model
    tok_dir = pack / "tokenizer"
    if not (tok_dir / "vocab.json").exists() or a.setup:
        say("Writing the tokenizer ...")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "jedai_tokenizer.py"), "--gguf", str(gguf),
                            "--out", str(pack), "--check"], text=True)
        if r.returncode:
            raise SystemExit("the tokenizer check failed (see above)")

    # 4. placement
    vram_gb = gpu.get("total_mib", 0) / 1024
    ctx = a.ctx or int(ask("Context size (tokens)", "65536" if vram_gb >= 15 else "32768", a.yes))
    mode = a.mode or ("cache" if "--moe-cache-mib" in feats else "layers")
    if mode == "cache" and "--moe-cache-mib" not in feats:
        say("  this llama.cpp has no --moe-cache-mib: experts are placed by layer instead")
        mode = "layers"
    p = plan(facts, gpu, ctx, mode, a.moe_cache_mib, a.vram_reserve_mib)
    if p["room_mib"] < 0:
        say(f"  The GPU part ({p['dense_mib'] / 1024:.1f} GB) and the KV cache ({p['kv_mib'] / 1024:.2f} GB) need more "
            "VRAM than is free: try a smaller --ctx, or close what uses the GPU.")
    cpu_side = (facts["expert_bytes"] - (p.get("gpu_expert_layers", 0) * facts["layer_expert_bytes"])) / GB
    say(f"Placement ({mode}): KV cache {p['kv_mib'] / 1024:.2f} GB at {ctx:,} tokens; "
        + (f"VRAM expert cache {p.get('cache_mib', 0) / 1024:.1f} GB (~{p['gpu_experts']:,} experts)" if mode == "cache"
           else f"{p['gpu_expert_layers']} of {facts['n_layer']} layers' experts on the GPU"))
    if cpu_side > ram_total - os_reserve:
        say(f"  RAM: the experts ({cpu_side:.1f} GB) are more than this PC's RAM leaves free; the operating system "
            "pages them from the disk as needed. Fine on an NVMe SSD, slow on a hard disk.")

    # 5. the config serve/ reads
    rel = lambda x: os.path.relpath(x, ROOT) if str(Path(x).resolve()).startswith(str(ROOT)) else str(x)
    cfg = {
        "exe": "engine_llamacpp/llamacpp_engine.py",
        "args": ["--llama-server", rel(exe), "--model", rel(gguf), "--max-context", str(ctx),
                 "--eos-ids", ",".join(str(i) for i in facts["eos_ids"]),
                 "--llama-args", " ".join(f'"{x}"' if " " in x else x for x in p["args"])],
        "cwd": ".",
        "tokenizer": rel(tok_dir),
        "model_name": a.model,
        "log": f"{BRAND['slug']}-{a.model}.log",
        "host": a.host,
        "port": a.port,
        "engine_silence_s": 600,
    }
    if a.api_key:
        cfg["api_key"] = a.api_key
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    say(f"\nWrote {cfg_path.name}. llama-server will run with: {' '.join(p['args'])}")
    if a.no_start:
        return 0
    return start(cfg_path, a)


def setup_native(a, model: dict, cfg_path: Path, gguf: Path, facts: dict, gpu: dict, data: Path,
                 ram_total: float, os_reserve: float) -> int:
    """A native model (gpt-oss-20b): llama-server runs it with its own chat template; serve/native.py puts JedAI's page
    in front. Placement is by layer: on an 8 GB laptop it measured 2x faster than the VRAM expert cache."""
    say(f"\n{gguf.name}: {facts['n_layer']} layers, {facts['n_expert']} experts each; experts "
        f"{facts['expert_bytes'] / GB:.1f} GB, rest {facts['gpu_bytes'] / GB:.2f} GB")
    needed = ("--n-cpu-moe", "--jinja")
    exe = Path(a.llama_server).resolve() if a.llama_server else install_llama(gpu, data / "llama.cpp", needed)
    missing = [f for f in needed if f not in llama_features(exe, needed)]
    if missing:
        raise SystemExit(f"this llama.cpp lacks {', '.join(missing)}: delete data-lite/llama.cpp (or pass a newer "
                         "--llama-server) and run setup again")
    ctx = a.ctx or int(ask("Context size (tokens)", str(model.get("default_ctx", 32768)), a.yes))
    p = plan(facts, gpu, ctx, "layers", None, a.vram_reserve_mib)
    on_gpu = p["gpu_expert_layers"]
    if p["room_mib"] < 0:
        say(f"  The GPU part and the KV cache ({p['kv_mib'] / 1024:.2f} GB at {ctx:,} tokens) need more VRAM than is "
            "free: try a smaller --ctx, or close what uses the GPU.")
    cpu_side = (facts["expert_bytes"] - on_gpu * facts["layer_expert_bytes"]) / GB
    say(f"Placement (layers): the experts of {on_gpu} of {facts['n_layer']} layers on the GPU, the rest ({cpu_side:.1f} GB) in RAM")
    if cpu_side > ram_total - os_reserve:
        say("  RAM: the experts are more than this PC's RAM leaves free; the operating system pages them from the disk.")
    rel = lambda x: os.path.relpath(x, ROOT) if str(Path(x).resolve()).startswith(str(ROOT)) else str(x)
    cfg = {
        "native": True,
        "llama_server": rel(exe),
        "model": rel(gguf),
        "model_name": a.model,
        "quant": quant_of(gguf.name),
        "max_context": ctx,
        "gpu_expert_layers": on_gpu,
        "llama_args": ["-ngl", "999", "--n-cpu-moe", str(facts["n_layer"] - on_gpu), "--jinja"],
        "log": f"{BRAND['slug']}-{a.model}.log",
        "host": a.host,
        "port": a.port,
    }
    if a.api_key:
        cfg["api_key"] = a.api_key
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    say(f"\nWrote {cfg_path.name}. llama-server will run with: {' '.join(cfg['llama_args'])}")
    if a.no_start:
        return 0
    return start(cfg_path, a)


UI_FILES = ("index.html", "monitor.html", "monitor.js", "app.js", "app.css", "components.css", "tokens.css", "sprite.svg")
UI_TABS = ("chat", "monitor", "about")


def check_ui(root: Path = ROOT) -> list[str]:
    """What is missing from the web app (the page with Chat, Monitor and About). Empty when it is all there. Every
    install runs this before starting, so a half-copied folder never ends up showing some other page."""
    web = root / "serve" / "web"
    problems = [f"serve/web/{n}" for n in UI_FILES if not (web / n).is_file()]
    try:
        html = (web / "index.html").read_text(encoding="utf-8")
    except OSError:
        return problems
    problems += [f"the {t.capitalize()} tab" for t in UI_TABS if f'data-tab="{t}"' not in html]
    return problems


def start(cfg_path: Path, a) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    port = str(cfg.get("port", a.port))
    host = cfg.get("host", "127.0.0.1")
    missing = check_ui()
    if missing:
        say(f"\nThe web app is incomplete here (missing: {', '.join(missing)}).")
        say(f"Download {BRAND['name']} again from GitHub and run START-HERE again; nothing was started.")
        return 1
    say(f"\nStarting {BRAND['name']} on http://{host}:{port} (close this window to stop) ...")
    say(f"Open http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/ - the {BRAND['name']} app with Chat, "
        "Monitor and About. llama.cpp's own page is switched off; its private port is not for browsing.")
    say("The first start reads the model into memory: the PC can be slow for a minute or two.\n")
    if cfg.get("native"):
        cmd = [sys.executable, "-m", "serve.native", "--config", str(cfg_path), "--port", port, "--open"]
    else:
        cmd = [sys.executable, "-m", "serve.server", "--engine", "jedai", "--config", str(cfg_path), "--port", port,
               "--open"]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
