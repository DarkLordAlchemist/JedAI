#!/usr/bin/env python3
"""rebrand.py - give this edition your own name, everywhere a user sees it.

    python rebrand.py --name "Lumen" [--slug lumen] [--url https://github.com/you/lumen] [--author "Your Name"]

Run it again with another name whenever you like: it sets the name, it does not search-and-replace the old one.
What it changes: the web app's title, header and About page, the API Monitor's title, the launchers' window titles and
which setup they run, brand.json (setup and config file names read it) and README.md (the original is kept as
docs/JEDAI_README.md). What it keeps: the license files (`LICENSE`, `LICENSE.strata` - upstream notices must
stay with the code) and every internal name (JEDAI_* settings, file names in serve/), so updates still merge.
"""
import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sub(path: Path, pattern: str, repl: str, count: int = 0, flags: int = 0) -> None:
    with path.open(encoding="utf-8", newline="") as f:     # keeps the .bat files' CRLF line ends
        text = f.read()
    new, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n == 0:
        raise SystemExit(f"{path.relative_to(ROOT)}: found nothing to change for {pattern!r} (has the file changed?)")
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(new)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\\", "\\\\")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the product name users see")
    ap.add_argument("--slug", help="lower-case, for file names (default: from the name)")
    ap.add_argument("--url", default="", help="your project's page (the About tab links to it)")
    ap.add_argument("--author", default="", help="your name, for the credits")
    a = ap.parse_args()
    name = a.name.strip()
    slug = a.slug or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not name or not slug:
        raise SystemExit("--name needs letters or digits")
    h = esc(name)

    # the web app
    web = ROOT / "serve" / "web"
    sub(web / "index.html", r"<title>[^<]*</title>", f"<title>{h}</title>", 1)
    sub(web / "index.html", r'(<span class="st-header__brand">)[^<]*(</span>)', rf"\g<1>{h}\g<2>", 1)
    link = (f'<a href="{esc(a.url)}" target="_blank" rel="noopener noreferrer">{h}</a> · ' if a.url else "") + \
        '<a href="https://github.com/DarkLordAlchemist/JedAI" target="_blank" rel="noopener noreferrer">based on JedAI</a> · ' \
        '<a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noopener noreferrer">llama.cpp</a>'
    sub(web / "index.html", r'<a href="https://github.com/DarkLordAlchemist/JedAI"[^>]*>JedAI on GitHub</a>'
                            r'|<!--brand-links-->.*?<!--/brand-links-->', f"<!--brand-links-->{link}<!--/brand-links-->",
        1, re.S)
    sub(web / "index.html", r"\(close [^)]*?'s window and start it again\)", f"(close {h}'s window and start it again)", 1)
    sub(web / "monitor.html", r"<title>[^<]*</title>", f"<title>{h} API Monitor</title>", 1)
    sub(web / "monitor.html", r'(<div class="brand">)[^<]*(<span>)', rf"\g<1>{h} \g<2>", 1)

    # launchers: their window title, and the small-PC setup
    bat = re.sub(r'([&|<>^])', r"^\1", name).replace("%", "%%")        # cmd.exe's special characters
    sub(ROOT / "START-HERE.bat", r"(?m)^title [^\r\n]*", "title " + bat.replace("\\", "\\\\"), 1)
    sub(ROOT / "START-HERE.bat", r'(\.venv\\Scripts\\python\.exe" )setup(?:_lite)?\.py', r"\g<1>setup_lite.py", 1)
    sub(ROOT / "setup.sh", r"exec \.venv/bin/python setup(?:_lite)?\.py", "exec .venv/bin/python setup_lite.py", 1)

    # brand.json: setup_lite.py names the config, the log and its messages from it
    (ROOT / "brand.json").write_text(json.dumps({"name": name, "slug": slug, "url": a.url, "author": a.author},
                                                indent=2) + "\n", encoding="utf-8")

    # README: yours; JedAI's kept beside the docs
    keep = ROOT / "docs" / "JEDAI_README.md"
    readme = ROOT / "README.md"
    if not keep.exists():
        shutil.copy2(readme, keep)
    by = f"\n\nMade by {a.author}." if a.author else ""
    readme.write_text(README.format(name=name, slug=slug, by=by, year=date.today().year,
                                    url=f"\n{a.url}\n" if a.url else ""), encoding="utf-8")
    print(f"{name} ({slug}): web app, Monitor, launchers, brand.json and README.md updated.")
    print("Restart the server (close its window, run START-HERE again) to see the new name.")
    return 0


README = """# {name}

**Run gpt-oss-20b, a 20-billion-parameter AI model, on a PC with an 8 GB graphics card and 16 GB of RAM.**{by}
{url}
{name} is a local AI app: a chat page, a live Monitor of the model and your hardware, and OpenAI- and
Anthropic-compatible APIs for your apps and coding agents. Nothing leaves your PC.

## Install

**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh`.

The first run checks your graphics card and RAM, suggests a model size, downloads llama.cpp and the model (both
resume if interrupted), and opens the app at `http://127.0.0.1:8080`. Later runs start right away.
`START-HERE.bat --setup` chooses again. Already have the GGUF? `START-HERE.bat --setup --gguf D:\\path\\model.gguf`.

## What you need

| | |
| --- | --- |
| Graphics card | NVIDIA with 8 GB or more (AMD and Intel through Vulkan: add `--vram-gb 8`) |
| RAM | 16 GB or more |
| Disk | about 15 GB free, on an SSD |

## How does it fit?

The model is a team of small "experts" in each of its 24 layers, and each word needs only a few. The part every word
needs (attention, the router, the KV cache) stays on the graphics card. The experts of as many layers as fit in your
VRAM go on the graphics card too; the rest stay in RAM.

## Use it from your apps

- OpenAI-compatible: base URL `http://127.0.0.1:8080/v1`, any API key, any model name.
- Anthropic-compatible: `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).

## Credits

- Built on [JedAI](https://github.com/DarkLordAlchemist/JedAI) (Apache 2.0), which itself builds on an
  MIT-licensed upstream project whose notice ships in `LICENSE.strata`.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Model: [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) by OpenAI (Apache 2.0).
"""

if __name__ == "__main__":
    raise SystemExit(main())
