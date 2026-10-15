# JedAI

**Run a large AI model on a PC with an 8 GB graphics card and 16 GB of RAM. Recommended: gpt-oss-20b. Built-in installer model: Qwen3.6-35B-A3B.**

Made by Jay Ortiz.

https://github.com/DarkLordAlchemist/JedAI

JedAI is a local AI app: a chat page, a live Monitor of the model and your hardware, and OpenAI- and
Anthropic-compatible APIs for your apps and coding agents. Nothing leaves your PC.

## Status

Run for real on a laptop with an RTX 5070 8 GB and 16.9 GB of RAM (2026-10-15):

| Model | Best setting | Generation speed |
| --- | --- | ---: |
| **gpt-oss-20b** (MXFP4, 12.1 GB), recommended | `--n-cpu-moe 18` | 22.7 tok/s |
| Qwen3.6-35B-A3B (about 21 GB) | layers mode, 10 layers on the GPU | about 40 tok/s |

One run per setting, so treat small gaps as noise. gpt-oss-20b is the one that fits in 16 GB of RAM with room to
spare, and it ran with a 131,072-token window on the 8 GB card. The numbers, the command and how the page is put in
front of it are in [docs/GPT_OSS_20B.md](docs/GPT_OSS_20B.md). **`START-HERE.bat` installs Qwen3.6-35B-A3B**;
gpt-oss-20b is set up by hand for now (that page).

The test suites (web app, server, engine adapter) also pass without a GPU, against a stand-in llama-server.

## Install

**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh`.

The first run checks your graphics card and RAM, suggests a model size, downloads llama.cpp and the model (both
resume if interrupted), and opens the app at `http://127.0.0.1:8080`. Later runs start right away.
`START-HERE.bat --setup` chooses again. Already have the GGUF? `START-HERE.bat --setup --gguf D:\path\model.gguf`.

The page you use is always JedAI's own app at `http://127.0.0.1:8080/` - **Chat, Monitor and About**. llama.cpp's own
web page is switched off (whichever llama.cpp build is installed), and the port llama.cpp listens on is private and
locked with a key only JedAI knows, so it is never the page you end up on. Every start also checks that the app's files
are all there and refuses to start, naming what is missing, if they are not.

## What you need

| | |
| --- | --- |
| Graphics card | NVIDIA with 8 GB or more (AMD and Intel through Vulkan: add `--vram-gb 8`) |
| RAM | 16 GB or more |
| Disk | about 25 GB free, on an SSD (NVMe recommended: with 16 GB of RAM part of the model is read from it) |

## How does it fit?

Both models are a "mixture of experts": a team of small experts per layer where each word uses only a few. The part
every word needs stays on the graphics card, the experts that do not fit stay in RAM. **On an 8 GB card, placing whole
layers' experts on the GPU (`--n-cpu-moe`) is faster than the VRAM expert cache** - 2x for gpt-oss-20b.

The details for the Qwen3.6-35B-A3B installer: the model is a team of 256 small "experts" in each of its 40 layers, and each word needs only 8 of them. The part
every word needs (attention, the router, the shared experts, the KV cache) stays on the graphics card. All experts stay
in RAM, and the graphics card keeps the most recently used ones in a VRAM cache. A built-in draft layer guesses the
next words and the model checks them in one pass, which makes answers faster.

## Use it from your apps

- OpenAI-compatible: base URL `http://127.0.0.1:8080/v1`, any API key, any model name.
- Anthropic-compatible: `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).

## Credits

- The web app, API server and installer design build on an MIT-licensed upstream project. JedAI itself is Apache 2.0 (`LICENSE`).
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Models: [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) by OpenAI (Apache 2.0) and
  [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team (Apache 2.0).
