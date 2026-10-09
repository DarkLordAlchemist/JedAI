# JedAI

**Run Qwen3.6-35B-A3B, a 35-billion-parameter AI model, on a PC with an 8 GB graphics card and 16 GB of RAM.**

Made by Ross.

https://github.com/DarkLordAlchemist/JedAI

JedAI is a local AI app: a chat page, a live Monitor of the model and your hardware, and OpenAI- and
Anthropic-compatible APIs for your apps and coding agents. Nothing leaves your PC.

## Status

Tested end to end **without a GPU**: the web app, server and engine adapter pass the protocol suite against a
stand-in llama-server, and a real llama.cpp build accepts every flag the installer produces. The real model has
not been run on a GPU yet — speed figures are estimates until measured.

## Install

**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh`.

The first run checks your graphics card and RAM, suggests a model size, downloads llama.cpp and the model (both
resume if interrupted), and opens the app at `http://127.0.0.1:8080`. Later runs start right away.
`START-HERE.bat --setup` chooses again. Already have the GGUF? `START-HERE.bat --setup --gguf D:\path\model.gguf`.

## What you need

| | |
| --- | --- |
| Graphics card | NVIDIA with 8 GB or more (AMD and Intel through Vulkan: add `--vram-gb 8`) |
| RAM | 16 GB or more |
| Disk | about 25 GB free, on an SSD (NVMe recommended: with 16 GB of RAM part of the model is read from it) |

## How does it fit?

The model is a team of 256 small "experts" in each of its 40 layers, and each word needs only 8 of them. The part
every word needs (attention, the router, the shared experts, the KV cache) stays on the graphics card. All experts stay
in RAM, and the graphics card keeps the most recently used ones in a VRAM cache. A built-in draft layer guesses the
next words and the model checks them in one pass, which makes answers faster.

## Use it from your apps

- OpenAI-compatible: base URL `http://127.0.0.1:8080/v1`, any API key, any model name.
- Anthropic-compatible: `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).

## Credits

- The web app, API server and installer design build on an MIT-licensed upstream project; its notice ships in
  `LICENSE.strata`. JedAI itself is Apache 2.0 (`LICENSE`).
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team (Apache 2.0).
