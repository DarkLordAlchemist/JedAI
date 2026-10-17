# JedAI

**Run a capable AI model on a PC with an 8 GB graphics card and 16 GB of RAM. Default: gpt-oss-20b, placed across the GPU and RAM layer by layer.**

Made by Jay Ortiz.

https://github.com/DarkLordAlchemist/JedAI

JedAI is a local AI app: a chat page, a live Monitor of the model and your hardware, and OpenAI- and
Anthropic-compatible APIs for your apps and coding agents. Nothing leaves your PC.

## Status

Run for real on a laptop with an RTX 5070 8 GB and 16.9 GB of RAM (2026-10-15):

| Model | Setting | Generation speed |
| --- | --- | ---: |
| **gpt-oss-20b** (MXFP4, 12.1 GB) | layers mode, `--n-cpu-moe 18` | 22.7 tok/s |

One run per setting, so treat small gaps as noise. The model file is 12.1 GB, so it fits in 16 GB of RAM with room to
spare, and it ran with a 131,072-token window on the 8 GB card. The numbers, the command and how the page is put in
front of it are in [docs/GPT_OSS_20B.md](docs/GPT_OSS_20B.md).

The test suites (web app, server, engine adapter) also pass without a GPU, against a stand-in llama-server.

## What gpt-oss-20b is good at

gpt-oss-20b is a 20-billion-parameter model running on a laptop, so use it for work it does well:

- Answering factual questions and explaining concepts
- Writing, editing, or brainstorming text in many styles (articles, stories, emails, reports, etc.)
- Summarizing documents, articles, or long conversations
- Assisting with programming, debugging, and explaining code snippets
- Providing guidance on research, study strategies, and project planning
- Offering creative ideas for projects, marketing, design, or content creation
- Translating text between languages or helping with language learning
- Helping with math, science, and logic problems, including step-by-step solutions

What to expect instead:

- **It has no internet or tools of its own.** The chat page sends your words to the model and nothing else, so it cannot
  search the web, open links or read today's news; it answers from what it learned in training and says so when you
  ask. Anything that needs the web has to come from a coding agent or app that brings its own tools and talks to
  JedAI's API (below).
- **It is a small model.** On facts it may be wrong or out of date, and on hard reasoning it is weaker than the large
  cloud models. Check what matters.

## Install

**Windows:** double-click `START-HERE.bat`. **Linux:** run `./setup.sh`.

The first run checks your graphics card and RAM, picks the optimal starting setup for them (gpt-oss-20b, the layer
placement below, and a 131,072-token window), downloads llama.cpp and the model (both resume if interrupted), and
opens the app at `http://127.0.0.1:8080`. Later runs start right away. `START-HERE.bat --setup` chooses again.
Already have the GGUF? `START-HERE.bat --setup --gguf D:\path\gpt-oss-20b-MXFP4.gguf`.

The page you use is always JedAI's own app at `http://127.0.0.1:8080/` - **Chat, Monitor and About**. llama.cpp's own
web page is switched off (whichever llama.cpp build is installed), and the port llama.cpp listens on is private and
locked with a key only JedAI knows, so it is never the page you end up on. Every start also checks that the app's files
are all there and refuses to start, naming what is missing, if they are not.

## What you need

| | |
| --- | --- |
| Graphics card | NVIDIA with 8 GB or more (AMD and Intel through Vulkan: add `--vram-gb 8`) |
| RAM | 16 GB or more |
| Disk | about 15 GB free, on an SSD |

## How does it fit?

gpt-oss-20b is a "mixture of experts": a team of small experts in each of its 24 layers, where each word uses only a
few. JedAI splits it layer by layer:

1. **Always on the graphics card:** the part every word needs (attention, the router, the KV cache).
2. **Experts, layer by layer:** the experts of as many layers as fit in your VRAM are placed on the graphics card; the
   experts of the remaining layers stay in RAM (llama.cpp `--n-cpu-moe`). On the 8 GB reference laptop that is 6 layers
   on the GPU and 18 in RAM.
3. **More VRAM, more layers on the GPU:** setup counts the free VRAM and moves more layers over, so a bigger card is
   faster with no settings to change.

On an 8 GB card this layer placement is about 2x faster than the VRAM expert cache, so it is the default.

## Use it from your apps

- OpenAI-compatible: base URL `http://127.0.0.1:8080/v1`, any API key, any model name.
- Anthropic-compatible: `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).

## Credits

- The web app, API server and installer design build on an MIT-licensed upstream project. JedAI itself is Apache 2.0 (`LICENSE`).
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Model: [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) by OpenAI (Apache 2.0).
