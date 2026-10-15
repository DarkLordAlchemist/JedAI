# The small-PC edition: how it is put together

JedAI gives you everything a user sees (a chat page, the Monitor, the OpenAI and Anthropic APIs, the MCP tools,
the one-click launchers) and runs it on llama.cpp: **Qwen3.6-35B-A3B** through a small engine adapter, on an 8 GB
graphics card with 16 GB of RAM. (**gpt-oss-20b**, the default model, bypasses this adapter and the server's
tokenizer - `serve/native.py` puts the same page in front of llama-server, see [GPT_OSS_20B.md](GPT_OSS_20B.md).
Everything below describes the Qwen path.)

```
 browser / apps ──HTTP──▶ serve/server.py ──stdin/stdout──▶ engine_llamacpp/llamacpp_engine.py ──HTTP──▶ llama-server
 (Chat, Monitor,          (tokenizer, chat template,         (speaks JedAI's engine protocol:      (Qwen3.6 GGUF,
  OpenAI/Anthropic)        tool calls, reasoning, queue)      GEN/STOP → T/PP/RESUME/DONE)           MoE cache, MTP)
```

## Why it works without touching the web app

`serve/server.py` never looks inside the model. It turns chats into token ids with its own tokenizer, sends the engine
one line - `GEN <max_new> [sampling] <ids>` - and reads back one `T <id>` line per token, `PP` lines while the prompt
is read, and a `DONE` line with the timings. Any program that speaks those lines is an engine. `llamacpp_engine.py`
is that program: it starts a private `llama-server` on a free local port and translates each `GEN` into a streaming
`POST /completion` with the prompt as token ids (`return_tokens`, `return_progress`, `cache_prompt`).

Qwen3.6 uses the same tokenizer family the web app was built for (byte-level BPE, pre-tokenizer `qwen35`), so the server's
tokenizer, its chat template handling, `<think>` parsing and `<tool_call><function=...>` tool calls all carry over.
Setup reads the tokenizer and the chat template from the GGUF itself (`tools/jedai_tokenizer.py`).

## What changed, file by file

| File | What it is |
| --- | --- |
| `engine_llamacpp/llamacpp_engine.py` | the engine shim (Python standard library only) |
| `engine_llamacpp/tests/` | protocol tests against `fake_llama_server.py`, a stand-in that answers like llama-server |
| `setup_lite.py` | the installer: checks the PC, picks a GGUF size, downloads llama.cpp and the model, writes the tokenizer and the config, starts the server |
| `rebrand.py` | sets your product's name in the web app, Monitor, launchers, README and `brand.json` |
| `serve/native.py` | JedAI's page in front of a model llama-server runs by itself (gpt-oss-20b); tests in `serve/test_native.py` |
| `serve/server.py` | one change: an engine whose path ends in `.py` is started with Python |
| `START-HERE.bat`, `setup.sh` | the launchers, and they run `setup_lite.py` (`rebrand.py` sets this) |

Only what this edition uses ships: no C++ engine sources, no CUDA build, no upstream docs.

## The design ideas, and where they live now

| Idea | Here |
| --- | --- |
| Attention, router, shared experts, KV cache on the GPU | `-ngl 999` with the experts kept out (`--cpu-moe`) |
| VRAM expert cache, filled with the busiest experts | llama.cpp's `--moe-cache-mib`: the most recently used experts in a VRAM bank; misses are uploaded over PCIe. Setup gives it all the VRAM left after the GPU part, the KV cache and a reserve |
| All experts in RAM, the rest from the SSD | llama.cpp memory-maps the GGUF: what does not fit in RAM is read from the disk by the OS |
| MTP "guess, then check" | `--spec-type draft-mtp --spec-draft-n-max 2`, when the GGUF carries the MTP layer (setup prefers ggml-org's MTP repo) |
| Conversation cache | `cache_prompt`: a follow-up reuses the prompt llama-server already holds (the Monitor's "Reused" column) |
| Placement by layer (`--n-cpu-moe`) | `--mode layers` in setup. **Measured faster than the cache on an 8 GB laptop** (gpt-oss-20b 22.7 against 10.7 tok/s; Qwen3.6 about 40 against 8.3), so on 8 GB of VRAM use `--mode layers` |

## Running it without a GPU (for working on the web app)

```sh
FAKE_TOKENIZER=<a tokenizer/ folder> python -m serve.server --engine jedai --config my-test.json --port 8099
```

with `my-test.json` pointing `--llama-server` at `engine_llamacpp/tests/fake_llama_server.py`. The stand-in streams a
canned answer with the same events llama-server sends, so chat, streaming, cancel and the Monitor all work.

## Not there yet

- **Pictures.** The web app sends images as embeddings from the upstream vision encoder (`GENI`); the shim refuses them.
  The way forward is to pass images to llama-server's own multimodal input (`--mmproj`).
- **The Monitor's hit-rate column** shows "–": llama.cpp counts MoE-cache hits but only prints them at shutdown. A small
  llama.cpp patch exposing the counters (for example in `/slots` or `/metrics`) would fill it.
- **One request at a time.** The `"parallel"` setting is for the upstream engine; leave it unset here.
- **Upstream-engine settings** in the About tab's model settings (`vram_elastic`, expert profiles, layer splits) do
  nothing with this engine.
- **Reasoning effort:** Qwen3.6's template knows thinking on or off; low / medium / high all think normally.

## Another model

gpt-oss-20b is in `MODELS` marked `"native"`: its tokenizer is not the Qwen family the server was built for, so
llama-server runs it with its own chat template (`--jinja`) behind `serve/native.py` ([GPT_OSS_20B.md](GPT_OSS_20B.md)).
Any other llama.cpp model that works with `--jinja` can be added the same way.

For a Qwen-family model through `serve/server.py`: add an entry to `MODELS` in `setup_lite.py` (its Hugging Face repos and default size). Any GGUF with the Qwen3.5-family
tokenizer works with the web app as it is; setup checks that and stops otherwise.
