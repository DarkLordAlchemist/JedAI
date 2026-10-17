# How JedAI is put together

JedAI gives you everything a user sees (a chat page, the Monitor, the OpenAI and Anthropic APIs, the one-click
launchers) and runs **gpt-oss-20b** on llama.cpp, on an 8 GB graphics card with 16 GB of RAM. llama-server runs the
model with its own chat template and tool calls; `serve/native.py` puts JedAI's page in front of it. See
[GPT_OSS_20B.md](GPT_OSS_20B.md) for the measurements, the command and the diagram.

## Files

| File | What it is |
| --- | --- |
| `setup_lite.py` | the installer: checks the PC, downloads llama.cpp and the model, decides how many layers of experts go on the GPU, writes the config, starts the app |
| `serve/native.py` | JedAI's page (Chat, Monitor, About) in front of a private llama-server; tests in `serve/test_native.py` |
| `serve/web/` | the page's files |
| `serve/server.py` | the OpenAI- and Anthropic-compatible API server |
| `engine_llamacpp/` | helpers that start and talk to llama-server; tests in `engine_llamacpp/tests/` |
| `rebrand.py` | sets your product's name in the web app, Monitor, launchers, README and `brand.json` |
| `START-HERE.bat`, `setup.sh` | the launchers; they run `setup_lite.py` |

## Where the model goes

| Part | Where |
| --- | --- |
| Attention, router, KV cache | on the GPU (`-ngl 999`) |
| Experts, layer by layer | as many layers as fit the free VRAM on the GPU, the rest in RAM (`--n-cpu-moe`); `--mode layers` is the default |
| Conversation cache | `cache_prompt`: a follow-up reuses the prompt llama-server already holds |

## Running it without a GPU (for working on the web app)

Point `--llama-server` in a test config at `engine_llamacpp/tests/fake_llama_server.py`. The stand-in streams a
canned answer with the same events llama-server sends, so chat, streaming, cancel and the Monitor all work.

## Not there yet

- **Pictures:** the page's attach button takes text files only.
- **The Monitor's hit-rate column** shows "–": llama.cpp prints cache hit counts only at shutdown.
- **One request at a time.**
