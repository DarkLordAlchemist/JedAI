# gpt-oss-20b on an 8 GB laptop

**gpt-oss-20b is JedAI's default model**, and the one recommended for an 8 GB graphics card with 16 GB of RAM.
`START-HERE.bat` installs it. This page has the measurements behind that, the command it runs, and how the Chat /
Monitor / About page is put in front of it.

- Model: [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF), file `gpt-oss-20b-MXFP4.gguf` (12.1 GB,
  24 layers, mixture of experts), Apache 2.0.
- Why it fits: the whole file is 12.1 GB, so with 16 GB of RAM the experts that are not on the graphics card are held
  in RAM and not paged from the SSD (Qwen3.6-35B-A3B is about 21 GB and is).

## Measured

PC: Intel 12 threads, 16.9 GB RAM (6.3 GB free during the run), NVIDIA GeForce RTX 5070 Laptop GPU 8 GB, Windows 11.
`laptop_bench.py`, 2026-10-15, three requests per setting. Generation speed in tokens per second.

| Setting | llama-server flags | Request 1 / 2 / 3 | Warm median | Prompt reading | VRAM used | Load |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| experts-cpu | `--cpu-moe` | 13.8 / 19.4 / 20.0 | 19.7 | 101 tok/s | 1.9 GB | 51 s |
| cache-3000 | `--cpu-moe --moe-cache-mib 3000` | 9.0 / 10.3 / 11.0 | 10.7 | 100 tok/s | 4.9 GB | 63 s |
| **layers-6** | `--n-cpu-moe 18` | **23.0 / 22.3 / 23.1** | **22.7** | **126 tok/s** | 4.3 GB | 41 s |

- **Layer placement wins**: the experts of the last 6 layers live on the GPU, the other 18 in RAM. It is about 15 %
  faster than all-experts-in-RAM and **2x faster than the VRAM expert cache**.
- **The VRAM expert cache loses again**, as it did with Qwen3.6 and Nemotron on this laptop. With 8 GB of VRAM there
  is too little room left for it to pay for its PCIe uploads. Do not use `--moe-cache-mib` on an 8 GB card.
- One run per setting, on a PC with little free RAM: treat differences under about 10 % as noise.
- For comparison, the same laptop with Qwen3.6-35B-A3B in layers mode measured about 40 tok/s. gpt-oss-20b is
  slower per token; it is the smaller download (12.1 GB against about 21 GB) and the one that stays in RAM.

### 131,072-token context

The setup that runs day to day uses the winning placement with a 128k window:

```
llama-server -m gpt-oss-20b-MXFP4.gguf -c 131072 --n-cpu-moe 18 --jinja
```

It started and answered on the 8 GB card; the speed above was measured at the bench's default window, not at 131,072.

## How the page is put in front of it

JedAI's own server (`serve/server.py`) only knows the Qwen tokenizer family, so it is **not** used for gpt-oss.
llama.cpp's `llama-server` already does the chat template (`--jinja`), tool calls and the Anthropic `/v1/messages` API
for this model. `serve/native.py` (Python standard library only; `START-HERE.bat` starts it) runs llama-server on a
private port, locked with a key only it knows, and puts JedAI's page in front:

```
 browser / Claude Code / apps ──▶ serve/native.py :8080 ──▶ llama-server (private port) ──▶ GPU + RAM
 (ANTHROPIC_BASE_URL              serves Chat, Monitor, About;    gpt-oss-20b-MXFP4.gguf
  = http://127.0.0.1:8080)        answers /health, /metrics,      -c 131072 --n-cpu-moe N --jinja
                                  /v1/status from the requests
                                  that pass and the GPU/CPU/RAM
                                  readings; forwards the rest
```

Use it like the other model: OpenAI base URL `http://127.0.0.1:8080/v1`, Claude Code `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.
What the Monitor shows: the live state and speed, GPU load / VRAM / temperature / power / PCIe, CPU, RAM, disk, the
context meter and the request table (tokens read and written, speed, time).
What it cannot show:

- The About tab's **Model settings** card is missing (it needs JedAI's run-config `/config`; this model has none).
- The **cache hit-rate** column and the conversation-cache card are empty (llama.cpp prints hit counts only at shutdown).
- **Prefill progress** while a prompt is read is not shown; the state reads "Reading prompt" without a percentage.
- No pictures yet (the page's attach button takes text files only).

### Where the experts go

`setup_lite.py` puts the experts of as many layers as fit the GPU on it (`--n-cpu-moe 24 - N`) and leaves the rest in
RAM: on the 8 GB laptop 4-6 layers, on a 32 GB card all 24. It never uses `--moe-cache-mib` for this model.

### Claude Code

Give it a short tool list: gpt-oss's chat template can choke on big connector schemas.

## If the Monitor says "Server not reachable"

The page polls `/metrics` every couple of seconds and gives up after 3 failures in a row. If llama-server was
restarted while the Monitor was open (for example while changing the context size), reload the page. In the one case
seen so far, every endpoint answered correctly when checked afterwards, so a stale open tab was the likely cause.
That is a guess, not a confirmed cause.
