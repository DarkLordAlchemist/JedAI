# Something went wrong?

The common problems and what to do. Everything the model printed while starting is in
`jedai-<model>.log` (for example `jedai-gpt-oss-20b.log`) in the JedAI folder - start there, and attach it when you ask for help.

## While installing or starting

**It stopped while downloading or installing.**
Run `START-HERE.bat` (Linux: `./setup.sh`) again. Downloads continue where they stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
JedAI is already running - look for its window. Or another program uses the port: `START-HERE.bat --port 8081`.

**I see a different page than Chat / Monitor / About (for example llama.cpp's own chat page).**
Open exactly `http://127.0.0.1:8080/` - the address the console prints after `Open`. Do not open the other port number
shown in the log: that is llama.cpp's private port, it needs a key and is not a web page. If `START-HERE.bat` says the
web app is incomplete, download JedAI again from GitHub (a half-copied folder is missing `serve/web`).

**AMD or Intel card not detected.**
Setup only sees NVIDIA cards on its own. Tell it about yours: `START-HERE.bat --setup --vram-gb 8`; it then uses
llama.cpp's Vulkan build.

**Hugging Face refuses the model download.**
The repo may be gated: set `HF_TOKEN` before running setup, or download the GGUF yourself and run
`START-HERE.bat --setup --gguf D:\models\model.gguf`.

**The first load takes several minutes, and the PC feels slow.**
Normal: with 16 GB of RAM llama.cpp reads a 21 GB model partly through the SSD. When the console says `ready:`,
the browser opens. Don't close the window.

**Out of memory on the graphics card while loading.**
The VRAM estimate was too generous - the desktop, the browser and other apps also live on the card. Run setup
again with `--vram-reserve-mib 2400`, or fix the expert cache: `--moe-cache-mib 2500`.

## While it answers

**Very slow, and the disk light is always on.**
The experts are being paged from the SSD: RAM is full. Close other programs, or pick a smaller size
(`--setup --quant IQ4_XS`). An NVMe SSD helps more than anything else.

**Slow, but the GPU sits idle.**
The experts are being computed on the CPU. Use layer placement: `--setup --mode layers`. On an 8 GB card it measured
2x faster than `--setup --mode cache`.

**The Monitor shows no "experts cached" number.**
The llama.cpp build you have lacks `--moe-cache-mib`, so setup fell back to layer placement. Delete
`data-lite\llama.cpp` and run setup again to get a newer build.

**The Monitor says "Server not reachable".**
Reload the page. The Monitor stops polling after 3 failed requests and does not always recover when llama-server was
restarted while the tab was open.

**"the engine said nothing for 600 s".**
The first prompt on a paging PC can take that long. Raise `"engine_silence_s"` in `jedai-<model>.json`,
for example to `1200`.

## Still stuck?

Open an [issue](https://github.com/DarkLordAlchemist/JedAI/issues) and attach
the `jedai-<model>.log` from the JedAI folder, with the console's last 30 lines.
