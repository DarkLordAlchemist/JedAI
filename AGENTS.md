# AGENTS.md

JedAI runs large mixture-of-experts models on a normal PC (gpt-oss-20b, run by llama-server behind `serve/native.py`, see
[docs/GPT_OSS_20B.md](docs/GPT_OSS_20B.md)): an 8 GB graphics
card plus 16 GB of RAM, on Windows or Linux. It has a Python server with an OpenAI- and Anthropic-compatible API
and a web app (`serve/`), an engine adapter that drives a private llama-server (`engine_llamacpp/`), and a
one-click installer (`setup_lite.py`, started by `START-HERE.bat` / `setup.sh`).

## Working on the code

- How the pieces fit and why the swap works: [docs/LITE_EDITION.md](docs/LITE_EDITION.md).
  Symptoms and fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
- The server talks to the engine over a small line protocol (`GEN`/`STOP` in, `T`/`PP`/`RESUME`/`DONE`/`INFO`/
  `READY`/`ERR` out). `engine_llamacpp/llamacpp_engine.py` speaks it - keep both sides in step.
- Tests run without a GPU or downloads:
  `python -m pytest engine_llamacpp/tests -q` (the adapter, against a stand-in llama-server) and
  `python -m pytest serve -q` (the server, `serve/native.py` and the web app).
- The tokenizer tool has its own tests: `python -m unittest tools.test_jedai_tokenizer`.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- One change per pull request. Say what it changes and what it leaves alone.
- A change is ready when both suites pass. Say how you checked, and on what hardware any GPU run was done.
