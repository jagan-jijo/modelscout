# ModelScout

Find local AI models that fit your hardware, with memory and speed estimates.

Downloading a model just to find out it eats all your RAM is a pain. ModelScout
checks your machine and gives you a shortlist before you download the weights.
You can use it in the terminal or open the local web dashboard to compare models.

I’m focusing on macOS and Linux first. Windows has a launcher, but hardware
detection there is still a basic fallback. Use explicit hardware settings on
Windows and check the results against your machine.

## Run it

```bash
git clone https://github.com/jagan-jijo/modelscout.git
cd modelscout
./start.sh
```

The launcher creates `.venv` and installs the Python dependencies. It uses Python
3.11 or newer if available. On macOS and Linux, it can use `uv` to download Python
when needed; if neither is installed, it downloads `uv` into `.local/bin` using
`curl`. You’ll need an internet connection for the first setup.

Run the same command again whenever you need it. Missing Python packages are
installed again automatically. A broken environment is moved into `.local/`
before a replacement is created, so any files you put there are kept.

On Windows, install Python 3.11+ or `uv`, then run:

```bat
start.bat
start.bat web
```

ModelScout doesn’t install GPU drivers or inference runtimes. It looks for tools
such as `nvidia-smi`, `rocm-smi`, `lspci` and Ollama when they’re available. You can
still use the catalogue and supply hardware settings without them.

## A few useful commands

```bash
# Open the web dashboard at http://127.0.0.1:1234
./start.sh web

# Inspect this machine
./start.sh hardware

# Pick a workload
./start.sh --profile coding
./start.sh --profile reasoning
./start.sh --profile vision
./start.sh --profile fast

# Try a different GPU and memory size
./start.sh --gpu "RTX 4090" --vram 24 --ram 64

# Compare entries from the local catalogue
./start.sh compare qwen llama

# Use a longer context window
./start.sh --context-length 16384

# Save a report for a script
./start.sh --json > report.json

# See all options
./start.sh --help
```

The web server runs in the foreground. Stop it with `Ctrl+C`. For a remote Linux
machine, forward the local port over SSH and open it in your browser:

```bash
ssh -L 1234:127.0.0.1:1234 user@server
# In that SSH session, run ./start.sh web from the checkout.
```

## What the results mean

Each recommendation includes a memory estimate, a token generation speed range,
and an explanation of its ranking. Profiles change which capabilities and scores
matter most. Hardware simulation uses a mix of supplied values and defaults;
pass `--vram` and `--ram` when you know the actual capacities.

Memory estimates include the weights and KV cache, plus working buffers and
runtime overhead. Speed estimates use memory bandwidth and the active model
size. For mixture-of-experts models, the full weights still count towards memory.

These are planning estimates. ModelScout doesn’t run model inference to measure
tokens per second. Context length and the inference runtime can change the result
a lot. And a GPU appearing in the catalogue doesn’t mean its drivers or a given
model runtime will work on your system.

## The catalogue

[`assets/dataset.json`](assets/dataset.json) is the source catalogue. It currently
contains 207 CPU/SoC entries, 199 GPU entries including Apple integrated GPUs,
and 13 canonical models. The separate source lists are metadata, so their counts
don’t represent additional ranked models.

The benchmark figures are bundled records with source labels and dates. Their
presence here doesn’t prove that an exact quantized model was independently
tested. I’d treat those scores as provisional until the underlying result is
checked. See [the dataset notes](assets/README.md) for the limits.

The first run seeds a SQLite cache at
`${XDG_CACHE_HOME:-~/.cache}/modelscout/modelscout.db`. After editing the dataset,
reload it with:

```bash
./start.sh database validate assets/dataset.json
./start.sh refresh
```

`refresh` reloads the local file. It does not fetch the latest leaderboards.
Set `MODELSCOUT_DATASET` to an absolute file path to use your own catalogue.
Ordinary scans use local data; runtime detection may contact Ollama on localhost.

## Working on it

```text
assets/             Dataset and notes about its sources
scripts/            Shared launcher setup
src/                Python package and pyproject.toml
  modelscout/
    hardware/       Machine detection and overrides
    estimation/     Memory fit and speed estimates
    recommendation/ Ranking and explanations
    benchmarks/     Benchmark aggregation and source adapters
    dataset/        Loading and validation
    database/       SQLite cache
    models/         Model metadata and quantization
    sources/        Model hub adapters
    cli/            Terminal commands
    web/            FastAPI app and browser UI
tests/             Tests for the CLI, API and calculations
```

From the repository root, after the first launch:

```bash
uv pip install --python .venv/bin/python -e './src[dev]'
.venv/bin/python -m pytest tests
```

Without `uv`, run `.venv/bin/python -m ensurepip --upgrade` first, then
`.venv/bin/python -m pip install -e './src[dev]'`.
Python packaging lives in `src/` to keep the root small, so manual installs use
`pip install ./src`. Local notes, environment files and generated output stay
out of Git.
