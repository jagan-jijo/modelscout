# ModelScout (`modelscout-llm`)

> **Find local AI models that fit your hardware, with accurate memory and generation speed estimates.**

Downloading a 10GB–50GB model file only to discover that it overflows your VRAM or crawls at 1 token per second is a frustrating experience. **ModelScout** inspects your machine and tells you exactly what open-source models will run smoothly — before you spend hours downloading weights.

Whether you're running on **Apple Silicon (M1–M4)**, **NVIDIA RTX**, **AMD Radeon / ROCm**, **Intel Arc**, or **CPU-only**, ModelScout calculates real-world weights, KV cache requirements, and memory-bandwidth token generation speeds.

---

## ⚡ Quick Start with `uv` (Zero Installation)

You don't even need to clone a repository or set up a virtual environment. If you have [`uv`](https://github.com/astral-sh/uv) installed, you can run ModelScout instantly via `uvx`:

```bash
# Scan your hardware and get the best matching models
uvx modelscout-llm@latest

# Launch the interactive local Web Dashboard (http://localhost:1234)
uvx modelscout-llm@latest web

# Inspect your detected GPU/CPU specs and AI capability score
uvx modelscout-llm@latest hardware
```

---

## 📦 Install via `pip`

You can also install ModelScout globally or into any Python 3.11+ environment:

```bash
pip install modelscout-llm
```

Once installed, the `modelscout` command is available everywhere:

```bash
# Run the scanner
modelscout

# Launch the Web UI
modelscout web

# Target a specific workload
modelscout --profile coding
modelscout --profile reasoning
modelscout --profile vision

# Plan hardware requirements for a model
modelscout plan "llama 3 70b"

# Generate ready-to-run Python code snippet
modelscout snippet "llama 3" --runner ollama

# Start an interactive chat session
modelscout run "llama-3.2-1b"
```

---

## 🛠️ Key Capabilities

- **Native Hardware Probing**: Automatically detects Apple Silicon unified memory & bandwidth, NVIDIA NVML/CUDA, AMD ROCm, Intel Arc, and CPU AVX/NEON instruction sets.
- **Architecture-Aware Memory Engine**: Accurately models weights + GQA/MQA KV cache footprints + activation buffers + framework overhead.
- **Bandwidth-Bound Speed Estimates**: Derives honest tokens/second ranges based on your system's actual memory bus bandwidth (GB/s).
- **Interactive Web Interface**: Complete browser dashboard running locally on port `1234`, featuring hardware autocompletion, real-time filtering, and side-by-side model comparison.
- **Hardware Simulation**: Test potential upgrades before buying hardware (e.g. `modelscout --gpu "2x RTX 4090"` or `modelscout upgrade`).
- **Flexible Formats**: Export clean GitHub-flavored Markdown (`-m`) or strict JSON (`--json`) for scripting and automation.

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/jagan-jijo/modelscout](https://github.com/jagan-jijo/modelscout)
- **Author**: Jagan Jijo ([Portfolio](https://jagan-jijo.github.io/portfolio/))
- **License**: MIT
