### How the Entire System Works
  ModelScout is designed as an independent local AI decision engine. Below is the complete
  end-to-end walkthrough of how it probes hardware, sources information, calculates memory
  and speed, and delivers explainable recommendations.
    ┌────────────────────────────────────────────────────────┐
    │               Host Hardware Detection                  │
    │  (sysctl / system_profiler / /proc / nvidia-smi / etc) │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │           Normalized Hardware Representation           │
    │  (CPU cores, clock GHz, RAM, VRAM, unified bus BW)     │
    └──────────────────────────┬─────────────────────────────┘
                               │
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
    ┌──────────────────────────────┐ ┌──────────────────────────────┐
    │     Local SQLite Store       │ │    Optional Live Hubs        │
    │  (Seed dataset.json: 245     │ │  (Hugging Face GGUF, Ollama  │
    │   CPUs, 214 GPUs, models)    │ │   daemon, NVIDIA Build)      │
    └──────────────┬───────────────┘ └──────────────┬───────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
    ┌────────────────────────────────────────────────────────┐
    │              Multi-Source Benchmark Aggregator         │
    │   (LiveBench, Aider, Arena, recency & evidence tiers)  │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │            Architecture-Aware Memory Engine            │
    │  (Weights + KV Cache [GQA] + Activations + Overhead)   │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │             Bandwidth-Bound Speed Estimator            │
    │  (Bus GB/s ÷ active memory/tok → honest tok/s range)   │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │               0–100 Decision Scoring Engine            │
    │  (Quality + Fit + Speed + Capability + Evidence)       │
    └──────────────────────────┬─────────────────────────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
          Terminal CLI / JSON       Local Web UI (1234)
  ──────
  #### 1. Hardware Detection: How System Specs Are Found
  Detection is handled by native OS modules in :
  • macOS (macos.py):
      • CPU & Clock Speeds: Uses native POSIX sysctl -n machdep.cpu.brand_string and hw.ncpu
      to query core counts and chip model (e.g. Apple M4, Intel Core i9-9980HK).
      • Memory & Bandwidth: Reads hw.memsize for total RAM, then queries sysctl and hardware
      dictionaries to determine whether unified memory is active. Apple Silicon memory bus
      bandwidths are automatically attributed based on SoC family (e.g., M1/M2/M3/M4 base:
      100–150 GB/s, Pro: 150–200 GB/s, Max: 300–400 GB/s, Ultra: 800 GB/s).
      • GPU: Uses system_profiler SPDisplaysDataType and IOKit to determine the integrated or
      discrete graphics configuration, VRAM allocations, and display controllers.
      • Storage: Uses statvfs and diskutil to check available SSD capacity without calling
      external shell pipelines when possible.
      • Runtimes: Checks for Apple Metal framework availability via PyTorch/MPS or macOS core
      frameworks.
  • Linux (linux.py, nvidia.py, amd.py, intel.py):
      • CPU & Features: Parses /proc/cpuinfo for model names, base/boost frequencies, thread
      topologies, and vector instructions (AVX-512, AVX2, AMX, VNNI).
      • RAM: Reads /proc/meminfo (MemTotal, MemAvailable).
      • NVIDIA GPUs: Probes nvidia-smi (or native NVML bindings) for GPU model, compute
      capability, VRAM capacity, driver version, and CUDA support.
      • AMD GPUs: Probes rocm-smi, rocminfo, or /sys/class/drm for ROCm architecture and VRAM.
      • Intel GPUs: Scans /sys/class/drm and lspci for Arc discrete and Iris/UHD integrated
      GPUs.
      • Storage: Uses POSIX statvfs on the filesystem containing the models directory.
  Safety guarantee: Every detector fails gracefully. If a command like nvidia-smi is not
  found, the detector logs a notice and continues without crashing.
  ──────
  #### 2. Information Sourcing: Local Seed vs. API Calls
  • 100% Offline-First by Default:
      • ModelScout never requires an active Internet connection to evaluate hardware, compute
      token speeds, or recommend models.
      • The local database (dataset.json imported into SQLite) contains:
          • 245 CPUs (Apple Silicon M1–M4/M5, Intel Core 6th Gen to 14th Gen, Core Ultra Lunar Lake & Arrow Lake, Xeon 6, AMD Ryzen 1000–9000 & X3D, Strix Halo, Threadripper, EPYC Turin).
          • 214 GPUs (NVIDIA RTX 50 Blackwell series, B200, GB200, RTX 40 Ada, Hopper H100/H200, A100, RTX 30/20, GTX 10 series, AMD RX 6000/7000/8000 & Instinct MI300, Intel Arc Battlemage/Alchemist & Gaudi).
          • 107 Curated canonical open-weight models with 318 quantized GGUF variants (Llama 3.3/3.2, Qwen 2.5/3, DeepSeek R1/V3, Gemma 2/4, Mistral Large/Nemo, Phi-4) spanning Q4_K_M, Q5_K_M, Q8_0, and FP16.
          • Standard benchmark evaluations from verified leaderboards (LiveBench, Aider, Chatbot Arena, Artificial Analysis).
  • Live Data Ingestion (Opt-in via --refresh or background discover):
      • Hugging Face: Queries the Hugging Face Hub API (https://huggingface.co/api/models)
      for newly uploaded GGUF repositories, active downloads, parameter counts, and model
      card metadata.
      • Ollama: Connects to the local daemon (http://localhost:11434/api/tags) to inspect
      already downloaded models, and verifies tags against the Ollama public registry.
      • NVIDIA Build: Normalizes identifiers to cloud-deployable NIM equivalents
      (https://build.nvidia.com/).
      • Caching: Live updates are persisted into SQLite (~/.cache/modelscout/ or local
      data/modelscout.db) with configurable TTLs (e.g. 6 hours for models, 24 hours for
      benchmarks).
  ──────
  #### 3. What "Evidence: Direct" Means vs. "Transferred"

  ModelScout tracks provenance for every benchmark score in :

   Evidence Tier        | Meaning & Treatment
  ----------------------|--------------------------------------------------------------------
   Direct               | Exact model scale and weights were benchmarked live on official
                        | evaluation suites (e.g. LiveBench, Aider code-editing, or Chatbot
                        | Arena). No speculative extrapolation or parameter scaling was
                        | applied. Carries a 1.0× confidence factor.
   Variant              | Evaluated on an equivalent instruction-tuned or chat derivative of
                        | the same base model. Carries a 0.95× factor.
   Base / Transferred   | The benchmark was executed on the unquantized 16-bit reference
                        | model. The score is mathematically transferred to the candidate
                        | using calibrated quantization loss penalties (e.g., Q8_0: -0.5%,
                        | Q5_K_M: -1.5%, Q4_K_M: -3.5%, Q2_K: -14.0%). Carries a 0.88×
                        | factor.
   Lineage-Interpolated | Estimated across models in the same lineage. ModelScout validates
                        | parameter bounds to prevent invalid inheritance (e.g., a 7B model
                        | can never inherit a 70B model's score). Carries a 0.75× factor.
   Self-Reported        | Author-reported results from model cards. Discounted to 0.60×
                        | factor.

  Recency & Lineage Demotion:
  Older frozen benchmark leaderboards are time-decayed using an exponential half-life (180
  days). This ensures that older architectures do not outscore modern, superior models simply
  because the older model has legacy leaderboard entries.
  ──────
  #### 4. Memory Footprint Calculation (The 4 Pools)

  Unlike simplistic tools that merely check file size on disk, memory.py calculates the real
  runtime memory:

    Total Memory = Weights + KV Cache + Activation Buffers + Driver Overhead

  1. Weights Memory:

                                   Bits Per Weight
    Weights (Bytes) = Parameters × ───────────────
                                          8

  Where bit depths are: FP16 = 16.0, Q8_0 = 8.5, Q6_K = 6.56, Q5_K_M = 5.5, Q4_K_M = 4.5,
  Q3_K_M = 3.44.
  2. KV Cache (Accounting for GQA/MQA):
  For a given context length C, layer count L, hidden dimension H, attention heads A, and
  key-value heads K:

                                   ⎛ K ⎞
    KV Cache (Bytes) = 2 × L × H × ⎜───⎟ × C × 2 bytes (FP16)
                                   ⎝ A ⎠

  Models using Grouped-Query Attention (such as Llama-3 with K/A = 8/64 = 0.125) require
  significantly less KV cache than traditional multi-head attention.
  3. Activation Buffers & Context Overhead:
  Dynamically allocated buffers for intermediate layer activations:

                     ⎛        Parameters          ⎞     C
    Activations ≈ min⎜1.5 GB, ────────── × 0.04 GB⎟ + ───── × 0.3 GB
                     ⎝           10⁹              ⎠   32768

  4. Runtime Driver Overhead:
  Fixed 0.5–0.8 GB reserve for Metal/CUDA command queues and runtime allocations.

  Hardware Fit Classification:

  • FULL_GPU: Entire working set fits in discrete GPU VRAM.
  • UNIFIED_MEMORY: Fits in high-speed unified memory (Apple Silicon) with zero bus offload
  penalty.
  • PARTIAL_OFFLOAD: Weights split between GPU VRAM and system RAM via PCIe (penalized in
  speed).
  • CPU_ONLY: Runs entirely on host CPU and system RAM.
  • UNUSABLE: Exceeds physical system memory or generates < 1 token/second.
  ──────
  #### 5. Performance & Generation Speed Estimation

  Autoregressive LLM generation is fundamentally memory bandwidth bound during the decoding
  phase. In speed.py:

                          Memory Bandwidth (GB/s)
    Theoretical tok/s = ────────────────────────────
                        Active Weight Footprint (GB)

  • Mixture of Experts (MoE): Speed is calculated using active parameters (e.g. DeepSeek-
  V3/R1 uses 37B active parameters out of 671B total), while memory fit checks the total
  parameters.
  • Hardware Efficiency & Quantization Penalty: A realistic hardware utilization efficiency
  factor (η ≈ 0.65–0.78) accounts for kernel launch latencies, GEMV dequantization overhead,
  and memory controller saturation.
  • Confidence Intervals: Rather than outputting artificial precision (e.g. 24.8192 tok/s),
  ModelScout outputs calibrated ranges:

    Speed Range = [tok/s × 0.85, tok/s × 1.15] ⟹∼ 21–27 tok/s
  ──────
  #### 6. The 0–100 Explainable Decision Engine

  The final ranking score in scoring.py balances six core pillars:

    Score =  w      · BenchmarkQuality
              bench
          +  w    · HardwareFit
              fit
          +  w      · SpeedScore
              speed
          +  w    · ModelCapability
              cap
          +  w     · EvidenceConfidence
              evid
          +  w    · RuntimeSupport
              run

  • General Profile: w_{bench} = 0.28, w_{fit} = 0.24, w_{speed} = 0.20, w_{cap} = 0.14,
  w_{evid} = 0.08, w_{run} = 0.06.
  • Coding Profile: Prioritizes Aider and code generation benchmarks (w_{bench} = 0.35).
  • Fast Profile: Penalizes candidates below 20 tok/s (w_{speed} = 0.38).
  • Reasoning Profile: Boosts chain-of-thought architectures (w_{cap} = 0.25, w_{bench} = 0.
  30).

  Every recommendation generates clear human explanations ("Fits comfortably in available
  memory", "Strong current benchmark results", "Q4_K_M artifact verified"), and excluded
  models state the exact reason ("Exceeds memory: Requires 21.5 GB, system has 16.0 GB
  available").
  ──────
  ──────
  ### 7. Modular Package Architecture & Repository Layout

  Following modern best practices and clean separation of concerns, ModelScout is structured with explicit package boundaries, modular dataset storage, and backward-compatible shims:

  ```text
  ├── assets/                 # Modular offline datasets & source provenance
  │   ├── cpus.json           # 245 curated CPUs (Apple, AMD, Intel)
  │   ├── gpus.json           # 214 curated GPUs (NVIDIA, AMD, Intel)
  │   ├── models.json         # 107 canonical models & 318 quantized artifacts
  │   └── dataset.json        # Combined fallback registry
  ├── scripts/                # Shared developer launchers & test utilities
  │   ├── start.sh            # Auto-detecting CLI & Web UI launcher
  │   ├── test.sh             # Pytest test suite runner
  │   ├── build.sh            # Wheel and sdist build script
  │   ├── verify.sh           # End-to-end verification script
  │   └── benchmark.sh        # Estimation engine benchmark suite
  ├── src/                    # Python distribution root
  │   ├── pyproject.toml      # Hatchling packaging config for 'modelscout-llm' (v0.1.1)
  │   ├── hatch_build.py      # Custom build hook packaging assets/ into wheels
  │   ├── README.md           # PyPI project description and quickstart docs
  │   └── modelscout/         # Core Python package
  │       ├── hardware/       # Machine detection & synthetic simulation engine
  │       │   ├── detector.py     # Orchestrates GPU, CPU, RAM, and runtime detection
  │       │   ├── apple.py        # Apple Silicon Metal & unified memory detection
  │       │   ├── nvidia.py       # NVIDIA GPU probe via NVML with nvidia-smi fallback
  │       │   ├── amd.py          # AMD GPU probe via ROCm SMI and Linux DRM/sysfs
  │       │   ├── intel.py        # Intel Arc & integrated GPU detection
  │       │   ├── macos.py        # macOS sysctl & IOKit hardware probes
  │       │   ├── linux.py        # Linux /proc/cpuinfo and /proc/meminfo probes
  │       │   ├── windows.py      # Windows WMI and PowerShell hardware probes
  │       │   ├── cpu.py          # CPU core topology, frequencies, AVX2/AVX-512 flags
  │       │   ├── memory.py       # Physical RAM, usable budget, and disk free space
  │       │   ├── gpu_simulator.py# Multi-GPU simulation (--gpu "2x RTX 4090", comma-separated)
  │       │   ├── gpu_db.py       # Static bandwidth & compute capability resolution
  │       │   └── types.py        # HardwareInfo, SystemHardware, CpuInfo, GpuInfo
  │       ├── estimation/     # Memory fit and generation speed estimation engine
  │       │   ├── memory.py       # The 4-pool memory calculation (Weights + KV + Act + Driver)
  │       │   ├── kv_cache.py     # Architecture-aware KV cache (MHA, GQA, MQA, FP16/quantized)
  │       │   ├── speed.py        # Bandwidth-bound decoding speed (GB/s ÷ active footprint)
  │       │   ├── fit.py          # Fit classification (FULL_GPU, UNIFIED, PARTIAL, CPU_ONLY)
  │       │   └── confidence.py   # Statistical confidence ranges (±15% calibrated bounds)
  │       ├── recommendation/ # Ranking, profiles, and explainable recommendations
  │       │   ├── scoring.py      # 0–100 composite scoring engine across 6 pillars
  │       │   ├── profiles.py     # User profiles (general, coding, fast, reasoning)
  │       │   ├── explanation.py  # Human-readable justification generation
  │       │   └── ranking.py      # Candidate ordering and exclusion tracking
  │       ├── benchmarks/     # Multi-source benchmark aggregation & normalization
  │       │   ├── aggregation.py  # Weighted benchmark scoring across sources
  │       │   ├── engine.py       # Benchmark evaluation engine
  │       │   ├── evidence.py     # Evidence tiers (Direct, Variant, Transferred, Lineage)
  │       │   ├── recency.py      # Exponential time-decay weighting (180-day half-life)
  │       │   ├── normalization.py# Benchmark score scaling to 0–100 scale
  │       │   └── sources/        # Source adapters (LiveBench, Aider, Arena, Leaderboard)
  │       ├── dataset/        # Dataset loading, validation, and live synchronization
  │       │   ├── loader.py       # Multi-file dataset loader (cpus, gpus, models)
  │       │   ├── schema.py       # Pydantic schema validation for catalog records
  │       │   ├── validator.py    # Catalog integrity and constraint checks
  │       │   ├── importer.py     # SQLite batch importer
  │       │   ├── updater.py      # Live open-endpoint synchronization engine (<10s)
  │       │   └── provenance.py   # Origin metadata and verification timestamps
  │       ├── database/       # SQLite storage & cache repository
  │       │   ├── repository.py   # Database access layer with mtime auto-reloading
  │       │   └── models.py       # Relational table schema definitions
  │       ├── models/         # Model metadata, variants, and quantization
  │       │   ├── parser.py       # Model tag and quantization format parser
  │       │   ├── parameters.py   # Architecture parameter resolver (total vs active)
  │       │   ├── types.py        # ModelInfo, GGUFVariant, ModelFamily
  │       │   ├── grouper.py      # Model family grouping and architecture grouping
  │       │   └── gguf.py         # GGUF quantization tiers, bytes/weight, quality loss
  │       ├── sources/        # Model hub and runtime integration adapters
  │       │   ├── huggingface.py  # HuggingFace Hub API search and download metrics
  │       │   ├── ollama.py       # Local Ollama daemon tags and inspection
  │       │   └── nvidia.py       # NVIDIA Build NIM mapping
  │       ├── cli/            # Typer-powered terminal CLI interface
  │       │   ├── main.py         # Main Typer entry point and subcommands
  │       │   ├── commands.py     # Command implementations (scan, run, plan, snippet, upgrade, update, web)
  │       │   └── output.py       # Rich terminal tables, panels, and Markdown export
  │       ├── web/            # Local FastAPI Web Application (default: http://localhost:1234)
  │       │   ├── app.py          # FastAPI application & lifespan background updater
  │       │   ├── api.py          # REST API endpoints (/api/scan, /api/models, /api/hardware)
  │       │   └── static/         # HTML5 responsive UI, dark theme, and interactive client
  │       ├── data/           # Backward-compatibility registries (framework, gpu, lineage, quant)
  │       ├── engine/         # Backward-compatibility calculation shims
  │       └── output/         # Backward-compatibility terminal formatting shims
  ├── tests/                  # 45 passing unit and integration tests
  └── .github/workflows/      # CI/CD and deployment workflows
      └── publish.yml         # PyPI Trusted Publishing via OIDC
  ```

  ### 8. Multi-File Dataset & Dynamic Mtime Auto-Sync

  - **Modular Dataset Architecture**:
    Instead of a single monolithic file, curated data is split into specialized JSON assets under `assets/`:
    • `assets/cpus.json`: 245 curated CPUs spanning Apple Silicon (M1 through M4 Max/Ultra, M5), Intel (Core 6th–14th Gen, Core Ultra Lunar Lake / Arrow Lake, Xeon 6), and AMD (Ryzen 1000–9000 & X3D, Strix Halo APU, Turin EPYC, Threadripper).
    • `assets/gpus.json`: 214 curated GPUs spanning NVIDIA Blackwell RTX 50 series (5090, 5080, 5070, 5060), B200/GB200, Ada Lovelace RTX 40 series, Hopper H100/H200, A100, RTX 30/20, GTX 10 series, AMD Radeon RX 6000/7000/8000, Instinct MI300X/A, and Intel Arc Battlemage (B580/B570), Alchemist, and Gaudi 2/3.
    • `assets/models.json`: 107 canonical models with 318 quantized GGUF variants (DeepSeek R1/V3, Qwen 2.5/3, Llama 3.3/3.2, Mistral Large/Nemo, Gemma 2/4, Phi-4) across Q4_K_M, Q5_K_M, Q8_0, and FP16 formats.
    • `assets/dataset.json`: Unified master dataset serving as a zero-dependency offline fallback.

  - **Automatic Mtime-Based Database Reseeding**:
    The SQLite cache (`DatabaseRepository`) guarantees that manual edits to dataset files take effect immediately:
    • `DatabaseRepository.init_db()` records the `last_dataset_mtime` in the `catalog_sync_meta` table.
    • Every time ModelScout starts (CLI or Web UI), `_get_dataset_mtime()` inspects the filesystem modification times (`st_mtime`) of `assets/models.json`, `assets/cpus.json`, `assets/gpus.json`, and `assets/dataset.json`.
    • If any dataset file has a newer `st_mtime` than `last_dataset_mtime`, the SQLite store automatically and idempotently re-seeds itself from disk without requiring manual cache wipes, `--force` flags, or database recreation.

  - **Dual Background & On-Demand Synchronization**:
    • **Non-Blocking Background Sync**: On application startup, `start_background_model_update(timeout=8.0)` spawns a daemon thread that queries open endpoints (Hugging Face trending models and local Ollama daemon) within a strict 10-second timeout guard. If updated download counts, likes, or tags are found, they are written back directly into `assets/models.json` and synchronized with SQLite.
    • **On-Demand Manual Sync**: Running `modelscout update` executes an immediate, synchronous update cycle with interactive Rich progress bars, reporting exact counts of discovered models and updated metadata.

  ### 9. PyPI Distribution, Packaging & GitHub Actions (OIDC)

  - **Distribution Naming & Binary Entry Point**:
    • **PyPI Package Name**: `modelscout-llm` (version `0.1.1`). The base name `modelscout` was previously registered on PyPI, so the package was renamed to `modelscout-llm` for PyPI distribution.
    • **Executable CLI Command**: The CLI command remains `modelscout`, mapped via `[project.scripts]` in `src/pyproject.toml`:
      ```toml
      [project.scripts]
      modelscout = "modelscout.cli.main:app"
      ```

  - **Build Hook & Asset Packaging**:
    • ModelScout uses Hatchling as its build backend.
    • To bundle the offline datasets into wheels and source distributions, `src/hatch_build.py` and `src/pyproject.toml` use `[tool.hatch.build.targets.sdist.force-include]` and wheel package mapping so that `assets/` files are included directly in the installable distribution.
    • **PyPI Readme Requirement**: Hatchling requires `readme` paths to reside inside the build root. `src/README.md` provides the full project description, quickstart instructions, and usage guide displayed on pypi.org.

  - **GitHub Actions Trusted Publishing (OIDC)**:
    • Workflow location: `.github/workflows/publish.yml`.
    • Triggered automatically on GitHub release publication or manually via `workflow_dispatch`.
    • Configured with `permissions: id-token: write`, enabling PyPI Trusted Publishing via OpenID Connect (OIDC) without long-lived API tokens or passwords.
    • Employs `pypa/gh-action-pypi-publish@release/v1` with `packages-dir: src/dist/` and `skip-existing: true` so re-runs or partial uploads succeed gracefully.

  - **PyPI Release Immutability**:
    • PyPI strictly forbids overwriting or re-uploading an existing version; attempts will fail with HTTP 400 (`File already exists`).
    • For every new release, increment the version string across:
      1. `src/pyproject.toml` (`version = "x.y.z"`)
      2. `src/modelscout/__init__.py` (`__version__ = "x.y.z"`)
      3. `src/modelscout/web/app.py` (`APP_VERSION = "x.y.z"`)
      4. `src/modelscout/web/api.py` (`APP_VERSION = "x.y.z"`)

  ### 10. Zero-Install Execution with `uvx`

  ModelScout can be executed instantly on any system without cloning the repository or manually managing Python virtual environments:

  ```bash
  # Auto-detect hardware and display top recommended models
  uvx modelscout-llm@latest

  # Launch the local Web UI dashboard on port 1234
  uvx modelscout-llm@latest web --port 1234

  # Interactive chat with the best fitting model
  uvx modelscout-llm@latest run llama

  # Generate runner code snippets (Ollama, vLLM, llama.cpp)
  uvx modelscout-llm@latest snippet llama --runner ollama

  # Inspect detected host CPU, GPU, RAM, and runtime capabilities
  uvx modelscout-llm@latest hardware

  # Memory & quantization requirements plan for a model
  uvx modelscout-llm@latest plan "llama 3 70b"

  # Simulate GPU upgrades and evaluate capability gains
  uvx modelscout-llm@latest upgrade

  # Synchronize model catalogue from open endpoints
  uvx modelscout-llm@latest update
  ```

  ### 11. Verification & Operational Commands

  - **Local Launcher**:
    ```bash
    ./start.sh          # Launches CLI scan
    ./start.sh web      # Launches local Web UI on http://localhost:1234
    ```
  - **Developer Scripts**:
    ```bash
    ./scripts/test.sh   # Executes full pytest suite
    ./scripts/verify.sh # Verifies CLI, API, memory math, and hardware detection
    ./scripts/build.sh  # Builds wheel and source distributions via uv
    ```
  - **Test Suite Status**: All 45 automated unit and integration tests passing.
  - **Git State**: Zero unreviewed commits. All changes remain in the local working tree for manual review and push.
