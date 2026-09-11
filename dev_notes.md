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
    │  (Seed dataset.json: 207     │ │  (Hugging Face GGUF, Ollama  │
    │   CPUs, 199 GPUs, models)    │ │   daemon, NVIDIA Build)      │
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
          • 207 CPUs (Apple Silicon M1–M4, Intel Core 6th Gen to 14th Gen & Arrow Lake, Xeon,
          AMD Ryzen 1000–9000, Threadripper, EPYC).
          • 199 GPUs (NVIDIA GTX 1060+, RTX 20/30/40/50 series, Quadro, RTX Ada, H100/H200,
          A100, B200, GB200, AMD RX & Instinct, Intel Arc & Gaudi).
          • Curated open-weight models across families (Llama 3.x, Qwen 2.5/3, DeepSeek-R1
          Distills, Gemma 2/4, Mistral, Phi-4) with GGUF quantizations (Q4_K_M, Q5_K_M, Q8_0,
          FP16).
          • Standard benchmark evaluations from verified leaderboards.
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
  ### 7. Modular Architecture & Directory Layout

  Following modern best practices and clean separation of concerns, the core engine in `src/modelscout` provides:

  ```text
  ├── cli.py              # Typer CLI entry point: main, plan, run, snippet, hardware, web, update
  ├── constants.py        # Backward-compatible exports for curated registries and thresholds
  ├── data/               # GPU, quantization, framework, and lineage registries
  │   ├── framework.py    # Framework memory overheads and compute capability limits
  │   ├── gpu.py          # Curated GPU specs, bandwidth tables, and NVIDIA compute capabilities
  │   ├── lineage.py      # Generation lineage version mappings and bonuses/penalties
  │   └── quantization.py # Quantization tiers, bytes per weight, quality penalties
  ├── web/                # Local FastAPI Web UI (default: http://localhost:1234)
  │   ├── app.py          # FastAPI application and lifespan background updater
  │   ├── api.py          # API endpoints (/api/scan, /api/models, /api/hardware)
  │   └── static/         # HTML5 dashboard, styles, and interactive client
  ├── hardware/           # Hardware detection & synthetic simulation engine
  │   ├── detector.py     # Orchestrates GPU, CPU, RAM, and runtime detection
  │   ├── nvidia.py       # NVIDIA GPU probe via NVML with nvidia-smi fallback
  │   ├── amd.py          # AMD GPU probe via ROCm SMI and Linux DRM/sysfs
  │   ├── apple.py        # Apple Silicon Metal and Asahi Linux devicetree detection
  │   ├── cpu.py          # CPU brand name, core count, AVX2 / AVX-512 detection
  │   ├── memory.py       # Physical RAM, usable budget, and disk free space
  │   ├── gpu_simulator.py# Multi-GPU simulation (--gpu "2x RTX 4090", comma-separated, catalog fallback)
  │   ├── gpu_db.py       # Static bandwidth & compute capability resolution
  │   └── types.py        # GPUInfo, HardwareInfo, SystemHardware, CpuInfo, MemoryInfo
  ├── models/             # Model sourcing, metadata parsing, and benchmark indexing
  │   ├── fetcher.py      # HuggingFace hub API fetcher with retry & sliding window awareness
  │   ├── benchmark.py    # Arena ELO, Open LLM Leaderboard, recency weighting & evidence lookup
  │   ├── grouper.py      # Model family grouping by base_model and normalized architecture
  │   ├── cache.py        # Local JSON cache with 6-hour TTL
  │   └── types.py        # ModelInfo, GGUFVariant, ModelFamily
  ├── engine/             # Execution and compatibility calculation engine
  │   ├── vram.py         # VRAM = weights + KV cache (FP16/GQA) + activation + overhead
  │   ├── compatibility.py# Full GPU, partial offload, CPU-only, disk, and compute warnings
  │   ├── performance.py  # Bandwidth-bound decoding speed and confidence ranges
  │   ├── quantization.py # Bytes per weight, quality loss, and non-GGUF format inference
  │   ├── ranker.py       # Scoring orchestrator, evidence filter, and profile matching
  │   └── types.py        # CompatibilityResult, FitType
  └── output/             # Rendering and export surfaces
      ├── ranking.py      # Rich terminal tables and top-pick confidence panels
      ├── json_output.py  # Machine-readable JSON output for CLI automation
      ├── plan.py         # Model hardware requirements breakdown across quant levels
      ├── upgrade.py      # GPU upgrade comparison table and verdict analysis
      ├── markdown.py     # GitHub-Flavored Markdown table export (-m / --markdown)
      └── display.py      # Compatibility re-export shim for output surfaces
  ```

  ### 8. Multi-File Dataset & Background Updater
  - **Dataset Split**: Located in `assets/`, cleanly separated into `cpus.json`, `gpus.json`, and `models.json`, unified via `dataset.json`.
  - **Catalogue Breadth**: Covers Intel (Lunar Lake, Arrow Lake, Xeon 6), AMD (Ryzen 9000/X3D, Strix Halo, Turin EPYC), NVIDIA (Blackwell RTX 50 series, B200, Ada Lovelace, Hopper), Apple Silicon (M1 through M4 Max/Ultra, M5), and latest models (DeepSeek R1/V3, Qwen 2.5/3, Llama 3.3/3.2, Mistral, Gemma 2/3/4, Phi-4).
  - **Fast Open API Sync**: `start_background_model_update(timeout=8.0)` runs in a non-blocking daemon thread on startup. If open endpoints take < 10 seconds, they enrich the local catalogue. Offline assets serve as an immediate zero-latency fallback.

  ### 9. Verification Summary
  - **CLI Scan**: `modelscout` or `./start.sh`
  - **One-Command Chat**: `modelscout run llama`
  - **Code Snippets**: `modelscout snippet llama --runner ollama`
  - **Hardware Inspection**: `modelscout hardware`
  - **Hardware Plan**: `modelscout plan "llama 3 70b"`
  - **GPU Upgrade Plan**: `modelscout upgrade`
  - **Catalog Sync**: `modelscout update`
  - **Web Dashboard**: `modelscout web --port 1234`
  - **Test Suite**: `pytest` -> All 45 tests passing.
  - **Git State**: Zero commits made. Ready for user's manual review and push.
