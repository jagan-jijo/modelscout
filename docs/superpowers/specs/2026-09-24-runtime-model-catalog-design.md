# Runtime Model Catalog Design

**Date:** 2026-09-24
**Status:** Approved for implementation

## Goal

Extend ModelScout's existing model catalogue with separate, current lists of Ollama-, AirLLM-, and Colibri-compatible models while preserving all current commands, ranking behavior, and offline fallback behavior.

## Compatibility model

- **Ollama:** Preserve exact existing `ollama_name` tags and `ollama pull` commands.
- **AirLLM:** Curate models documented by AirLLM's `AutoModel.from_pretrained` interface. Display the exact Hugging Face repository ID and a ready-to-run Python install/load snippet.
- **Colibri:** Curate only the model families supported by the current Colibri documentation. Display both the source checkpoint ID and, when applicable, the exact Colibri container ID. Do not imply that arbitrary dense models are compatible.

## Curated records

The initial AirLLM list is:

1. `Qwen/Qwen3-32B`
2. `Qwen/Qwen3.8-27B`
3. `Qwen/Qwen3.8-Flash-Next`
4. `Qwen/Qwen3-235B-A22B`
5. `deepseek-ai/DeepSeek-V3`
6. `moonshotai/Kimi-K3`

The initial Colibri list is:

1. `moonshotai/Kimi-K3`
2. `deepseek-ai/DeepSeek-V4-Flash`
3. `Qwen/Qwen3.8-Flash-Next`
4. `Qwen/Qwen3.6-35B-A3B` with `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64` as the exact Colibri container
5. `Justvugg/GLM-5.3-colibri-int4-g64`
6. `mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp` as the 5,000-download fallback

The registry stores source and container IDs separately so a low-download conversion artifact does not erase a popular source model's eligibility.

## Popularity and freshness

- Query the Hugging Face API for exact model IDs and enrich records with `downloads`, `likes`, `createdAt`, and `lastModified`.
- Cache metadata with the existing six-hour model-cache policy and retain the bundled snapshot when the network is unavailable.
- Select records with at least 10,000 downloads first. If a runtime cannot produce the requested 5–6 records at that threshold, allow records with at least 5,000 downloads for that runtime only.
- Rank by compatibility, recency, and downloads; never present stale metadata as a live measurement without displaying the check timestamp.
- Do not download weights or launch inference as part of catalogue generation.

## Runtime-specific information

Each record includes runtime, model family/architecture type, exact model ID, install command or loader snippet, source URL, popularity metadata, and runtime notes.

- AirLLM notes distinguish its one-layer-at-a-time GPU placement and required model-specific dependencies. Reported VRAM figures are labelled as source documentation, not universal guarantees.
- Colibri notes distinguish streamed expert storage, RAM/VRAM requirements, optional conversion, and whether the checkpoint is directly loadable. Source-reported measured tok/s values are shown only when available; otherwise the output says that a local benchmark is required.
- Existing Ollama speed estimates remain the source of truth for Ollama rows.

## Interfaces

- Add a runtime catalog loader and typed records under `models/`, with a JSON registry under `assets/`.
- Add a `runtimes` CLI command and a `--runtime`/JSON-capable rendering path without removing the existing `models`, `benchmarks`, or scan commands.
- Add a read-only `/api/runtime-models` endpoint returning the three grouped sections and metadata.
- Render the same grouped data in the web catalogue, with separate Ollama, AirLLM, and Colibri sections and exact IDs visible.

## Compatibility and failure behavior

- Existing imports and commands remain valid.
- Offline mode uses bundled records and marks metadata as cached/snapshot data.
- A missing or invalid runtime record produces a clear warning and does not prevent existing recommendations from rendering.
- No new inference runtime dependency is installed by ModelScout; AirLLM and Colibri remain optional tools.

## Verification

- Unit tests cover exact IDs, threshold selection, 10k-to-5k fallback, cache/offline behavior, and JSON serialization.
- CLI tests verify all three section headings and exact model IDs in terminal and JSON output.
- API tests verify grouped response shape and no model download side effects.
- Run the full existing pytest suite and an offline CLI smoke test.
