# Runtime Model Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separate, popularity-filtered Ollama, AirLLM, and Colibri model lists with exact install identifiers, current Hugging Face metadata, CLI output, and web display without changing existing recommendation behavior.

**Architecture:** Keep compatibility facts in a small bundled `assets/runtime_models.json` registry. A focused `models/runtime_catalog.py` loader resolves exact model metadata through the existing Hugging Face client/cache, applies the 10,000-download primary threshold with a per-runtime 5,000-download fallback, and returns plain serializable records. CLI and FastAPI consume the same loader; the web client renders the grouped response.

**Tech Stack:** Python 3.11+, Pydantic/datatypes already used by the project, `httpx`, Typer/Rich, FastAPI, vanilla HTML/CSS/JavaScript, pytest.

---

### Task 1: Add the runtime registry and typed loader

**Files:**
- Create: `assets/runtime_models.json`
- Create: `src/modelscout/models/runtime_types.py`
- Create: `src/modelscout/models/runtime_catalog.py`
- Modify: `src/modelscout/models/__init__.py` only if exports are needed
- Test: `tests/test_runtime_catalog.py`

- [ ] **Step 1: Write failing loader and selection tests.**

Cover the exact IDs from the approved design, the three runtime keys, serialization, and the per-runtime fallback rule. Use a small in-memory record fixture rather than the network.

```python
def test_select_runtime_models_prefers_ten_thousand_downloads():
    records = [
        RuntimeModel(runtime="airllm", model_id="Qwen/Qwen3-32B", downloads=10_000),
        RuntimeModel(runtime="airllm", model_id="Qwen/Qwen3.8-27B", downloads=9_999),
    ]
    selected = select_runtime_models(records, runtime="airllm", limit=1, primary=10_000, fallback=5_000)
    assert [m.model_id for m in selected] == ["Qwen/Qwen3-32B"]


def test_select_runtime_models_uses_fallback_only_when_needed():
    records = [
        RuntimeModel(runtime="colibri", model_id="fallback", downloads=5_001),
        RuntimeModel(runtime="colibri", model_id="primary", downloads=12_000),
    ]
    selected = select_runtime_models(records, runtime="colibri", limit=2, primary=10_000, fallback=5_000)
    assert [m.model_id for m in selected] == ["primary", "fallback"]


def test_runtime_record_serializes_exact_install_id():
    record = RuntimeModel(runtime="airllm", model_id="Qwen/Qwen3-32B", downloads=10_000)
    payload = record.to_dict()
    assert payload["model_id"] == "Qwen/Qwen3-32B"
    assert payload["runtime"] == "airllm"
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `.venv/bin/pytest tests/test_runtime_catalog.py -q`
Expected: FAIL because the runtime modules do not exist.

- [ ] **Step 3: Add typed records and the bundled registry.**

`RuntimeModel` must contain: `runtime`, `model_id`, `container_id`, `display_name`, `family`, `architecture`, `parameters`, `active_parameters`, `install_command`, `huggingface_url`, `downloads`, `likes`, `created_at`, `last_modified`, `reported_vram_gb`, `ram_gb`, `disk_gb`, `benchmark`, `notes`, and `source`. `to_dict()` must preserve the exact `model_id` and `container_id` strings and expose `popularity_fallback` when the source record was selected through the 5,000 threshold.

The JSON must include the six approved AirLLM IDs, the six approved Colibri source/container pairs, and six current Ollama tags already present in `assets/models.json` (`qwen3:8b`, `qwen3:32b`, `deepseek-r1:7b`, `llama3.3:70b`, `gemma3:12b`, and `mistral-small3.1`). Each record must include its official source URL and the existing exact Ollama pull command for Ollama records.

- [ ] **Step 4: Implement the loader and deterministic selector.**

Expose:

```python
def load_runtime_models(*, enrich: bool = True, limit: int = 6) -> dict[str, list[RuntimeModel]]: ...
def select_runtime_models(records, *, runtime: str, limit: int, primary: int = 10_000, fallback: int = 5_000) -> list[RuntimeModel]: ...
```

Resolve the asset from checkout and installed-package paths, matching the existing dataset loader. Never raise for a missing optional network field. Sort eligible records by `last_modified` descending, then `downloads` descending, then exact ID for stable output. Apply the fallback only when fewer than `limit` records pass the primary threshold.

- [ ] **Step 5: Run the focused tests and verify they pass.**

Run: `.venv/bin/pytest tests/test_runtime_catalog.py -q`
Expected: PASS.

---

### Task 2: Enrich exact Hugging Face metadata with bounded caching

**Files:**
- Modify: `src/modelscout/sources/huggingface.py`
- Modify: `src/modelscout/models/runtime_catalog.py`
- Test: `tests/test_runtime_catalog.py`

- [ ] **Step 1: Add failing tests for exact-ID enrichment and offline fallback.**

Mock the HTTP client response for `Qwen/Qwen3-32B` with `downloads`, `likes`, `createdAt`, and `lastModified`, then assert those fields replace bundled snapshot values. Mock a network exception and assert bundled values remain available with `metadata_source="bundled"`.

- [ ] **Step 2: Run the tests and verify the new cases fail.**

Run: `.venv/bin/pytest tests/test_runtime_catalog.py -q`
Expected: FAIL because exact-ID metadata enrichment is not implemented.

- [ ] **Step 3: Add an exact-model lookup function.**

Add a bounded `fetch_huggingface_model(model_id, timeout=6.0, use_cache=True)` function that requests `https://huggingface.co/api/models/{model_id}` with `expand[]=downloads`, `expand[]=likes`, `expand[]=createdAt`, and `expand[]=lastModified`. Store entries under a versioned key in the existing `hf_models_cache.json`; return the last usable cache entry on HTTP/network failure.

- [ ] **Step 4: Enrich only curated AirLLM/Colibri source IDs.**

For Colibri, fetch the source checkpoint ID for popularity and retain the exact container ID in a separate field. Do not fetch or download container files. Keep the existing Ollama catalogue behavior unchanged; use its bundled tags and exact pull commands.

- [ ] **Step 5: Run the focused tests and verify they pass.**

Run: `.venv/bin/pytest tests/test_runtime_catalog.py -q`
Expected: PASS.

---

### Task 3: Add the grouped CLI command and output renderer

**Files:**
- Create: `src/modelscout/output/runtime_models.py`
- Modify: `src/modelscout/output/display.py`
- Modify: `src/modelscout/output/json_output.py`
- Modify: `src/modelscout/cli/commands.py`
- Modify: `src/modelscout/cli/main.py` only if command registration requires it
- Test: `tests/test_runtime_models_cli.py`

- [ ] **Step 1: Write failing CLI tests.**

Assert `modelscout runtimes --json` returns an object with `ollama`, `airllm`, and `colibri` arrays; assert the terminal command prints all three section headings and at least one exact ID from each non-empty section.

- [ ] **Step 2: Run the focused CLI tests and verify they fail.**

Run: `.venv/bin/pytest tests/test_runtime_models_cli.py -q`
Expected: FAIL because the command and renderer are absent.

- [ ] **Step 3: Implement the renderer.**

Create `render_runtime_models_terminal(report)` and `render_runtime_models_json(report)`. The terminal table must show exact `Model ID`, `Install`, `HF downloads`, `Updated`, `VRAM/RAM`, and `Notes`; use separate Rich panels titled `OLLAMA MODELS`, `AIRLLM COMPATIBLE MODELS`, and `COLIBRI COMPATIBLE MODELS`. JSON must be strict and include `checked_at`, `metadata_source`, and each record's exact IDs.

- [ ] **Step 4: Register `runtimes` with options `--json`, `--limit` (default 6), and `--offline`.**

The command must not download weights, invoke AirLLM, or invoke Colibri. `--offline` must skip HTTP enrichment and return bundled records.

- [ ] **Step 5: Re-export the JSON renderer from `output/display.py` and verify tests.**

Run: `.venv/bin/pytest tests/test_runtime_models_cli.py tests/test_ranking_and_cli.py -q`
Expected: PASS.

---

### Task 4: Add the read-only FastAPI endpoint

**Files:**
- Modify: `src/modelscout/web/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write a failing endpoint test.**

```python
def test_api_runtime_models_groups_exact_ids():
    response = client.get("/api/runtime-models?limit=6&offline=true")
    assert response.status_code == 200
    data = response.json()
    assert set(data["runtimes"]) == {"ollama", "airllm", "colibri"}
    airllm_ids = {row["model_id"] for row in data["runtimes"]["airllm"]}
    assert "Qwen/Qwen3-32B" in airllm_ids
    colibri_ids = {row["model_id"] for row in data["runtimes"]["colibri"]}
    assert "moonshotai/Kimi-K3" in colibri_ids
```

- [ ] **Step 2: Run the endpoint test and verify it fails.**

Run: `.venv/bin/pytest tests/test_api.py::test_api_runtime_models_groups_exact_ids -q`
Expected: FAIL with 404.

- [ ] **Step 3: Implement `GET /api/runtime-models`.**

Accept `limit` constrained to 1–6 and `offline` as a boolean. Call the same loader used by the CLI, return HTTP 200 for an empty/missing optional section with an explanatory `warnings` list, and do not expose filesystem paths or secrets.

- [ ] **Step 4: Run API tests and verify they pass.**

Run: `.venv/bin/pytest tests/test_api.py -q`
Expected: PASS.

---

### Task 5: Add separate web catalogue sections

**Files:**
- Modify: `src/modelscout/web/static/index.html`
- Modify: `src/modelscout/web/static/app.js`
- Modify: `src/modelscout/web/static/styles.css`
- Test: browser or static DOM smoke test if the repository's test tooling supports it

- [ ] **Step 1: Add a runtime catalogue navigation target and loading container.**

Add a `Runtime Models` section under the existing catalogue navigation with three labelled panels. Keep the current catalogue table unchanged.

- [ ] **Step 2: Add the fetch/render path.**

Fetch `/api/runtime-models?limit=6&offline=false` when the user opens the section, render each runtime independently, and show exact IDs in `<code>` elements. Display downloads, last-modified date, container ID (when present), install command, and runtime notes. Use `escapeHtml()` for every value and `safeHref()` for source links.

- [ ] **Step 3: Add responsive styles and loading/error states.**

Reuse the existing card, tag, table, and button classes; add only runtime-section modifiers. Ensure long HF IDs wrap rather than overflow at narrow widths. Show a clear message when a runtime has no qualifying records or when metadata is from the bundled snapshot.

- [ ] **Step 4: Verify the static UI contract.**

Run the existing local app smoke check and inspect `/api/runtime-models` output through the browser/API test. Confirm no unescaped model metadata is inserted into HTML and that the current catalogue tab still loads.

---

### Task 6: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `src/README.md` if the packaged quickstart needs the new command
- Test: all existing tests

- [ ] **Step 1: Document the new command and thresholds.**

Add examples for:

```bash
./start.sh runtimes
./start.sh runtimes --json
./start.sh runtimes --offline
```

Explain that 10,000 downloads is the primary filter, 5,000 is only a per-runtime fallback, exact IDs are installation identifiers, and no model weights are downloaded by the command.

- [ ] **Step 2: Run formatting/lint checks available in the project.**

Run the repository's configured test/lint commands if present; otherwise run `.venv/bin/pytest -q` and a Python compile check for the changed package.

- [ ] **Step 3: Run the full test suite.**

Run: `.venv/bin/pytest -q`
Expected: all existing and new tests PASS.

- [ ] **Step 4: Run CLI smoke checks.**

Run: `.venv/bin/python -m modelscout runtimes --offline --limit 6`
Expected: separate Ollama, AirLLM, and Colibri sections with exact model IDs.

Run: `.venv/bin/python -m modelscout runtimes --offline --json --limit 6`
Expected: valid JSON with three runtime keys and no inference side effects.

- [ ] **Step 5: Review the final diff without committing.**

Run: `git status --short` and `git diff --stat`; inspect only intended files. Do not commit unless the user explicitly requests it.
