"""SQLite database schema and SQL definitions."""

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS hardware_cpu (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vendor TEXT NOT NULL,
    family TEXT,
    series TEXT,
    architecture TEXT,
    cores INTEGER,
    threads INTEGER,
    base_ghz REAL,
    boost_ghz REAL,
    frequency_ghz TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hardware_gpu (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vendor TEXT NOT NULL,
    architecture TEXT,
    family TEXT,
    series TEXT,
    vram_gb REAL,
    memory_type TEXT,
    memory_bandwidth_gbps REAL,
    compute_capability TEXT,
    cuda_support INTEGER DEFAULT 0,
    rocm_support INTEGER DEFAULT 0,
    metal_support INTEGER DEFAULT 0,
    unified_memory INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hardware_alias (
    alias TEXT PRIMARY KEY,
    target_type TEXT NOT NULL, -- 'cpu' or 'gpu'
    target_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hardware_cpu_name ON hardware_cpu(name);
CREATE INDEX IF NOT EXISTS idx_hardware_cpu_freq ON hardware_cpu(frequency_ghz);
CREATE INDEX IF NOT EXISTS idx_hardware_gpu_name ON hardware_gpu(name);
CREATE INDEX IF NOT EXISTS idx_hardware_alias_target ON hardware_alias(target_id);

CREATE TABLE IF NOT EXISTS model_family (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    publisher TEXT NOT NULL,
    architecture_type TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    family_id TEXT NOT NULL,
    publisher TEXT NOT NULL,
    total_parameters INTEGER NOT NULL,
    active_parameters INTEGER,
    architecture_type TEXT DEFAULT 'Dense',
    context_length INTEGER DEFAULT 131072,
    has_text INTEGER DEFAULT 1,
    has_vision INTEGER DEFAULT 0,
    has_audio INTEGER DEFAULT 0,
    has_reasoning INTEGER DEFAULT 0,
    has_coding INTEGER DEFAULT 0,
    has_tool_calling INTEGER DEFAULT 0,
    has_agents INTEGER DEFAULT 0,
    ollama_name TEXT,
    nvidia_build_name TEXT,
    huggingface_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    format TEXT NOT NULL, -- GGUF, MLX, etc.
    quantization TEXT NOT NULL, -- Q4_K_M, Q5_K_M, Q8_0, etc.
    file_size_bytes INTEGER,
    source TEXT,
    FOREIGN KEY(model_id) REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    score REAL NOT NULL,
    normalized_score REAL NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT,
    date TEXT,
    evidence_type TEXT DEFAULT 'direct', -- direct, variant, base_model, line_interpolated, self_reported
    confidence TEXT DEFAULT 'high', -- high, medium, low
    tier TEXT DEFAULT 'current', -- current, frozen
    retrieved_at TEXT,
    FOREIGN KEY(model_id) REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS benchmark_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    tier TEXT DEFAULT 'current',
    weight REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS data_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT,
    retrieved_at TEXT NOT NULL,
    confidence TEXT DEFAULT 'high'
);

CREATE INDEX IF NOT EXISTS idx_hardware_cpu_name ON hardware_cpu(name);
CREATE INDEX IF NOT EXISTS idx_hardware_gpu_name ON hardware_gpu(name);
CREATE INDEX IF NOT EXISTS idx_models_canonical ON models(canonical_name);
CREATE INDEX IF NOT EXISTS idx_models_family ON models(family_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_model ON benchmarks(model_id);
"""
