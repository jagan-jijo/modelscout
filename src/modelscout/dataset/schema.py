"""Pydantic schemas for the dataset catalogues, models, and benchmarks."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AppleSiliconRecord(BaseModel):
    id: str
    vendor: str = "Apple"
    name: str
    family: str  # M1, M2, M3, M4
    generation: Optional[int] = None
    architecture: str = "Apple Silicon"
    memory_type: str = "unified"
    max_unified_memory_gb: Optional[int] = None
    cpu_cores: Optional[int] = None
    gpu_cores: Optional[int] = None
    memory_bandwidth_gbps: Optional[float] = None
    base_ghz: Optional[float] = None
    boost_ghz: Optional[float] = None
    frequency_ghz: Optional[str] = None
    metal_support: bool = True
    unified_memory: bool = True
    aliases: List[str] = Field(default_factory=list)


class CpuRecord(BaseModel):
    vendor: str  # AMD, Intel
    name: str
    family: str  # Ryzen, Core, Core Ultra
    series: Optional[str] = None  # 5000, 7000, Series 2
    architecture: Optional[str] = None  # Zen 3, Zen 4, etc.
    generation: Optional[int] = None  # 12, 13, 14
    cores: Optional[int] = None
    threads: Optional[int] = None
    base_ghz: Optional[float] = None
    boost_ghz: Optional[float] = None
    frequency_ghz: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)


class GpuRecord(BaseModel):
    vendor: str  # NVIDIA, AMD, Intel
    name: str
    architecture: Optional[str] = None  # Turing, Ampere, Ada Lovelace, Blackwell, RDNA, Xe
    family: Optional[str] = None  # Radeon RX, Arc, GeForce
    series: Optional[str] = None
    vram_gb: Optional[float] = None
    memory_type: Optional[str] = None
    memory_bandwidth_gbps: Optional[float] = None
    compute_capability: Optional[str] = None
    cuda_support: bool = False
    rocm_support: bool = False
    metal_support: bool = False
    aliases: List[str] = Field(default_factory=list)


class OllamaModelRecord(BaseModel):
    ollama_name: str
    family: str
    parameters: Optional[str] = None
    active_parameters: Optional[str] = None
    effective_parameters: Optional[str] = None
    moe: Optional[bool] = False
    context_tokens: Optional[int] = None
    vision: Optional[bool] = False
    audio: Optional[bool] = False
    coding: Optional[bool] = False


class NvidiaBuildModelRecord(BaseModel):
    nvidia_build_name: str
    publisher: str
    type: str  # LLM, multimodal
    parameters: Optional[str] = None
    active_parameters: Optional[str] = None
    moe: Optional[bool] = False
    context_tokens: Optional[int] = None
    capabilities: List[str] = Field(default_factory=list)


class HuggingFaceModelRecord(BaseModel):
    huggingface_id: str
    model_name: str
    publisher: str
    library_name: Optional[str] = "transformers"
    pipeline_tag: Optional[str] = "text-generation"
    parameters: Optional[int] = None
    active_parameters: Optional[int] = None
    architecture: Optional[str] = None
    moe: Optional[bool] = False
    context_tokens: Optional[int] = None
    modalities: List[str] = Field(default_factory=lambda: ["text"])
    license: Optional[str] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source: str = "huggingface"
    source_url: Optional[str] = None


class ArtifactRecord(BaseModel):
    format: str = "GGUF"  # GGUF, MLX, AWQ, EXL2
    quantization: str  # Q4_K_M, Q5_K_M, Q8_0, FP16
    size_bytes: Optional[int] = None
    source: Optional[str] = None


class CanonicalModelIdentity(BaseModel):
    canonical_name: str
    family: str
    publisher: str


class CanonicalModelSources(BaseModel):
    huggingface: Optional[Dict[str, Any]] = None
    ollama: List[str] = Field(default_factory=list)
    nvidia_build: List[str] = Field(default_factory=list)


class CanonicalModelArchitecture(BaseModel):
    type: str = "Dense"  # Dense, MoE
    total_parameters: int
    active_parameters: Optional[int] = None
    experts: Optional[int] = None
    active_experts: Optional[int] = None
    layers: Optional[int] = None
    kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None


class CanonicalModelCapabilities(BaseModel):
    text: bool = True
    vision: bool = False
    audio: bool = False
    video: bool = False
    reasoning: bool = False
    coding: bool = False
    tool_calling: bool = False
    agents: bool = False


class CanonicalModelRuntime(BaseModel):
    ollama: bool = True
    llama_cpp: bool = True
    vllm: bool = True
    transformers: bool = True
    mlx: bool = False


class BenchmarkResult(BaseModel):
    model_id: Optional[str] = None
    benchmark: str
    score: float
    normalized_score: Optional[float] = None
    source: str
    source_url: Optional[str] = None
    date: Optional[str] = None
    retrieved_at: Optional[str] = None
    method: str = "direct"
    evidence_type: str = "direct"  # direct, variant, base_model, line_interpolated, self_reported
    confidence: str = "high"  # high, medium, low
    tier: str = "current"  # current, frozen


class CanonicalModelRecord(BaseModel):
    id: str
    identity: CanonicalModelIdentity
    sources: CanonicalModelSources = Field(default_factory=CanonicalModelSources)
    architecture: CanonicalModelArchitecture
    context: Dict[str, Any] = Field(default_factory=lambda: {"maximum_tokens": 131072})
    capabilities: CanonicalModelCapabilities = Field(default_factory=CanonicalModelCapabilities)
    runtime: CanonicalModelRuntime = Field(default_factory=CanonicalModelRuntime)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    benchmarks: List[BenchmarkResult] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=lambda: {"confidence": "high"})


class SourceRecord(BaseModel):
    id: str
    name: str
    type: str
    url: str
    authority: str = "official"
    refreshable: bool = True


class DatasetCatalog(BaseModel):
    version: str = "1.0.0"
    updated_at: str = "2026-09-11"
    apple_silicon: List[AppleSiliconRecord] = Field(default_factory=list)
    amd_cpus: List[CpuRecord] = Field(default_factory=list)
    intel_cpus: List[CpuRecord] = Field(default_factory=list)
    nvidia_gpus: List[GpuRecord] = Field(default_factory=list)
    amd_gpus: List[GpuRecord] = Field(default_factory=list)
    intel_gpus: List[GpuRecord] = Field(default_factory=list)
    ollama_models: List[OllamaModelRecord] = Field(default_factory=list)
    nvidia_build_models: List[NvidiaBuildModelRecord] = Field(default_factory=list)
    huggingface_models: List[HuggingFaceModelRecord] = Field(default_factory=list)
    canonical_models: List[CanonicalModelRecord] = Field(default_factory=list)
    benchmarks: List[BenchmarkResult] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
