"""FastAPI REST API endpoints for hardware, catalogue, benchmarks, and recommendations."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from modelscout.database.repository import DatabaseRepository
from modelscout.hardware.detector import detect_runtimes, detect_system_hardware
from modelscout.hardware.types import SystemHardware
from modelscout.recommendation.ranking import (
    ModelRecommendation,
    RecommendationReport,
    generate_recommendations,
)

router = APIRouter(prefix="/api")


class AnalyzeRequest(BaseModel):
    cpu: Optional[str] = None
    gpu: Optional[str] = None
    ram_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    storage_gb: Optional[float] = None
    storage_type: Optional[str] = "SSD"
    profile: str = "general"
    quantization: str = "Q4_K_M"
    top_n: int = 15
    fit_filter: Optional[str] = None
    speed_filter: Optional[str] = None
    evidence_filter: Optional[str] = None
    context_length: Optional[int] = None


class HardwareBenchmarkResponse(BaseModel):
    measured_cpu_gflops: float
    measured_ram_bandwidth_gbps: float
    duration_seconds: float
    status: str = "success"
    notes: str = "Real hardware test measured compute throughput and memory bandwidth."


@router.get("/health")
def get_health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "app": "modelscout", "version": "0.1.0"}


@router.get("/hardware")
def get_hardware() -> SystemHardware:
    """Returns detected host hardware and AI readiness scores."""
    return detect_system_hardware()


@router.get("/catalog/processors")
def search_processors(
    q: str = Query("", min_length=0),
    limit: int = Query(500, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Local autocomplete search for processors."""
    repo = DatabaseRepository()
    return repo.search_processors(q, limit=limit)


@router.get("/catalog/gpus")
def search_gpus(
    q: str = Query("", min_length=0),
    limit: int = Query(500, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    """Local autocomplete search for GPUs (supports '4090', 'RTX 4090', etc.)."""
    repo = DatabaseRepository()
    return repo.search_gpus(q, limit=limit)


@router.get("/catalog/search")
def search_all_catalog(q: str = Query("", min_length=1)) -> Dict[str, List[Dict[str, Any]]]:
    """Searches across processors, GPUs, and models."""
    repo = DatabaseRepository()
    cpus = repo.search_processors(q, limit=5)
    gpus = repo.search_gpus(q, limit=5)
    models = [m for m in repo.get_all_models() if q.lower() in m["canonical_name"].lower()][:5]
    return {"processors": cpus, "gpus": gpus, "models": models}


@router.get("/models")
def get_models() -> List[Dict[str, Any]]:
    """Returns the full normalized local model catalogue."""
    repo = DatabaseRepository()
    return repo.get_all_models()


@router.get("/models/{model_id}")
def get_model_details(model_id: str) -> Dict[str, Any]:
    """Returns detailed metadata, quantizations, and benchmarks for a specific model."""
    repo = DatabaseRepository()
    m = repo.get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    return m


@router.get("/benchmarks")
def get_benchmarks() -> List[Dict[str, Any]]:
    """Returns latest benchmark evidence and sources."""
    repo = DatabaseRepository()
    conn = repo.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM benchmarks ORDER BY score DESC LIMIT 50")
        return [dict(r) for r in cur.fetchall()]
    finally:
        if repo.db_path != ":memory:":
            conn.close()


@router.get("/runtimes")
def get_runtimes() -> Dict[str, Any]:
    """Detects installed local AI inference runtimes."""
    rt = detect_runtimes()
    return rt.model_dump()


@router.post("/analyze")
def analyze_hardware(req: AnalyzeRequest) -> RecommendationReport:
    """Analyzes hardware configuration and returns complete explainable recommendations."""
    repo = DatabaseRepository()
    hw = detect_system_hardware(
        cpu_override=req.cpu,
        gpu_override=req.gpu,
        ram_override_gb=req.ram_gb,
        vram_override_gb=req.vram_gb,
    )

    report = generate_recommendations(
        hardware=hw,
        profile=req.profile,
        quantization=req.quantization,
        top_n=req.top_n,
        repo=repo,
        fit_filter=req.fit_filter,
        speed_filter=req.speed_filter,
        evidence_filter=req.evidence_filter,
        context_length=req.context_length,
    )
    return report


@router.post("/benchmark")
def run_hardware_benchmark() -> HardwareBenchmarkResponse:
    """Opt-in real hardware compute and memory bandwidth benchmark (5-10s)."""
    t0 = time.time()

    # 1. Measure memory bandwidth by allocating and sequentially reading/writing a 128MB bytearray
    chunk_size = 128 * 1024 * 1024
    buf = bytearray(chunk_size)
    iters = 6
    bw_start = time.time()
    for _ in range(iters):
        buf[::64] = b"\x01" * (chunk_size // 64)
    bw_elapsed = time.time() - bw_start
    total_bytes = chunk_size * iters
    ram_bw_gbps = round((total_bytes / (1024**3)) / max(0.001, bw_elapsed), 1)

    # 2. Measure CPU FLOPS via tight vector dot-product loop
    n = 2_000_000
    vec_a = [1.0001] * 1000
    vec_b = [0.9999] * 1000
    f_start = time.time()
    accum = 0.0
    for _ in range(1000):
        accum += sum(x * y for x, y in zip(vec_a, vec_b))
    f_elapsed = time.time() - f_start
    gflops = round((1000 * 2000 / 1e9) / max(0.0001, f_elapsed), 2)

    total_duration = round(time.time() - t0, 2)
    return HardwareBenchmarkResponse(
        measured_cpu_gflops=gflops,
        measured_ram_bandwidth_gbps=ram_bw_gbps,
        duration_seconds=total_duration,
    )
