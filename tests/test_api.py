"""Tests for FastAPI endpoints and web server."""

import pytest
from fastapi.testclient import TestClient
from modelscout.web.app import app

client = TestClient(app, base_url="http://127.0.0.1")


def test_api_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_browser_requests_stay_local():
    response = client.post("/api/benchmark", headers={"Origin": "https://example.org"})
    assert response.status_code == 403
    assert client.get("/api/hardware", headers={"Host": "example.org"}).status_code == 400
    assert client.get("/api/hardware", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1"}).status_code == 200
    assert "script-src 'self'" in client.get("/").headers["content-security-policy"]


def test_api_hardware():
    r = client.get("/api/hardware")
    assert r.status_code == 200
    data = r.json()
    assert "cpu" in data
    assert "memory" in data
    assert "hardware_score" in data


def test_api_catalog_search():
    # Processors search with Ryzen
    r_cpu = client.get("/api/catalog/processors?q=Ryzen")
    assert r_cpu.status_code == 200
    assert len(r_cpu.json()) > 0

    # Processors search with Apple
    r_apple = client.get("/api/catalog/processors?q=Apple")
    assert r_apple.status_code == 200
    assert len(r_apple.json()) > 0
    assert any("M" in c["name"] for c in r_apple.json())

    # Processors search with Intel
    r_intel = client.get("/api/catalog/processors?q=Intel")
    assert r_intel.status_code == 200
    assert len(r_intel.json()) > 0
    assert any("Core" in c["name"] for c in r_intel.json())

    # Default catalog search contains all vendors
    r_all = client.get("/api/catalog/processors")
    assert r_all.status_code == 200
    all_data = r_all.json()
    vendors = {p["vendor"] for p in all_data}
    assert {"Apple", "Intel", "AMD"}.issubset(vendors)

    r_gpu = client.get("/api/catalog/gpus?q=4090")
    assert r_gpu.status_code == 200
    assert len(r_gpu.json()) > 0


def test_api_models_and_details():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert len(models) > 0

    first_id = models[0]["id"]
    r_detail = client.get(f"/api/models/{first_id}")
    assert r_detail.status_code == 200
    assert r_detail.json()["id"] == first_id


def test_api_analyze():
    req = {
        "cpu": "Apple M3 Pro",
        "gpu": "Apple M3 Pro GPU",
        "ram_gb": 36,
        "profile": "coding",
        "quantization": "Q4_K_M",
    }
    r = client.post("/api/analyze", json=req)
    assert r.status_code == 200
    data = r.json()
    assert "hardware_score" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0


def test_api_benchmark_endpoint():
    r = client.post("/api/benchmark")
    assert r.status_code == 200
    data = r.json()
    assert data["measured_cpu_gflops"] >= 0
    assert data["measured_ram_bandwidth_gbps"] > 0
    assert data["duration_seconds"] > 0
