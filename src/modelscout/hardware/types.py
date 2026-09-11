"""Hardware types and models for ModelScout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict, Field as PydanticField


@dataclass
class GPUInfo:
    name: str = "Unknown GPU"
    vendor: str = "Unknown"  # "nvidia" | "amd" | "apple" | "intel"
    vram_bytes: int = 0
    usable_vram_bytes: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None  # NVIDIA only
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    memory_bandwidth_gbps: Optional[float] = None
    shared_memory: bool = False
    vram_overridden: bool = False
    device_count: int = 1
    architecture: str = "Unknown"
    _cuda_support: Optional[bool] = None
    _metal_support: Optional[bool] = None
    _rocm_support: Optional[bool] = None

    def __init__(
        self,
        name: str = "Unknown GPU",
        vendor: str = "Unknown",
        vram_bytes: int = 0,
        usable_vram_bytes: Optional[int] = None,
        compute_capability: Any = None,
        cuda_version: Optional[str] = None,
        rocm_version: Optional[str] = None,
        memory_bandwidth_gbps: Optional[float] = None,
        shared_memory: bool = False,
        vram_overridden: bool = False,
        vram_gb: Optional[float] = None,
        cuda_support: Optional[bool] = None,
        metal_support: Optional[bool] = None,
        rocm_support: Optional[bool] = None,
        vulkan_support: Optional[bool] = None,
        unified_memory: Optional[bool] = None,
        device_count: int = 1,
        architecture: str = "Unknown",
        **kwargs: Any,
    ):
        self.name = name
        self.vendor = vendor.lower() if vendor else "unknown"
        if vram_gb is not None and vram_bytes == 0:
            self.vram_bytes = int(vram_gb * (1024**3))
        else:
            self.vram_bytes = vram_bytes
        self.usable_vram_bytes = usable_vram_bytes
        if isinstance(compute_capability, str):
            try:
                parts = compute_capability.split('.')
                self.compute_capability = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
            except Exception:
                self.compute_capability = None
        else:
            self.compute_capability = compute_capability
        self.cuda_version = cuda_version
        self.rocm_version = rocm_version
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self.shared_memory = shared_memory if unified_memory is None else unified_memory
        self.vram_overridden = vram_overridden
        self.device_count = device_count
        self.architecture = architecture
        self._cuda_support = cuda_support
        self._metal_support = metal_support
        self._rocm_support = rocm_support

    @property
    def vram_gb(self) -> float:
        return round(self.vram_bytes / (1024**3), 2)

    @vram_gb.setter
    def vram_gb(self, val: float):
        self.vram_bytes = int(val * (1024**3))

    @property
    def cuda_support(self) -> bool:
        if self._cuda_support is not None:
            return self._cuda_support
        return bool(self.cuda_version or self.compute_capability or self.vendor in ["nvidia"])

    @property
    def metal_support(self) -> bool:
        if self._metal_support is not None:
            return self._metal_support
        return self.vendor in ["apple"]

    @property
    def rocm_support(self) -> bool:
        if self._rocm_support is not None:
            return self._rocm_support
        return bool(self.rocm_version or self.vendor in ["amd"])

    @property
    def vulkan_support(self) -> bool:
        return True

    @property
    def unified_memory(self) -> bool:
        return self.shared_memory


GpuInfo = GPUInfo


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    ram_bytes: int = 0
    ram_budget_bytes: Optional[int] = None
    disk_free_bytes: int = 0
    os: str = "linux"  # "linux" | "darwin" | "windows"
    budget_notes: list[str] = field(default_factory=list)


class CpuInfo(BaseModel):
    name: str = "Unknown CPU"
    vendor: str = "Unknown"
    architecture: str = "Unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    features: List[str] = PydanticField(default_factory=list)
    base_clock_mhz: Optional[float] = None
    max_clock_mhz: Optional[float] = None


class MemoryInfo(BaseModel):
    total_ram_bytes: int = 0
    total_ram_gb: float = 0.0
    available_ram_bytes: int = 0
    available_ram_gb: float = 0.0
    unified_memory: bool = False
    memory_type: str = "Unknown"
    memory_bandwidth_gbps: Optional[float] = None


class StorageInfo(BaseModel):
    total_bytes: int = 0
    total_gb: float = 0.0
    available_bytes: int = 0
    available_gb: float = 0.0
    storage_type: str = "SSD"


class RuntimeCapability(BaseModel):
    ollama_installed: bool = False
    ollama_running: bool = False
    llama_cpp_installed: bool = False
    metal_available: bool = False
    cuda_available: bool = False
    rocm_available: bool = False
    vulkan_available: bool = False
    python_version: Optional[str] = None
    pytorch_version: Optional[str] = None


class HardwareScoreBreakdown(BaseModel):
    overall_score: int = PydanticField(ge=0, le=100)
    gpu_score: int = PydanticField(ge=0, le=100)
    memory_capacity_score: int = PydanticField(ge=0, le=100)
    memory_bandwidth_score: int = PydanticField(ge=0, le=100)
    cpu_score: int = PydanticField(ge=0, le=100)
    ai_readiness_score: int = PydanticField(ge=0, le=100)
    summary: str = ""


class SystemHardware(BaseModel):
    os_name: str
    os_version: str
    architecture: str
    cpu: CpuInfo
    gpus: List[GpuInfo] = PydanticField(default_factory=list)
    memory: MemoryInfo
    storage: StorageInfo
    runtimes: RuntimeCapability = PydanticField(default_factory=RuntimeCapability)
    is_apple_silicon: bool = False
    apple_silicon_generation: Optional[str] = None
    hardware_score: Optional[HardwareScoreBreakdown] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def primary_gpu(self) -> Optional[GpuInfo]:
        if self.gpus:
            return max(self.gpus, key=lambda g: g.vram_gb)
        return None

    @property
    def total_effective_vram_gb(self) -> float:
        if self.memory.unified_memory:
            return round(self.memory.total_ram_gb * 0.75, 1)
        if self.gpus:
            return round(sum(g.vram_gb * max(1, getattr(g, 'device_count', 1)) for g in self.gpus), 1)
        return 0.0
