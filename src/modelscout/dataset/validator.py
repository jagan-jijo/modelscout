"""Validation module for dataset consistency, bounds, and integrity."""

import re
from typing import List
from pydantic import BaseModel, Field

from modelscout.dataset.schema import DatasetCatalog


class ValidationError(BaseModel):
    category: str
    item_id: str
    field: str
    message: str


class ValidationResult(BaseModel):
    is_valid: bool
    total_checks: int
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid and not self.warnings:
            return f"✓ Dataset validation passed ({self.total_checks} checks clean)"
        msg = f"{'✓' if self.is_valid else '✗'} Dataset validation: {len(self.errors)} errors, {len(self.warnings)} warnings ({self.total_checks} checks)"
        return msg


def validate_catalog(catalog: DatasetCatalog) -> ValidationResult:
    """Performs deep validation across all entities in DatasetCatalog."""
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    total_checks = 0

    # 1. Check Apple Silicon IDs and Memory Sizes
    apple_ids = set()
    for soc in catalog.apple_silicon:
        total_checks += 1
        if soc.id in apple_ids:
            errors.append(ValidationError(category="AppleSilicon", item_id=soc.id, field="id", message=f"Duplicate SoC ID: {soc.id}"))
        apple_ids.add(soc.id)

        if soc.max_unified_memory_gb is not None:
            total_checks += 1
            if soc.max_unified_memory_gb < 8 or soc.max_unified_memory_gb > 1024:
                errors.append(ValidationError(category="AppleSilicon", item_id=soc.id, field="max_unified_memory_gb", message=f"Impossible memory size: {soc.max_unified_memory_gb} GB"))

    # 2. Check CPUs
    cpu_names = set()
    for cpu in catalog.amd_cpus + catalog.intel_cpus:
        total_checks += 1
        if cpu.name in cpu_names:
            warnings.append(ValidationError(category="CPU", item_id=cpu.name, field="name", message=f"Duplicate CPU name entry: {cpu.name}"))
        cpu_names.add(cpu.name)

        if cpu.vendor not in ["AMD", "Intel", "Apple"]:
            errors.append(ValidationError(category="CPU", item_id=cpu.name, field="vendor", message=f"Invalid vendor: {cpu.vendor}"))

    # 3. Check GPUs & VRAM bounds
    gpu_names = set()
    for gpu in catalog.nvidia_gpus + catalog.amd_gpus + catalog.intel_gpus:
        total_checks += 1
        if gpu.name in gpu_names:
            warnings.append(ValidationError(category="GPU", item_id=gpu.name, field="name", message=f"Duplicate GPU entry: {gpu.name}"))
        gpu_names.add(gpu.name)

        if gpu.vram_gb is not None:
            total_checks += 1
            if gpu.vram_gb <= 0 or gpu.vram_gb > 512:
                errors.append(ValidationError(category="GPU", item_id=gpu.name, field="vram_gb", message=f"Impossible VRAM size: {gpu.vram_gb} GB"))

    # 4. Check Canonical Models
    model_ids = set()
    for model in catalog.canonical_models:
        total_checks += 1
        if model.id in model_ids:
            errors.append(ValidationError(category="CanonicalModel", item_id=model.id, field="id", message=f"Duplicate model ID: {model.id}"))
        model_ids.add(model.id)

        total_checks += 1
        if model.architecture.total_parameters <= 0:
            errors.append(ValidationError(category="CanonicalModel", item_id=model.id, field="total_parameters", message=f"Invalid parameter count: {model.architecture.total_parameters}"))

        ctx = model.context.get("maximum_tokens", 0)
        total_checks += 1
        if ctx <= 0 or ctx > 10_000_000:
            warnings.append(ValidationError(category="CanonicalModel", item_id=model.id, field="context_tokens", message=f"Suspicious context length: {ctx}"))

    # 5. Check Benchmarks
    for b in catalog.benchmarks:
        total_checks += 1
        if b.score < 0.0 or b.score > 2000.0:  # Chatbot Arena can be ~1400, percentage 0-100
            errors.append(ValidationError(category="Benchmark", item_id=b.model_id, field="score", message=f"Impossible benchmark score: {b.score} for {b.benchmark}"))

        if b.source_url:
            total_checks += 1
            if not re.match(r"^https?://", b.source_url):
                warnings.append(ValidationError(category="Benchmark", item_id=b.model_id, field="source_url", message=f"Malformed URL: {b.source_url}"))

    # 6. Check Sources
    for s in catalog.sources:
        total_checks += 1
        if not re.match(r"^https?://", s.url):
            warnings.append(ValidationError(category="Source", item_id=s.id, field="url", message=f"Malformed source URL: {s.url}"))

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, total_checks=total_checks, errors=errors, warnings=warnings)
