"""SQLite repository for models, hardware catalogues, benchmarks, and autocomplete."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from modelscout.database.models import CREATE_TABLES_SQL

if TYPE_CHECKING:
    from modelscout.dataset.schema import DatasetCatalog


def get_default_db_path() -> Path:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "modelscout"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "modelscout.db"


def _get_dataset_mtime() -> float:
    """Returns the newest mtime among dataset.json, models.json, cpus.json, and gpus.json."""
    candidates = [
        Path(__file__).resolve().parents[3] / "assets",
        Path(__file__).resolve().parents[1] / "dataset" / "data",
    ]
    max_mtime = 0.0
    for c in candidates:
        if c.is_dir():
            for f in ["dataset.json", "models.json", "cpus.json", "gpus.json"]:
                p = c / f
                if p.is_file():
                    try:
                        max_mtime = max(max_mtime, p.stat().st_mtime)
                    except OSError:
                        pass
    return max_mtime


class DatabaseRepository:
    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = str(db_path or get_default_db_path())
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self.get_connection()
        needs_seed = False
        try:
            conn.executescript(CREATE_TABLES_SQL)
            conn.execute("CREATE TABLE IF NOT EXISTS catalog_sync_meta (key TEXT PRIMARY KEY, val TEXT)")
            conn.commit()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM models")
            model_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM hardware_cpu WHERE vendor = 'Intel'")
            intel_cpu_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM hardware_gpu WHERE vendor = 'NVIDIA' AND vram_gb >= 40")
            industry_gpu_count = cur.fetchone()[0]

            cur.execute("SELECT val FROM catalog_sync_meta WHERE key = 'last_dataset_mtime'")
            row = cur.fetchone()
            last_mtime = float(row[0]) if row and row[0] else 0.0
            current_mtime = _get_dataset_mtime()

            if (
                model_count == 0
                or intel_cpu_count < 20
                or industry_gpu_count < 5
                or (current_mtime > 0.0 and current_mtime > last_mtime)
            ):
                needs_seed = True
        finally:
            if self.db_path != ":memory:":
                conn.close()

        if needs_seed:
            self._auto_seed_if_available()

    def _auto_seed_if_available(self) -> None:
        from modelscout.dataset.loader import load_dataset

        self.import_catalog(load_dataset())

    def import_catalog(self, catalog: DatasetCatalog) -> Dict[str, int]:
        """Idempotently imports DatasetCatalog into SQLite tables."""
        conn = self.get_connection()
        stats = {"cpus": 0, "gpus": 0, "models": 0, "artifacts": 0, "benchmarks": 0}
        try:
            cursor = conn.cursor()

            # 1. Apple Silicon SoCs (both CPU and GPU records)
            for soc in catalog.apple_silicon:
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_cpu (id, name, vendor, family, architecture, cores, base_ghz, boost_ghz, frequency_ghz)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (soc.id, soc.name, soc.vendor, soc.family, soc.architecture, soc.cpu_cores or 8, soc.base_ghz, soc.boost_ghz, soc.frequency_ghz),
                )
                stats["cpus"] += 1

                # GPU entry
                gpu_id = f"{soc.id}-gpu"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_gpu
                       (id, name, vendor, architecture, family, vram_gb, memory_bandwidth_gbps, metal_support, unified_memory)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1)""",
                    (
                        gpu_id,
                        f"{soc.name} GPU",
                        soc.vendor,
                        "Apple Silicon GPU",
                        soc.family,
                        soc.max_unified_memory_gb or 36.0,
                        soc.memory_bandwidth_gbps or 150.0,
                    ),
                )
                stats["gpus"] += 1

                # Aliases
                for alias in soc.aliases:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'cpu', ?)", (alias, soc.id))
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'gpu', ?)", (alias, gpu_id))

            # 2. AMD CPUs
            for cpu in catalog.amd_cpus:
                cid = f"amd-{cpu.name.lower().replace(' ', '-')}"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_cpu (id, name, vendor, family, series, architecture, cores, threads, base_ghz, boost_ghz, frequency_ghz)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (cid, cpu.name, cpu.vendor, cpu.family, cpu.series, cpu.architecture, cpu.cores or 8, cpu.threads or 16, cpu.base_ghz, cpu.boost_ghz, cpu.frequency_ghz),
                )
                stats["cpus"] += 1
                for alias in cpu.aliases + [cpu.name.replace("AMD ", "")]:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'cpu', ?)", (alias, cid))

            # 3. Intel CPUs
            for cpu in catalog.intel_cpus:
                cid = f"intel-{cpu.name.lower().replace(' ', '-')}"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_cpu (id, name, vendor, family, series, architecture, cores, threads, base_ghz, boost_ghz, frequency_ghz)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (cid, cpu.name, cpu.vendor, cpu.family, cpu.series, str(cpu.generation or ""), cpu.cores or 8, cpu.threads or 16, cpu.base_ghz, cpu.boost_ghz, cpu.frequency_ghz),
                )
                stats["cpus"] += 1
                for alias in cpu.aliases + [cpu.name.replace("Intel ", "")]:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'cpu', ?)", (alias, cid))

            # 4. NVIDIA GPUs
            for gpu in catalog.nvidia_gpus:
                gid = f"nvidia-{gpu.name.lower().replace(' ', '-')}"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_gpu
                       (id, name, vendor, architecture, family, series, vram_gb, memory_bandwidth_gbps, compute_capability, cuda_support)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        gid,
                        gpu.name,
                        gpu.vendor,
                        gpu.architecture,
                        gpu.family or "GeForce RTX",
                        gpu.series,
                        gpu.vram_gb,
                        gpu.memory_bandwidth_gbps or 500.0,
                        gpu.compute_capability,
                        1 if gpu.cuda_support else 0,
                    ),
                )
                stats["gpus"] += 1
                for alias in gpu.aliases + [gpu.name.replace("NVIDIA GeForce ", ""), gpu.name.replace("NVIDIA ", "")]:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'gpu', ?)", (alias, gid))

            # 5. AMD GPUs
            for gpu in catalog.amd_gpus:
                gid = f"amd-{gpu.name.lower().replace(' ', '-')}"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_gpu
                       (id, name, vendor, architecture, family, series, vram_gb, memory_bandwidth_gbps, rocm_support)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        gid,
                        gpu.name,
                        gpu.vendor,
                        gpu.architecture or "RDNA",
                        gpu.family or "Radeon RX",
                        gpu.series,
                        gpu.vram_gb,
                        gpu.memory_bandwidth_gbps or 500.0,
                        1 if gpu.rocm_support else 0,
                    ),
                )
                stats["gpus"] += 1
                for alias in gpu.aliases + [gpu.name.replace("AMD Radeon ", ""), gpu.name.replace("AMD ", "")]:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'gpu', ?)", (alias, gid))

            # 6. Intel GPUs
            for gpu in catalog.intel_gpus:
                gid = f"intel-{gpu.name.lower().replace(' ', '-')}"
                cursor.execute(
                    """INSERT OR REPLACE INTO hardware_gpu
                       (id, name, vendor, architecture, family, vram_gb)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (gid, gpu.name, gpu.vendor, gpu.architecture, "Arc", gpu.vram_gb),
                )
                stats["gpus"] += 1
                for alias in gpu.aliases + [gpu.name.replace("Intel ", "")]:
                    cursor.execute("INSERT OR REPLACE INTO hardware_alias VALUES (?, 'gpu', ?)", (alias, gid))

            # 7. Canonical Models
            for model in catalog.canonical_models:
                fam_id = model.identity.family.lower().replace(" ", "-")
                cursor.execute(
                    "INSERT OR REPLACE INTO model_family VALUES (?, ?, ?, ?)",
                    (fam_id, model.identity.family, model.identity.publisher, model.architecture.type),
                )

                ol_name = model.sources.ollama[0] if model.sources.ollama else None
                nv_name = model.sources.nvidia_build[0] if model.sources.nvidia_build else None
                hf_id = model.sources.huggingface.get("id") if model.sources.huggingface else None

                cursor.execute(
                    """INSERT OR REPLACE INTO models (
                        id, canonical_name, family_id, publisher, total_parameters, active_parameters,
                        architecture_type, context_length, has_text, has_vision, has_audio, has_reasoning,
                        has_coding, has_tool_calling, has_agents, ollama_name, nvidia_build_name, huggingface_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model.id,
                        model.identity.canonical_name,
                        fam_id,
                        model.identity.publisher,
                        model.architecture.total_parameters,
                        model.architecture.active_parameters or model.architecture.total_parameters,
                        model.architecture.type,
                        model.context.get("maximum_tokens", 131072),
                        1 if model.capabilities.text else 0,
                        1 if model.capabilities.vision else 0,
                        1 if model.capabilities.audio else 0,
                        1 if model.capabilities.reasoning else 0,
                        1 if model.capabilities.coding else 0,
                        1 if model.capabilities.tool_calling else 0,
                        1 if model.capabilities.agents else 0,
                        ol_name,
                        nv_name,
                        hf_id,
                    ),
                )
                stats["models"] += 1

                # Artifacts
                cursor.execute("DELETE FROM artifacts WHERE model_id = ?", (model.id,))
                for art in model.artifacts:
                    cursor.execute(
                        """INSERT INTO artifacts (model_id, format, quantization, file_size_bytes, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (model.id, art.format, art.quantization, art.size_bytes, art.source),
                    )
                    stats["artifacts"] += 1

                # Benchmarks on model
                for b in model.benchmarks:
                    cursor.execute(
                        """INSERT INTO benchmarks (model_id, benchmark_name, score, normalized_score, source, source_url, date, tier)
                           SELECT ?, ?, ?, ?, ?, ?, ?, ?
                           WHERE NOT EXISTS (
                               SELECT 1 FROM benchmarks WHERE model_id = ? AND benchmark_name = ?
                               AND source = ? AND date IS ? AND score = ?
                           )""",
                        (model.id, b.benchmark, b.score, b.normalized_score if b.normalized_score is not None else b.score, b.source, b.source_url, b.date, b.tier,
                         model.id, b.benchmark, b.source, b.date, b.score),
                    )
                    stats["benchmarks"] += 1

            # 8. Top-level Benchmarks
            for b in catalog.benchmarks:
                cursor.execute(
                    """INSERT INTO benchmarks (model_id, benchmark_name, score, normalized_score, source, source_url, date, evidence_type, confidence, tier, retrieved_at)
                       SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                       WHERE NOT EXISTS (
                           SELECT 1 FROM benchmarks WHERE model_id = ? AND benchmark_name = ?
                           AND source = ? AND date IS ? AND score = ?
                       )""",
                    (b.model_id, b.benchmark, b.score, b.normalized_score if b.normalized_score is not None else b.score, b.source, b.source_url, b.date, b.evidence_type, b.confidence, b.tier, b.retrieved_at,
                     b.model_id, b.benchmark, b.source, b.date, b.score),
                )
            cursor.execute("CREATE TABLE IF NOT EXISTS catalog_sync_meta (key TEXT PRIMARY KEY, val TEXT)")
            cursor.execute("INSERT OR REPLACE INTO catalog_sync_meta (key, val) VALUES ('last_dataset_mtime', ?)", (str(_get_dataset_mtime()),))
            conn.commit()
            return stats
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def insert_models(self, models: List[Any]) -> int:
        """Inserts a list of CanonicalModelRecord instances into the database."""
        conn = self.get_connection()
        count = 0
        try:
            cursor = conn.cursor()
            for model in models:
                fam_id = model.identity.family.lower().replace(" ", "-")
                cursor.execute(
                    "INSERT OR REPLACE INTO model_family VALUES (?, ?, ?, ?)",
                    (fam_id, model.identity.family, model.identity.publisher, model.architecture.type),
                )
                ol_name = model.sources.ollama[0] if model.sources.ollama else None
                nv_name = model.sources.nvidia_build[0] if model.sources.nvidia_build else None
                hf_id = model.sources.huggingface.get("id") if model.sources.huggingface else None

                cursor.execute(
                    """INSERT OR REPLACE INTO models (
                        id, canonical_name, family_id, publisher, total_parameters, active_parameters,
                        architecture_type, context_length, has_text, has_vision, has_audio, has_reasoning,
                        has_coding, has_tool_calling, has_agents, ollama_name, nvidia_build_name, huggingface_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model.id,
                        model.identity.canonical_name,
                        fam_id,
                        model.identity.publisher,
                        model.architecture.total_parameters,
                        model.architecture.active_parameters or model.architecture.total_parameters,
                        model.architecture.type,
                        model.context.get("maximum_tokens", 131072) if isinstance(model.context, dict) else 131072,
                        1 if model.capabilities.text else 0,
                        1 if model.capabilities.vision else 0,
                        1 if model.capabilities.audio else 0,
                        1 if model.capabilities.reasoning else 0,
                        1 if model.capabilities.coding else 0,
                        1 if model.capabilities.tool_calling else 0,
                        1 if model.capabilities.agents else 0,
                        ol_name,
                        nv_name,
                        hf_id,
                    ),
                )
                count += 1
                cursor.execute("DELETE FROM artifacts WHERE model_id = ?", (model.id,))
                for art in model.artifacts:
                    cursor.execute(
                        """INSERT INTO artifacts (model_id, format, quantization, file_size_bytes, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (model.id, art.format, art.quantization, art.size_bytes, art.source),
                    )
                for b in getattr(model, "benchmarks", []):
                    cursor.execute(
                        """INSERT INTO benchmarks (model_id, benchmark_name, score, normalized_score, source, source_url, date, tier)
                           SELECT ?, ?, ?, ?, ?, ?, ?, ?
                           WHERE NOT EXISTS (
                               SELECT 1 FROM benchmarks WHERE model_id = ? AND benchmark_name = ?
                               AND source = ? AND date IS ? AND score = ?
                           )""",
                        (model.id, b.benchmark, b.score, getattr(b, "normalized_score", b.score), b.source, b.source_url, b.date, b.tier,
                         model.id, b.benchmark, b.source, b.date, b.score),
                    )
            conn.commit()
            return count
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def search_processors(self, query: str = "", limit: int = 500) -> List[Dict[str, Any]]:
        """Autocomplete search for processors supporting exact, prefix, alias, clock speeds, and multi-word matching."""
        conn = self.get_connection()
        try:
            clean = query.strip()
            cur = conn.cursor()
            if not clean:
                cur.execute("SELECT * FROM hardware_cpu ORDER BY vendor ASC, name ASC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]

            words = clean.split()
            like_clauses = []
            params = []
            for w in words:
                pat = f"%{w}%"
                like_clauses.append("(c.name LIKE ? OR c.vendor LIKE ? OR c.family LIKE ? OR a.alias LIKE ? OR coalesce(c.frequency_ghz, '') LIKE ?)")
                params.extend([pat, pat, pat, pat, pat])
            params.append(limit)

            sql = f"""
                SELECT DISTINCT c.* FROM hardware_cpu c
                LEFT JOIN hardware_alias a ON a.target_id = c.id AND a.target_type = 'cpu'
                WHERE {" AND ".join(like_clauses)}
                ORDER BY c.vendor ASC, c.name ASC LIMIT ?
            """
            cur.execute(sql, params)
            rows = cur.fetchall()
            if rows:
                return [dict(r) for r in rows]

            cur.execute(
                """SELECT DISTINCT * FROM hardware_cpu
                   WHERE name LIKE ? OR vendor LIKE ? OR family LIKE ? OR frequency_ghz LIKE ?
                   ORDER BY vendor ASC, name ASC LIMIT ?""",
                (f"%{clean}%", f"%{clean}%", f"%{clean}%", f"%{clean}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def search_gpus(self, query: str = "", limit: int = 500) -> List[Dict[str, Any]]:
        """Autocomplete search for GPUs supporting '4090', 'RTX 4090', 'GeForce 4090', and multi-word matching."""
        conn = self.get_connection()
        try:
            clean = query.strip()
            cur = conn.cursor()
            if not clean:
                cur.execute("SELECT * FROM hardware_gpu ORDER BY vendor ASC, vram_gb DESC, name ASC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]

            words = clean.split()
            like_clauses = []
            params = []
            for w in words:
                pat = f"%{w}%"
                like_clauses.append("(g.name LIKE ? OR g.vendor LIKE ? OR g.family LIKE ? OR g.architecture LIKE ? OR a.alias LIKE ?)")
                params.extend([pat, pat, pat, pat, pat])
            params.append(limit)

            sql = f"""
                SELECT DISTINCT g.* FROM hardware_gpu g
                LEFT JOIN hardware_alias a ON a.target_id = g.id AND a.target_type = 'gpu'
                WHERE {" AND ".join(like_clauses)}
                ORDER BY g.vram_gb DESC, g.name ASC LIMIT ?
            """
            cur.execute(sql, params)
            rows = cur.fetchall()
            if rows:
                return [dict(r) for r in rows]

            cur.execute(
                """SELECT DISTINCT * FROM hardware_gpu
                   WHERE name LIKE ? OR vendor LIKE ? OR architecture LIKE ? OR family LIKE ?
                   ORDER BY vram_gb DESC, name ASC LIMIT ?""",
                (f"%{clean}%", f"%{clean}%", f"%{clean}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_all_models(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM models ORDER BY canonical_name ASC")
            models = [dict(r) for r in cur.fetchall()]
            for m in models:
                m["artifacts"] = self.get_model_artifacts(m["id"])
                m["benchmarks"] = self.get_model_benchmarks(m["id"])
            return models
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM models WHERE id = ? OR canonical_name = ?", (model_id, model_id))
            row = cur.fetchone()
            if not row:
                return None
            m = dict(row)
            m["artifacts"] = self.get_model_artifacts(m["id"])
            m["benchmarks"] = self.get_model_benchmarks(m["id"])
            return m
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_model_artifacts(self, model_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM artifacts WHERE model_id = ? ORDER BY quantization ASC", (model_id,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_model_benchmarks(self, model_id: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM benchmarks WHERE model_id = ? ORDER BY score DESC", (model_id,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_stats(self) -> Dict[str, int]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM hardware_cpu")
            cpus = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM hardware_gpu")
            gpus = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM models")
            models = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM benchmarks")
            benchmarks = cur.fetchone()[0]
            return {"cpus": cpus, "gpus": gpus, "models": models, "benchmarks": benchmarks}
        finally:
            if self.db_path != ":memory:":
                conn.close()
