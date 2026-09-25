import json
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import typer

from modelscout.cli.output import (
    console,
    err_console,
    render_runtime_models_json,
    render_runtime_models_terminal,
    render_scan_json,
    render_scan_markdown,
    render_scan_terminal,
)
from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.importer import import_dataset_to_db
from modelscout.dataset.loader import load_dataset
from modelscout.dataset.updater import start_background_model_update, sync_models_from_open_apis
from modelscout.dataset.validator import validate_catalog
from modelscout.hardware.detector import detect_system_hardware
from modelscout.models.runtime_catalog import build_runtime_report
from modelscout.recommendation.ranking import generate_recommendations

cmd_app = typer.Typer(help="Find local AI models that fit your hardware, with memory and speed estimates.")


@cmd_app.command(name="scan")
def scan_cmd(
    profile: str = typer.Option("general", "--profile", "-p", help="Recommendation profile (general, coding, reasoning, math, vision, creative, fast, quality, low-memory, cpu)"),
    quant: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization format to target (e.g. Q4_K_M, Q5_K_M, Q8_0)"),
    top: int = typer.Option(10, "--top", "-n", help="Top N models to recommend"),
    as_json: bool = typer.Option(False, "--json", help="Output strictly valid JSON to stdout (no Rich formatting)"),
    as_markdown: bool = typer.Option(False, "--markdown", "-m", help="Output GitHub-flavored Markdown table"),
    gpu: Optional[List[str]] = typer.Option(None, "--gpu", help="Simulate GPU (repeat --gpu, use commas, or write '2x RTX 4090')"),
    cpu: Optional[str] = typer.Option(None, "--cpu", help="Simulate CPU (e.g. 'Ryzen 9 7950X', 'Core i9-14900K')"),
    ram: Optional[float] = typer.Option(None, "--ram", help="Simulate RAM in GB (e.g. 32, 64, 128)"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Simulate VRAM in GB (e.g. 16, 24)"),
    vram_headroom: float = typer.Option(0.0, "--vram-headroom", help="Reserve VRAM headroom in GB (e.g. 2.0 to keep 2GB free)"),
    ram_budget: Optional[float] = typer.Option(None, "--ram-budget", help="Limit maximum RAM budget to allocate in GB"),
    fit_filter: Optional[str] = typer.Option(None, "--fit", help="Fit filter (full-gpu, partial, cpu-only)"),
    gpu_only: bool = typer.Option(False, "--gpu-only", help="Require full GPU fit"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Require CPU only fit"),
    speed_filter: Optional[str] = typer.Option(None, "--speed", help="Speed filter (usable, fast)"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed", help="Minimum estimated tok/s"),
    context_length: Optional[int] = typer.Option(None, "--context-length", help="Context length to evaluate at"),
    evidence: Optional[str] = typer.Option(None, "--evidence", help="Evidence filter (direct, base)"),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh data before scanning"),
) -> None:
    """Detects host hardware and recommends best matching local AI models."""
    # Launch non-blocking background model update from open endpoints
    start_background_model_update(timeout=8.0)

    repo = DatabaseRepository()

    # Ensure database is seeded from dataset.json if empty
    stats = repo.get_stats()
    if stats["models"] == 0 or refresh:
        try:
            catalog = load_dataset()
            repo.import_catalog(catalog)
        except Exception as e:
            if not as_json:
                err_console.print(f"[yellow]Warning: Could not seed from dataset.json: {e}[/yellow]")

    # Detect or simulate hardware
    hw = detect_system_hardware(
        cpu_override=cpu,
        gpu_override=gpu,
        ram_override_gb=ram,
        vram_override_gb=vram,
    )

    fit_val = fit_filter
    if gpu_only:
        fit_val = "full-gpu"
    elif cpu_only:
        fit_val = "cpu-only"

    report = generate_recommendations(
        hardware=hw,
        profile=profile,
        quantization=quant,
        top_n=top,
        repo=repo,
        fit_filter=fit_val,
        speed_filter=speed_filter,
        min_speed=min_speed,
        evidence_filter=evidence,
        context_length=context_length,
        vram_headroom_gb=vram_headroom,
        ram_budget_gb=ram_budget,
    )

    if as_json:
        render_scan_json(report)
    elif as_markdown:
        render_scan_markdown(report)
    else:
        render_scan_terminal(report)


@cmd_app.command(name="hardware")
def hardware_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Shows detailed host hardware detection, memory specs, and AI score."""
    hw = detect_system_hardware()
    if as_json:
        sys.stdout.write(hw.model_dump_json(indent=2) + "\n")
        return

    console.print(Panel("[bold cyan]HOST HARDWARE CONFIGURATION[/bold cyan]", box=ROUNDED))
    table = Table(box=ROUNDED, header_style="bold magenta")
    table.add_column("Component", style="bold white")
    table.add_column("Details", style="green")

    table.add_row("Operating System", f"{hw.os_name} {hw.os_version} ({hw.architecture})")
    table.add_row("Processor (CPU)", f"{hw.cpu.name} ({hw.cpu.physical_cores} physical / {hw.cpu.logical_cores} logical cores)")
    table.add_row("CPU Features", ", ".join(hw.cpu.features) if hw.cpu.features else "Standard")
    table.add_row("System Memory (RAM)", f"{hw.memory.total_ram_gb:.1f} GB ({hw.memory.memory_type})")
    table.add_row("Available RAM", f"{hw.memory.available_ram_gb:.1f} GB")
    table.add_row("Unified Memory", "Yes (Apple Silicon)" if hw.memory.unified_memory else "No (Discrete / Host RAM)")

    if hw.gpus:
        for i, g in enumerate(hw.gpus, 1):
            table.add_row(f"GPU #{i}", f"{g.name} — {g.vram_gb:.1f} GB VRAM ({g.architecture}), {g.memory_bandwidth_gbps or 0:.0f} GB/s bandwidth")
    else:
        table.add_row("GPU", "No discrete GPU detected (using CPU/integrated graphics)")

    table.add_row("Storage Free", f"{hw.storage.available_gb:.1f} GB of {hw.storage.total_gb:.1f} GB ({hw.storage.storage_type})")
    table.add_row("Ollama Runtime", "✓ Installed and Running" if hw.runtimes.ollama_running else ("✓ Installed (Stopped)" if hw.runtimes.ollama_installed else "✗ Not detected"))
    table.add_row("llama.cpp Runtime", "✓ Installed" if hw.runtimes.llama_cpp_installed else "✗ Not detected")
    table.add_row("Acceleration", f"Metal: {'✓' if hw.runtimes.metal_available else '✗'} | CUDA: {'✓' if hw.runtimes.cuda_available else '✗'} | ROCm: {'✓' if hw.runtimes.rocm_available else '✗'}")

    console.print(table)
    if hw.hardware_score:
        console.print(f"\n[bold]AI Hardware Score:[/bold] [bold yellow]{hw.hardware_score.overall_score}/100[/bold yellow]")
        console.print(f"[dim]{hw.hardware_score.summary}[/dim]\n")


@cmd_app.command(name="models")
def models_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Displays the normalized local model catalogue and metadata."""
    repo = DatabaseRepository()
    models = repo.get_all_models()
    if as_json:
        sys.stdout.write(json.dumps(models, indent=2) + "\n")
        return

    table = Table(title=f"LOCAL MODEL CATALOGUE ({len(models)} models)", box=ROUNDED, header_style="bold cyan")
    table.add_column("Model ID", style="bold white")
    table.add_column("Family", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Capabilities")
    table.add_column("Ollama Tag", style="green")

    for m in models:
        caps = []
        if m["has_coding"]: caps.append("Code")
        if m["has_reasoning"]: caps.append("Reason")
        if m["has_vision"]: caps.append("Vision")
        table.add_row(
            m["canonical_name"],
            m["family_id"].replace("-", " ").title(),
            f"{m['total_parameters'] / 1e9:.1f}B",
            f"{m['context_length'] // 1024}k",
            ", ".join(caps) if caps else "Text",
            m["ollama_name"] or "-",
        )
    console.print(table)


@cmd_app.command(name="runtimes")
def runtimes_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output strict grouped JSON"),
    limit: int = typer.Option(6, "--limit", min=1, max=6, help="Maximum models per runtime (1-6)"),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Use bundled metadata without network access",
    ),
) -> None:
    """Lists documented models for Ollama, AirLLM, and Colibri."""
    report = build_runtime_report(
        limit=limit,
        enrich=not offline,
        offline=offline,
    )
    if as_json:
        render_runtime_models_json(report)
    else:
        render_runtime_models_terminal(report)


@cmd_app.command(name="benchmarks")
def benchmarks_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Shows benchmark records, tiers, and sources."""
    repo = DatabaseRepository()
    conn = repo.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM benchmarks ORDER BY score DESC LIMIT 30")
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        if repo.db_path != ":memory:":
            conn.close()

    if as_json:
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return

    table = Table(title="BENCHMARK RECORDS & EVIDENCE TIERS", box=ROUNDED, header_style="bold magenta")
    table.add_column("Model ID", style="bold white")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Score", justify="right", style="bold yellow")
    table.add_column("Source")
    table.add_column("Tier", justify="center")
    table.add_column("Evidence", justify="center")
    table.add_column("Date", justify="center", style="dim")

    for r in rows:
        tier_style = "green" if r["tier"] == "current" else "yellow"
        table.add_row(
            r["model_id"],
            r["benchmark_name"],
            f"{r['score']:.1f}",
            r["source"],
            f"[{tier_style}]{r['tier'].title()}[/{tier_style}]",
            r["evidence_type"].title() if r.get("evidence_type") else "Direct",
            r.get("date") or "2026-08",
        )
    console.print(table)


@cmd_app.command(name="compare")
def compare_cmd(
    models: List[str] = typer.Argument(..., help="Model names or IDs to compare (e.g. qwen llama gemma)"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Compares candidate models side-by-side."""
    repo = DatabaseRepository()
    all_models = repo.get_all_models()
    selected = []

    for query in models:
        q_lower = query.lower()
        match = next((m for m in all_models if q_lower in m["id"].lower() or q_lower in m["canonical_name"].lower()), None)
        if match and match not in selected:
            selected.append(match)

    if not selected:
        console.print(f"[red]No models matched queries: {models}[/red]")
        return

    if as_json:
        sys.stdout.write(json.dumps(selected, indent=2) + "\n")
        return

    table = Table(title="MODEL COMPARISON MATRIX", box=ROUNDED, header_style="bold cyan")
    table.add_column("Attribute", style="bold white")
    for m in selected:
        table.add_column(m["canonical_name"], justify="center")

    table.add_row("Family", *[m["family_id"].replace("-", " ").title() for m in selected])
    table.add_row("Total Parameters", *[f"{m['total_parameters'] / 1e9:.1f}B" for m in selected])
    table.add_row("Active Parameters", *[f"{m['active_parameters'] / 1e9:.1f}B" for m in selected])
    table.add_row("Architecture", *[m["architecture_type"] for m in selected])
    table.add_row("Context Length", *[f"{m['context_length'] // 1024}k tokens" for m in selected])
    table.add_row("Reasoning", *["✓ Yes" if m["has_reasoning"] else "✗ No" for m in selected])
    table.add_row("Coding", *["✓ Yes" if m["has_coding"] else "✗ No" for m in selected])
    table.add_row("Vision", *["✓ Yes" if m["has_vision"] else "✗ No" for m in selected])
    table.add_row("Ollama Tag", *[m["ollama_name"] or "None" for m in selected])

    console.print(table)


# Database subcommand group
db_app = typer.Typer(help="Database management: import and validate dataset.json.")


@db_app.command(name="import")
def db_import_cmd(
    file_path: Optional[str] = typer.Argument(None, help="Path to dataset.json"),
) -> None:
    """Imports dataset.json into SQLite database idempotently."""
    val_res, stats = import_dataset_to_db(file_path)
    if not val_res.is_valid:
        console.print(f"[bold red]Import failed: Dataset has {len(val_res.errors)} validation errors.[/bold red]")
        for e in val_res.errors:
            console.print(f"  - [{e.category}] {e.field}: {e.message}")
        raise typer.Exit(1)

    console.print("[bold green]✓ Successfully imported dataset.json into SQLite![/bold green]")
    console.print(f"Records imported: {stats['cpus']} CPUs, {stats['gpus']} GPUs, {stats['models']} Models, {stats['benchmarks']} Benchmarks.")


@db_app.command(name="validate")
def db_validate_cmd(
    file_path: Optional[str] = typer.Argument(None, help="Path to dataset.json"),
) -> None:
    """Validates dataset.json schema, bounds, and consistency."""
    try:
        catalog = load_dataset(file_path)
        val = validate_catalog(catalog)
        console.print(f"[bold]{val.summary()}[/bold]")
        if val.errors:
            console.print("[bold red]Errors:[/bold red]")
            for e in val.errors:
                console.print(f"  ✗ [{e.category}] {e.field}: {e.message}")
        if val.warnings:
            console.print("[bold yellow]Warnings:[/bold yellow]")
            for w in val.warnings:
                console.print(f"  ⚠ [{w.category}] {w.field}: {w.message}")
        if not val.is_valid:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Validation failed: {e}[/bold red]")
        raise typer.Exit(1)


cmd_app.add_typer(db_app, name="database")


@cmd_app.command(name="refresh")
def refresh_cmd() -> None:
    """Reloads the local dataset into the catalogue cache."""
    console.print("[cyan]Reloading the local dataset...[/cyan]")
    repo = DatabaseRepository()
    try:
        catalog = load_dataset()
        val = validate_catalog(catalog)
        if not val.is_valid:
            raise ValueError(val.summary())
        stats = repo.import_catalog(catalog)
        console.print("✓ dataset.json validated")
        console.print(f"✓ Processors catalogue: {stats['cpus']} records")
        console.print(f"✓ GPU catalogue: {stats['gpus']} records")
        console.print(f"✓ Model catalogue: {stats['models']} models")
        console.print(f"✓ Benchmark evidence: {stats['benchmarks']} records")
        console.print("[bold green]Local catalogue reloaded.[/bold green]")
    except Exception as e:
        console.print(f"[red]Error during refresh: {e}[/red]")
        raise typer.Exit(1)


@cmd_app.command(name="web")
def web_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind (default 127.0.0.1)"),
    port: int = typer.Option(1234, "--port", help="Port to listen on (default 1234)"),
) -> None:
    """Starts the local web interface."""
    import uvicorn
    console.print(f"[bold green]Starting ModelScout Web Dashboard on http://{host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    uvicorn.run("modelscout.web.app:app", host=host, port=port, log_level="info")


@cmd_app.command(name="plan")
def plan_cmd(
    model_name: str = typer.Argument(..., help="Model name to plan requirements for (e.g. 'llama 70b', 'qwen 32b')"),
) -> None:
    """Plans hardware requirements for running a target model."""
    repo = DatabaseRepository()
    models = repo.get_all_models()
    tokens = model_name.lower().replace("-", " ").split()
    match = next(
        (
            m
            for m in models
            if all(
                t in m["canonical_name"].lower().replace("-", " ")
                or t in m["id"].lower().replace("-", " ")
                for t in tokens
            )
        ),
        None,
    )
    if not match:
        console.print(f"[red]Model '{model_name}' not found in catalogue.[/red]")
        return

    hw = detect_system_hardware()
    p_b = match["total_parameters"] / 1e9

    # Q4 requirement
    q4_req = round((match["total_parameters"] * 4.5 / 8.0) / (1024**3) + 2.0, 1)
    q8_req = round((match["total_parameters"] * 8.5 / 8.0) / (1024**3) + 3.0, 1)

    console.print(Panel(f"[bold cyan]HARDWARE EXECUTION PLAN: {match['canonical_name']}[/bold cyan]", box=ROUNDED))
    console.print(f"Parameters: {p_b:.1f}B ({match['architecture_type']})")
    console.print(f"Memory Needed (Q4_K_M): ~{q4_req:.1f} GB VRAM / Unified Memory")
    console.print(f"Memory Needed (Q8_0):   ~{q8_req:.1f} GB VRAM / Unified Memory")
    console.print(f"\nYour Current System: {hw.cpu.name}, {hw.memory.total_ram_gb:.1f} GB RAM, GPU: {hw.primary_gpu.name if hw.primary_gpu else 'Integrated'}")

    avail_vram = hw.total_effective_vram_gb
    if avail_vram >= q4_req:
        console.print(f"[bold green]✓ Compatible:[/bold green] Fits comfortably on your current system ({q4_req:.1f} GB required vs {avail_vram:.1f} GB available)!")
    else:
        gap = q4_req - avail_vram
        console.print(f"[bold yellow]⚠ Hardware Upgrade Needed:[/bold yellow] Deficit of ~{gap:.1f} GB VRAM.")
        console.print(f"[bold]Recommended hardware:[/bold] {q4_req:.0f}+ GB VRAM (e.g., RTX 4090/5090 or Apple Silicon with {int(q4_req * 1.3)}+ GB unified memory).")


@cmd_app.command(name="upgrade")
def upgrade_cmd() -> None:
    """Analyzes potential GPU/RAM upgrades and unlocked model performance."""
    hw = detect_system_hardware()
    console.print(Panel("[bold cyan]AI HARDWARE UPGRADE ADVISOR[/bold cyan]", box=ROUNDED))
    console.print(f"Current Hardware: {hw.cpu.name}, {hw.memory.total_ram_gb:.1f} GB RAM")
    if hw.gpus:
        console.print(f"Current GPU: {hw.primary_gpu.name} ({hw.primary_gpu.vram_gb:.1f} GB VRAM)")

    table = Table(title="UPGRADE PATHS & UNLOCKED CAPABILITIES", box=ROUNDED, header_style="bold green")
    table.add_column("Upgrade Option", style="bold white")
    table.add_column("VRAM / Memory", justify="center")
    table.add_column("Est. Speedup", justify="center", style="yellow")
    table.add_column("Unlocked Models", style="cyan")

    table.add_row("Apple Mac Studio (M4 Max)", "64 GB Unified", "+2.5x Bandwidth", "70B models at ~18–22 tok/s, 32B at ~35 tok/s")
    table.add_row("Apple Mac Studio (M4 Ultra)", "128 GB Unified", "+5.0x Bandwidth", "405B models (Q4), full DeepSeek R1 70B at Q8")
    table.add_row("NVIDIA GeForce RTX 4070 Ti SUPER", "16 GB VRAM", "+2.0x Bandwidth", "14B at ~45 tok/s, 32B (Q4) partial offload")
    table.add_row("NVIDIA GeForce RTX 4090", "24 GB VRAM", "+4.0x Bandwidth", "32B models in full VRAM at ~35 tok/s, 70B offload")
    table.add_row("NVIDIA GeForce RTX 5090", "32 GB VRAM", "+7.0x Bandwidth", "70B models (Q3/Q4) near-full VRAM at ~28 tok/s")

    console.print(table)


@cmd_app.command(name="run")
def run_cmd(
    model: Optional[str] = typer.Argument(None, help="Model name or tag to run (e.g. 'llama3.2', 'qwen', 'deepseek-r1')"),
    quant: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization format to target"),
) -> None:
    """One-command chat: downloads (if needed) and starts an interactive chat session instantly."""
    repo = DatabaseRepository()
    all_models = repo.get_all_models()

    target_tag = None
    target_name = None

    if model:
        q = model.lower().strip()
        if ":" in q:
            target_tag = model
            target_name = model
        else:
            match = next(
                (
                    m for m in all_models
                    if q in m["id"].lower()
                    or q in m["canonical_name"].lower()
                    or (m.get("ollama_name") and q in m["ollama_name"].lower())
                ),
                None,
            )
            if match:
                target_tag = match.get("ollama_name") or match["canonical_name"].lower().replace(" ", "-")
                target_name = match["canonical_name"]
            else:
                target_tag = model
                target_name = model
    else:
        hw = detect_system_hardware()
        report = generate_recommendations(hardware=hw, top_n=5, repo=repo)
        if report.best_overall:
            target_tag = report.best_overall.ollama_name or report.best_overall.model_id
            target_name = report.best_overall.display_name
            console.print(f"[bold green]⚡ Auto-selected best matching model for your system:[/bold green] [bold cyan]{target_name}[/bold cyan] ({target_tag})")
        else:
            console.print("[red]No compatible local models found for your hardware.[/red]")
            return

    ollama_path = shutil.which("ollama")
    if ollama_path:
        console.print(f"[bold cyan]Launching one-command chat session with {target_name}...[/bold cyan]")
        console.print(f"[dim]Running 'ollama run {target_tag}' (type /bye or press Ctrl+D to exit)[/dim]\n")
        try:
            res = subprocess.call([ollama_path, "run", target_tag])
            if res != 0:
                console.print(f"[yellow]Ollama session closed with code {res}. If Ollama daemon is stopped, start it with 'ollama serve'.[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[dim]Chat session ended.[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to launch chat: {e}[/red]")
        return

    llama_cli = shutil.which("llama-cli")
    if llama_cli:
        console.print(f"[bold cyan]Launching llama-cli session for {target_name}...[/bold cyan]")
        try:
            subprocess.call([llama_cli, "-p", "You are a helpful assistant.", "--conversation"])
        except Exception as e:
            console.print(f"[red]Failed to run llama-cli: {e}[/red]")
        return

    console.print(f"[yellow]To start an instant chat with {target_name}, install Ollama:[/yellow]")
    console.print("  [bold white]curl -fsSL https://ollama.com/install.sh | sh[/bold white]  (or [bold white]brew install ollama[/bold white])")
    console.print(f"Then run: [bold cyan]modelscout run {target_tag}[/bold cyan] or [bold cyan]ollama run {target_tag}[/bold cyan]\n")


@cmd_app.command(name="snippet")
def snippet_cmd(
    model: Optional[str] = typer.Argument(None, help="Model name or query (e.g. 'llama3.1', 'qwen', 'deepseek')"),
    runner: str = typer.Option("all", "--runner", "-r", help="Runner snippet: 'ollama', 'openai', 'transformers', 'llama-cpp', or 'all'"),
) -> None:
    """Prints ready-to-run Python code snippet for any model."""
    repo = DatabaseRepository()
    all_models = repo.get_all_models()

    match = None
    if model:
        q = model.lower().strip()
        match = next(
            (
                m for m in all_models
                if q in m["id"].lower()
                or q in m["canonical_name"].lower()
                or (m.get("ollama_name") and q in m["ollama_name"].lower())
            ),
            None,
        )
    if not match:
        hw = detect_system_hardware()
        report = generate_recommendations(hardware=hw, top_n=1, repo=repo)
        if report.best_overall:
            match = next((m for m in all_models if m["id"] == report.best_overall.model_id), None)

    m_name = match["canonical_name"] if match else (model or "Llama-3.2-3B-Instruct")
    ollama_tag = (match.get("ollama_name") if match else None) or "llama3.2:3b"
    hf_id = (match.get("huggingface_id") if match else None) or f"meta-llama/{m_name}"

    console.print(Panel(f"[bold cyan]READY-TO-RUN PYTHON SNIPPETS: {m_name}[/bold cyan]", box=ROUNDED))

    r_lower = runner.lower()
    if r_lower in ["ollama", "all"]:
        console.print("[bold green]1. Official Ollama Python SDK (`pip install ollama`):[/bold green]")
        ollama_code = f'''import ollama

response = ollama.chat(
    model="{ollama_tag}",
    messages=[
        {{"role": "system", "content": "You are a helpful and concise assistant."}},
        {{"role": "user", "content": "Explain quantum computing in three sentences."}},
    ],
)
print(response["message"]["content"])'''
        console.print(Syntax(ollama_code, "python", theme="monokai", line_numbers=True))
        console.print()

    if r_lower in ["openai", "all"]:
        console.print("[bold green]2. OpenAI-Compatible Client (`pip install openai`):[/bold green]")
        openai_code = f'''from openai import OpenAI

# Works out-of-the-box with Ollama, vLLM, or llama-server
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="{ollama_tag}",
    messages=[
        {{"role": "system", "content": "You are a helpful and concise assistant."}},
        {{"role": "user", "content": "Explain quantum computing in three sentences."}},
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)'''
        console.print(Syntax(openai_code, "python", theme="monokai", line_numbers=True))
        console.print()

    if r_lower in ["llama-cpp", "llama_cpp", "all"]:
        console.print("[bold green]3. llama-cpp-python (`pip install llama-cpp-python`):[/bold green]")
        llama_cpp_code = f'''from llama_cpp import Llama

# Automatically downloads and loads GGUF with GPU offload
llm = Llama.from_pretrained(
    repo_id="{hf_id}",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1,  # Offload all layers to Metal / CUDA
    n_ctx=4096,
)

output = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Explain quantum computing in three sentences."}}]
)
print(output["choices"][0]["message"]["content"])'''
        console.print(Syntax(llama_cpp_code, "python", theme="monokai", line_numbers=True))
        console.print()

    if r_lower in ["transformers", "all"]:
        console.print("[bold green]4. Hugging Face Transformers (`pip install transformers torch`):[/bold green]")
        transformers_code = f'''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{hf_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

inputs = tokenizer("Explain quantum computing in three sentences.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))'''
        console.print(Syntax(transformers_code, "python", theme="monokai", line_numbers=True))
        console.print()


@cmd_app.command(name="update")
def update_cmd(
    timeout: float = typer.Option(8.0, "--timeout", "-t", help="Timeout in seconds (<10s)"),
) -> None:
    """Syncs model catalogue from open endpoints (Hugging Face Hub and local Ollama)."""
    console.print(f"[cyan]Querying open endpoints (Hugging Face & Ollama, timeout={timeout}s)...[/cyan]")
    repo = DatabaseRepository()
    try:
        stats = sync_models_from_open_apis(timeout=timeout, repo=repo)
        console.print(f"✓ Hugging Face models queried: [bold]{stats['hf_found']}[/bold]")
        console.print(f"✓ Local Ollama tags inspected: [bold]{stats['ollama_found']}[/bold]")
        if stats.get("updated_models", 0) > 0:
            console.print(f"[bold cyan]✓ Refreshed versions and metrics for {stats['updated_models']} stored models.[/bold cyan]")
        if stats["added_models"] > 0:
            console.print(f"[bold green]✓ Added {stats['added_models']} newly discovered models to stored dataset and catalogue![/bold green]")
        elif stats.get("updated_models", 0) == 0:
            console.print("[dim]Stored model dataset and catalogue are up to date with latest online models.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Update completed with notice: {e}[/yellow]")
        console.print("[dim]Local stored dataset remains fully operational.[/dim]")
