"""CLI formatting and presentation using Rich and JSON serialization."""

import json
import sys
from typing import Any, Dict, List
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modelscout.hardware.types import SystemHardware
from modelscout.recommendation.ranking import ModelRecommendation, RecommendationReport
from modelscout.output.runtime_models import (
    display_runtime_models_json,
    display_runtime_models_terminal,
    render_runtime_models_json,
    render_runtime_models_terminal,
)

console = Console()
err_console = Console(stderr=True)

def render_scan_json(report: RecommendationReport) -> None:
    """Outputs strictly valid JSON on stdout for scripting/jq."""
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    sys.stdout.flush()


def render_scan_terminal(report: RecommendationReport) -> None:
    """Renders rich terminal interface with header, hardware score, and recommendations."""
    hw = report.hardware

    # Header Panel
    header_text = Text()
    header_text.append("MODELSCOUT\n", style="bold cyan")
    header_text.append("Find local AI models that fit your hardware\n", style="dim")
    header_text.append("Created by Jagan Jijo • https://jagan-jijo.github.io/portfolio/\n\n", style="cyan")
    header_text.append(f"OS: {hw.os_name} {hw.os_version} ({hw.architecture})  •  ", style="white")
    header_text.append(f"CPU: {hw.cpu.name}  •  ", style="white")
    header_text.append(f"RAM: {hw.memory.total_ram_gb:.1f} GB ({'Unified' if hw.memory.unified_memory else 'System RAM'})\n", style="white")
    if hw.gpus:
        for g in hw.gpus:
            header_text.append(f"GPU: {g.name} ({g.vram_gb:.1f} GB VRAM, {g.memory_bandwidth_gbps or 0:.0f} GB/s bandwidth)\n", style="green")
    header_text.append(f"Storage: {hw.storage.available_gb:.1f} GB free  •  ", style="dim")
    header_text.append(
        f"Runtimes: {'✓ Metal ' if hw.runtimes.metal_available else ''}"
        f"{'✓ CUDA ' if hw.runtimes.cuda_available else ''}"
        f"{'✓ Ollama ' if hw.runtimes.ollama_installed else ''}"
        f"{'✓ llama.cpp ' if hw.runtimes.llama_cpp_installed else ''}",
        style="yellow",
    )
    console.print(Panel(header_text, border_style="cyan", box=ROUNDED))

    # Hardware Score Card
    if hw.hardware_score:
        hs = hw.hardware_score
        score_table = Table(box=ROUNDED, show_header=True, header_style="bold magenta", expand=False)
        score_table.add_column("AI Hardware Score", justify="center", style="bold yellow")
        score_table.add_column("GPU", justify="center")
        score_table.add_column("Memory Capacity", justify="center")
        score_table.add_column("Bandwidth", justify="center")
        score_table.add_column("CPU", justify="center")
        score_table.add_column("AI Readiness", justify="center")

        score_table.add_row(
            f"{hs.overall_score} / 100",
            f"{hs.gpu_score}/100",
            f"{hs.memory_capacity_score}/100",
            f"{hs.memory_bandwidth_score}/100",
            f"{hs.cpu_score}/100",
            f"{hs.ai_readiness_score}/100",
        )
        console.print(score_table)
        console.print(f"[dim]{hs.summary}[/dim]\n")

    # Recommendations Table
    table = Table(
        title=f"RECOMMENDED MODELS FOR YOUR SYSTEM (Profile: {report.profile.upper()})",
        box=ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="bold white")
    table.add_column("Size (GB)", justify="center", style="bold cyan")
    table.add_column("Fit", justify="center")
    table.add_column("Est. Speed", justify="center", style="green")
    table.add_column("RAM/VRAM", justify="center")
    table.add_column("Benchmark", justify="center")
    table.add_column("Score", justify="center", style="bold yellow")

    for i, r in enumerate(report.recommendations, 1):
        fit_style = "green" if "Full" in r.fit_label or "Unified" in r.fit_label else ("yellow" if "Partial" in r.fit_label else "red")
        table.add_row(
            str(i),
            f"{r.display_name}\n[dim]{r.quantization} • {r.parameters / 1e9:.1f}B params[/dim]",
            f"{r.file_size_gb:.1f} GB",
            f"[{fit_style}]{r.fit_label}[/{fit_style}]",
            f"{r.speed_display}\n[dim]{r.speed_confidence} Conf[/dim]",
            f"{r.vram_required_gb:.1f} GB",
            f"{r.benchmark_score:.1f} ({r.benchmark_source})\n[dim]{r.benchmark_evidence.title()}[/dim]",
            f"[bold yellow]{r.recommendation_score:.1f}[/bold yellow]",
        )
    console.print(table)

    # Category Highlights
    console.print("\n[bold cyan]KEY HIGHLIGHTS[/bold cyan]")
    if report.best_overall:
        console.print(f"★ [bold]Best Overall:[/bold] {report.best_overall.display_name} (Score: {report.best_overall.recommendation_score})")
    if report.best_quality:
        console.print(f"✦ [bold]Highest Quality:[/bold] {report.best_quality.display_name} (Benchmark: {report.best_quality.benchmark_score})")
    if report.fastest:
        console.print(f"⚡ [bold]Fastest Candidate:[/bold] {report.fastest.display_name} ({report.fastest.speed_display})")
    if report.best_coding:
        console.print(f"⌨ [bold]Best for Coding:[/bold] {report.best_coding.display_name}")
    if report.best_reasoning:
        console.print(f"🧠 [bold]Best for Reasoning:[/bold] {report.best_reasoning.display_name}")
    if report.best_vision:
        console.print(f"👁 [bold]Best for Vision:[/bold] {report.best_vision.display_name}")

    # Excluded preview
    if report.excluded:
        console.print(f"\n[bold red]EXCLUDED CANDIDATES ({len(report.excluded)} models)[/bold red]")
        for ex in report.excluded[:4]:
            console.print(f"  ✗ [bold]{ex.display_name}[/bold]: [dim]{ex.reason}[/dim]")
        if len(report.excluded) > 4:
            console.print(f"  [dim]... and {len(report.excluded) - 4} more excluded models.[/dim]")

    console.print("\n[dim]Run 'modelscout --help' to see all profiles, filters, and simulation options.[/dim]\n")


def render_scan_markdown(report: RecommendationReport) -> None:
    """Renders recommendations as a GitHub-flavored Markdown table."""
    md = [
        f"# ModelScout Recommendation Report",
        f"**Generated:** {report.generated_at}",
        f"**Hardware Score:** {report.hardware_score}/100",
        f"**Profile:** {report.profile}",
        "",
        "## Recommended Models",
        "| # | Model | Size (GB) | Fit | Est. Speed | Memory | Benchmark | Score |",
        "|---|-------|-----------|-----|------------|--------|-----------|-------|",
    ]
    for i, r in enumerate(report.recommendations, 1):
        md.append(
            f"| {i} | **{r.display_name}** ({r.quantization}) | {r.file_size_gb:.1f} GB | {r.fit_label} | {r.speed_display} | {r.vram_required_gb:.1f} GB | {r.benchmark_score:.1f} ({r.benchmark_source}) | **{r.recommendation_score:.1f}** |"
        )
    md.append("")
    sys.stdout.write("\n".join(md) + "\n")
