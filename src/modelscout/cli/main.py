"""ModelScout's command-line entry point."""

from typing import List, Optional, Union
import typer

from modelscout.cli.commands import cmd_app, scan_cmd

app = typer.Typer(
    help="Find local AI models that fit your hardware, with memory and speed estimates.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    profile: str = typer.Option("general", "--profile", "-p", help="Recommendation profile"),
    quant: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization format to target"),
    top: int = typer.Option(10, "--top", "-n", help="Top N models to recommend"),
    as_json: bool = typer.Option(False, "--json", help="Output strictly valid JSON"),
    as_markdown: bool = typer.Option(False, "--markdown", "-m", help="Output GitHub-flavored Markdown table"),
    gpu: Optional[List[str]] = typer.Option(None, "--gpu", help="Simulate GPU (repeat --gpu, use commas, or write '2x RTX 4090')"),
    cpu: Optional[str] = typer.Option(None, "--cpu", help="Simulate CPU"),
    ram: Optional[float] = typer.Option(None, "--ram", help="Simulate RAM in GB"),
    vram: Optional[float] = typer.Option(None, "--vram", help="Simulate VRAM in GB"),
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
    """Detects hardware and recommends models when invoked without subcommands."""
    if ctx.invoked_subcommand is None:
        scan_cmd(
            profile=profile,
            quant=quant,
            top=top,
            as_json=as_json,
            as_markdown=as_markdown,
            gpu=gpu,
            cpu=cpu,
            ram=ram,
            vram=vram,
            vram_headroom=vram_headroom,
            ram_budget=ram_budget,
            fit_filter=fit_filter,
            gpu_only=gpu_only,
            cpu_only=cpu_only,
            speed_filter=speed_filter,
            min_speed=min_speed,
            context_length=context_length,
            evidence=evidence,
            refresh=refresh,
        )


# Merge all subcommands into app
for cmd in cmd_app.registered_commands:
    app.registered_commands.append(cmd)

for sub_typer in cmd_app.registered_groups:
    app.registered_groups.append(sub_typer)


if __name__ == "__main__":
    app()
