"""ModelScout Typer CLI entry point."""

from modelscout.cli.main import app, main
from modelscout.cli.commands import (
    plan_cmd as plan,
    run_cmd as run,
    snippet_cmd as snippet,
    hardware_cmd as hardware,
    upgrade_cmd as upgrade,
    web_cmd as web,
    update_cmd as update,
    compare_cmd as compare,
    scan_cmd as scan,
)

__all__ = [
    "app",
    "main",
    "plan",
    "run",
    "snippet",
    "hardware",
    "upgrade",
    "web",
    "update",
    "compare",
    "scan",
]

if __name__ == "__main__":
    app()
