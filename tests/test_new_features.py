"""Tests for new features: multi-GPU simulation, memory budgets, snippets, run, and background updater."""

from typer.testing import CliRunner

from modelscout.cli.main import app
from modelscout.dataset.updater import sync_models_from_open_apis
from modelscout.hardware.detector import detect_system_hardware

runner = CliRunner()


def test_multi_gpu_simulation_multipliers():
    hw = detect_system_hardware(gpu_override="2x RTX 4090")
    assert len(hw.gpus) == 2
    assert hw.gpus[0].vram_gb == 24.0
    assert hw.gpus[1].vram_gb == 24.0
    assert hw.total_effective_vram_gb == 48.0


def test_multi_gpu_simulation_commas():
    hw = detect_system_hardware(gpu_override="RTX 4090, RTX 4090")
    assert len(hw.gpus) == 2
    assert hw.total_effective_vram_gb == 48.0


def test_multi_gpu_simulation_repeated():
    hw = detect_system_hardware(gpu_override=["RTX 4090", "RTX 4090"])
    assert len(hw.gpus) == 2
    assert hw.total_effective_vram_gb == 48.0


def test_cli_vram_headroom_and_ram_budget():
    result = runner.invoke(app, ["--gpu", "RTX 4090", "--vram-headroom", "10.0", "--json"])
    assert result.exit_code == 0
    assert '"vram_required_gb"' in result.output


def test_cli_snippet_command():
    result = runner.invoke(app, ["snippet", "llama", "--runner", "ollama"])
    assert result.exit_code == 0
    assert "import ollama" in result.output
    assert "ollama.chat" in result.output


def test_cli_plan_command_flexible_query():
    result = runner.invoke(app, ["plan", "llama 3 70b"])
    assert result.exit_code == 0
    assert "Llama-3.3-70B-Instruct" in result.output or "Llama-3.1-70B-Instruct" in result.output
    assert "HARDWARE EXECUTION PLAN" in result.output


def test_cli_upgrade_command():
    result = runner.invoke(app, ["upgrade"])
    assert result.exit_code == 0
    assert "UPGRADE PATHS & UNLOCKED CAPABILITIES" in result.output
    assert "5090" in result.output
    assert "4090" in result.output


def test_cli_markdown_flag():
    result = runner.invoke(app, ["-m", "--top", "2"])
    assert result.exit_code == 0
    assert "| Model |" in result.output
    assert "# ModelScout Recommendation Report" in result.output


def test_live_updater_timeout_guard():
    stats = sync_models_from_open_apis(timeout=1.0)
    assert isinstance(stats, dict)
    assert "hf_found" in stats
    assert "ollama_found" in stats
