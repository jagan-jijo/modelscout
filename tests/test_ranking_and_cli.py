"""Tests for recommendation scoring, profiles, ranking, and CLI execution."""

import json
import pytest
from typer.testing import CliRunner

from modelscout.cli.main import app
from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.loader import load_dataset
from modelscout.hardware.detector import detect_system_hardware
from modelscout.recommendation.profiles import get_profile_weights
from modelscout.recommendation.ranking import generate_recommendations

runner = CliRunner()


def test_recommendation_profiles():
    coding = get_profile_weights("coding")
    assert coding.requires_coding is True
    assert coding.benchmark_weight > 0.30

    vision = get_profile_weights("vision")
    assert vision.requires_vision is True

    fast = get_profile_weights("fast")
    assert fast.min_speed_threshold >= 15.0


def test_recommendation_ranking_pipeline():
    repo = DatabaseRepository(":memory:")
    catalog = load_dataset()
    repo.import_catalog(catalog)

    hw = detect_system_hardware(ram_override_gb=32, vram_override_gb=16)
    report = generate_recommendations(hw, profile="general", repo=repo)

    assert len(report.recommendations) > 0
    assert report.best_overall is not None
    # Must be sorted in descending order of recommendation score
    scores = [r.recommendation_score for r in report.recommendations]
    assert scores == sorted(scores, reverse=True)


def test_cli_json_output():
    res = runner.invoke(app, ["--json"])
    assert res.exit_code == 0
    # Must parse strictly as valid JSON
    data = json.loads(res.output)
    assert "hardware" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_cli_hardware_command():
    res = runner.invoke(app, ["hardware", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert "cpu" in data
    assert "memory" in data
    assert "gpus" in data


def test_cli_models_and_benchmarks_commands():
    res_m = runner.invoke(app, ["models", "--json"])
    assert res_m.exit_code == 0
    assert len(json.loads(res_m.output)) > 0

    res_b = runner.invoke(app, ["benchmarks", "--json"])
    assert res_b.exit_code == 0
    assert len(json.loads(res_b.output)) > 0


def test_cli_database_validate():
    res = runner.invoke(app, ["database", "validate"])
    assert res.exit_code == 0
    assert "validation passed" in res.output.lower()
