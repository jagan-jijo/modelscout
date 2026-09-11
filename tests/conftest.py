"""Keep tests away from the user's catalogue cache."""

import pytest


@pytest.fixture(autouse=True)
def isolated_catalogue_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("MODELSCOUT_DATASET", raising=False)
