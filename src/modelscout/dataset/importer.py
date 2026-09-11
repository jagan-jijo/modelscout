"""Importer module to load, validate, and store dataset.json into SQLite."""

from pathlib import Path
from typing import Optional, Tuple
from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.loader import load_dataset
from modelscout.dataset.validator import ValidationResult, validate_catalog


def import_dataset_to_db(
    file_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
) -> Tuple[ValidationResult, dict]:
    """Loads dataset.json, validates it, and imports all records into SQLite idempotently."""
    catalog = load_dataset(file_path)
    val_res = validate_catalog(catalog)
    if not val_res.is_valid:
        return val_res, {}

    repo = DatabaseRepository(db_path)
    stats = repo.import_catalog(catalog)
    return val_res, stats
