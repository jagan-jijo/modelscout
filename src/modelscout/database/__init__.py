"""Database package for SQLite queries, models, and autocomplete caching."""

from modelscout.database.repository import DatabaseRepository, get_default_db_path

__all__ = ["DatabaseRepository", "get_default_db_path"]
