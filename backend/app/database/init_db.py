"""Database start-up logic: create tables and seed reference data."""

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.base import Base
from app.database.session import create_db_engine, create_session_factory, init_db
from app.repositories.language_repository import LanguageRepository
from app.utils.language import LANGUAGE_NAMES

logger = get_logger("database.init")


def seed_languages(session: Session) -> int:
    """Insert missing supported languages; returns the number created."""
    repository = LanguageRepository(session)
    created = 0
    for code, name in LANGUAGE_NAMES.items():
        if repository.get_by_code(code) is None:
            repository.add(_new_language(code, name))
            created += 1
    session.commit()
    return created


def _new_language(code: str, name: str):
    from app.models import Language

    return Language(code=code, name=name)


def ensure_indexes(engine: Engine) -> None:
    """Create indexes that are missing on tables created by an older version."""
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            index.create(bind=engine, checkfirst=True)


def initialize_database(engine: Engine | None = None) -> Engine:
    """Create all tables and indexes and seed languages. Safe to call repeatedly."""
    active_engine = engine or create_db_engine()
    init_db(active_engine)
    ensure_indexes(active_engine)
    with create_session_factory(active_engine)() as session:
        created = seed_languages(session)
    logger.info("Database ready (languages seeded: %d)", created)
    return active_engine


def main() -> None:
    """CLI entry point: ``python -m app.database.init_db``."""
    initialize_database()


if __name__ == "__main__":
    main()
