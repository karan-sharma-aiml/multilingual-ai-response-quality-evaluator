"""Engine and session factory helpers (SQLite by default)."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.database.base import Base

SQLITE_PREFIX: str = "sqlite"
SQLITE_MEMORY_MARKER: str = ":memory:"


def create_db_engine(database_url: str | None = None, echo: bool | None = None) -> Engine:
    """Create a SQLAlchemy engine for ``database_url`` (defaults to settings)."""
    settings = get_settings()
    url = database_url or settings.database_url
    is_sqlite = url.startswith(SQLITE_PREFIX)
    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    if not is_sqlite:
        engine_kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )
    if is_sqlite:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", f"sqlite:///{SQLITE_MEMORY_MARKER}"):
            engine_kwargs["poolclass"] = StaticPool
    engine = create_engine(
        url,
        echo=settings.debug if echo is None else echo,
        **engine_kwargs,
    )
    if is_sqlite:
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite ignores foreign keys unless the pragma is enabled per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables that are registered on the declarative base."""
    import app.models  # noqa: F401  (registers ORM models on Base.metadata)

    Base.metadata.create_all(bind=engine)
