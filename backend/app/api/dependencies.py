"""FastAPI dependency providers (dependency injection)."""

from collections.abc import Iterator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.analytics.service import AnalyticsService
from app.core.cache import get_analytics_cache
from app.core.config import Settings, get_settings
from app.database.session import create_db_engine, create_session_factory
from app.evaluator import ResponseEvaluator
from app.services.batch_service import BatchEvaluationService
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationService
from app.services.gemini_service import GeminiEvaluationService
from app.services.report_service import ReportService


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_db_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_engine())


def get_db() -> Iterator[Session]:
    """Yield a request-scoped database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def get_rule_evaluator() -> ResponseEvaluator:
    return ResponseEvaluator()


@lru_cache(maxsize=1)
def get_gemini_service() -> GeminiEvaluationService:
    return GeminiEvaluationService(get_settings())


def get_database_service(session: Session = Depends(get_db)) -> DatabaseService:
    return DatabaseService(session)


def get_evaluation_service(
    database: DatabaseService = Depends(get_database_service),
    gemini: GeminiEvaluationService = Depends(get_gemini_service),
    settings: Settings = Depends(get_settings),
) -> EvaluationService:
    return EvaluationService(
        rule_evaluator=get_rule_evaluator(),
        gemini=gemini,
        database=database,
        settings=settings,
    )


def get_batch_service(
    evaluation: EvaluationService = Depends(get_evaluation_service),
    database: DatabaseService = Depends(get_database_service),
    settings: Settings = Depends(get_settings),
) -> BatchEvaluationService:
    return BatchEvaluationService(evaluation, database, settings)


def get_analytics_service(
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyticsService:
    cache = get_analytics_cache() if settings.cache_enabled else None
    return AnalyticsService(session, cache=cache, settings=settings)


def get_report_service(
    database: DatabaseService = Depends(get_database_service),
    settings: Settings = Depends(get_settings),
) -> ReportService:
    return ReportService(database, settings)


def shutdown_resources() -> None:
    """Release long-lived resources (called on application shutdown)."""
    if get_gemini_service.cache_info().currsize:
        get_gemini_service().close()
    if get_engine.cache_info().currsize:
        get_engine().dispose()
