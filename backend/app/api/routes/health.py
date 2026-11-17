"""Health-check endpoint."""

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_database_service, get_gemini_service
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.schemas.api import ComponentHealth, HealthResponse
from app.services.database_service import DatabaseService
from app.services.gemini_service import GeminiEvaluationService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health(
    database: DatabaseService = Depends(get_database_service),
    gemini: GeminiEvaluationService = Depends(get_gemini_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Report API, database and Gemini configuration status."""
    from app.main import APP_VERSION

    try:
        await run_in_threadpool(database.ping)
        db_health = ComponentHealth(status="ok")
    except AppError as exc:
        db_health = ComponentHealth(status="error", detail=exc.message)

    if gemini.is_configured:
        gemini_health = ComponentHealth(status="ok", detail=f"model={gemini.model}")
    else:
        gemini_health = ComponentHealth(
            status="not_configured",
            detail="Set GEMINI_API_KEY to enable AI evaluation; rule-based scoring still works.",
        )
    return HealthResponse(
        status="ok" if db_health.status == "ok" else "degraded",
        app_name=settings.app_name,
        version=APP_VERSION,
        database=db_health,
        gemini=gemini_health,
    )
