"""Analytics endpoints."""

from datetime import date

from fastapi import APIRouter, Depends, Query

from app.analytics.service import AnalyticsService
from app.api.dependencies import get_analytics_service
from app.core.exceptions import InvalidParameterError
from app.schemas.analytics import (
    CategoryAnalytics,
    HallucinationAnalytics,
    LanguageAnalytics,
    LowQualityResponse,
    OverviewStats,
    TrendPeriod,
    TrendsResponse,
)
from app.schemas.api import (
    DEFAULT_PAGE_SIZE,
    MAX_CATEGORY_LENGTH,
    MAX_PAGE_SIZE,
    ErrorResponse,
    validate_language,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", response_model=OverviewStats, summary="Global quality statistics")
def overview(service: AnalyticsService = Depends(get_analytics_service)) -> OverviewStats:
    """Totals and average scores across all stored evaluations."""
    return service.overview()


@router.get("/languages", response_model=LanguageAnalytics, summary="Statistics per language")
def languages(service: AnalyticsService = Depends(get_analytics_service)) -> LanguageAnalytics:
    """Per-language totals and averages with the best and worst language."""
    return service.languages()


@router.get("/categories", response_model=CategoryAnalytics, summary="Statistics per category")
def categories(service: AnalyticsService = Depends(get_analytics_service)) -> CategoryAnalytics:
    return service.categories()


@router.get(
    "/hallucinations",
    response_model=HallucinationAnalytics,
    summary="Most common hallucination types",
)
def hallucinations(service: AnalyticsService = Depends(get_analytics_service)) -> HallucinationAnalytics:
    """Frequency and percentage of each detected hallucination type."""
    return service.hallucinations()


@router.get(
    "/low-quality",
    response_model=LowQualityResponse,
    summary="Evaluations below the quality threshold",
    responses={422: {"model": ErrorResponse}},
)
def low_quality(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    threshold: float | None = Query(None, ge=0, le=100, description="Defaults to 50."),
    language: str | None = Query(None),
    category: str | None = Query(None, max_length=MAX_CATEGORY_LENGTH),
    service: AnalyticsService = Depends(get_analytics_service),
) -> LowQualityResponse:
    """Responses whose overall score is below the threshold (worst first)."""
    language_code = None
    if language:
        try:
            language_code = validate_language(language)
        except ValueError as exc:
            raise InvalidParameterError(str(exc)) from exc
    return service.low_quality(page, page_size, threshold, language_code, category or None)


@router.get(
    "/trends",
    response_model=TrendsResponse,
    summary="Daily, weekly and monthly evaluation trends",
    responses={422: {"model": ErrorResponse}},
)
def trends(
    period: TrendPeriod = Query(TrendPeriod.ALL),
    start_date: date | None = Query(None, description="Inclusive (UTC)."),
    end_date: date | None = Query(None, description="Inclusive (UTC)."),
    service: AnalyticsService = Depends(get_analytics_service),
) -> TrendsResponse:
    """Evaluation counts and average scores grouped by day, ISO week and month."""
    if start_date and end_date and start_date > end_date:
        raise InvalidParameterError("start_date cannot be after end_date.")
    return service.trends(period, start_date, end_date)
