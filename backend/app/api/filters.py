"""Shared query-parameter dependencies for filtered endpoints."""

from datetime import date, datetime, time, timedelta, timezone

from fastapi import Query

from app.core.exceptions import InvalidParameterError
from app.repositories import ResultFilters
from app.schemas.api import MAX_CATEGORY_LENGTH, EvaluationMethod, validate_language


def get_result_filters(
    language: str | None = Query(None, description="Language name or code."),
    category: str | None = Query(None, max_length=MAX_CATEGORY_LENGTH),
    evaluation_method: EvaluationMethod | None = Query(None),
    min_overall_score: float | None = Query(None, ge=0, le=100),
    max_overall_score: float | None = Query(None, ge=0, le=100),
    start_date: date | None = Query(None, description="Inclusive (UTC)."),
    end_date: date | None = Query(None, description="Inclusive (UTC)."),
) -> ResultFilters:
    """Build validated filters from query parameters."""
    language_code: str | None = None
    if language:
        try:
            language_code = validate_language(language)
        except ValueError as exc:
            raise InvalidParameterError(str(exc)) from exc
    if (
        min_overall_score is not None
        and max_overall_score is not None
        and min_overall_score > max_overall_score
    ):
        raise InvalidParameterError("min_overall_score cannot be greater than max_overall_score.")
    if start_date and end_date and start_date > end_date:
        raise InvalidParameterError("start_date cannot be after end_date.")
    return ResultFilters(
        language_code=language_code,
        evaluation_method=evaluation_method.value if evaluation_method else None,
        category=category or None,
        min_overall_score=min_overall_score,
        max_overall_score=max_overall_score,
        created_from=datetime.combine(start_date, time.min, tzinfo=timezone.utc) if start_date else None,
        created_to=(
            datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
            if end_date
            else None
        ),
    )
