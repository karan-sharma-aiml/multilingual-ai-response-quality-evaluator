"""Stored result endpoints: list, read, delete."""

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import get_database_service
from app.core.exceptions import InvalidParameterError
from app.repositories import ResultFilters
from app.schemas.api import (
    DEFAULT_PAGE_SIZE,
    MAX_CATEGORY_LENGTH,
    MAX_PAGE_SIZE,
    DeleteResponse,
    ErrorResponse,
    EvaluationMethod,
    EvaluationResponse,
    PaginatedResults,
    SortField,
    SortOrder,
    validate_language,
)
from app.services.database_service import DatabaseService
from app.services.mappers import record_to_response

router = APIRouter(tags=["Results"])


@router.get("/results", response_model=PaginatedResults, summary="List stored evaluations")
def list_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: SortField = Query(SortField.CREATED_AT),
    order: SortOrder = Query(SortOrder.DESC),
    language: str | None = Query(None, description="Language name or code, e.g. 'fr' or 'French'."),
    evaluation_method: EvaluationMethod | None = Query(None),
    category: str | None = Query(None, max_length=MAX_CATEGORY_LENGTH),
    min_overall_score: float | None = Query(None, ge=0, le=100),
    max_overall_score: float | None = Query(None, ge=0, le=100),
    database: DatabaseService = Depends(get_database_service),
) -> PaginatedResults:
    """Paginated, sortable and filterable list of stored evaluations."""
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

    filters = ResultFilters(
        language_code=language_code,
        evaluation_method=evaluation_method.value if evaluation_method else None,
        category=category,
        min_overall_score=min_overall_score,
        max_overall_score=max_overall_score,
    )
    return database.list_results(filters, page, page_size, sort_by, order)


@router.get(
    "/results/{result_id}",
    response_model=EvaluationResponse,
    summary="Get one stored evaluation",
    responses={404: {"model": ErrorResponse}},
)
def get_result(
    result_id: int = Path(ge=1),
    database: DatabaseService = Depends(get_database_service),
) -> EvaluationResponse:
    return record_to_response(database.get_result(result_id))


@router.delete(
    "/results/{result_id}",
    response_model=DeleteResponse,
    summary="Delete a stored evaluation",
    responses={404: {"model": ErrorResponse}},
)
def delete_result(
    result_id: int = Path(ge=1),
    database: DatabaseService = Depends(get_database_service),
) -> DeleteResponse:
    database.delete_result(result_id)
    return DeleteResponse(id=result_id, message=f"Evaluation result {result_id} deleted.")
