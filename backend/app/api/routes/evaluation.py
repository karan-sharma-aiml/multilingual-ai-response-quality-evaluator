"""Single and pairwise evaluation endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_evaluation_service
from app.schemas.api import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluationResponse,
)
from app.services.evaluation_service import EvaluationService

router = APIRouter(tags=["Evaluation"])


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    status_code=201,
    summary="Evaluate one LLM response",
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def evaluate(
    request: EvaluateRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResponse:
    """Run the rule engine, then Gemini (when enabled and available), merge the
    scores, store the evaluation and return it.

    If Gemini is not configured or fails, the rule-based scores are used and
    ``gemini.error`` explains why (``evaluation_method`` is ``rule_based``).
    """
    return service.evaluate_and_save(request)


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two responses to the same question",
    responses={422: {"model": ErrorResponse}},
)
def compare(
    request: CompareRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> CompareResponse:
    """Evaluate both responses (each is stored) and return the better one,
    the per-metric score comparison and the reason."""
    return service.compare(request)
