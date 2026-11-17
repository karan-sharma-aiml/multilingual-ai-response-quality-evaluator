"""Conversions between ORM records and API schemas."""

from app.models import EvaluationResultRecord
from app.schemas.api import EvaluationMethod, EvaluationResponse, GeminiInfo


def record_to_response(record: EvaluationResultRecord) -> EvaluationResponse:
    """Flatten a result record (with its response and language) for the API."""
    response = record.response_record
    gemini_details = (record.details or {}).get("gemini") or {}
    return EvaluationResponse(
        id=record.id,
        response_id=response.id,
        language=response.language.code,
        language_name=response.language.name,
        category=response.category,
        question=response.question,
        response=response.response,
        reference_answer=response.reference_answer,
        relevance_score=record.relevance_score,
        correctness_score=record.correctness_score,
        clarity_score=record.clarity_score,
        safety_score=record.safety_score,
        hallucination_score=record.hallucination_score,
        duplicate_percentage=record.duplicate_percentage,
        overall_score=record.overall_score,
        evaluation_method=EvaluationMethod(record.evaluation_method),
        warnings=list(record.warnings or []),
        gemini=GeminiInfo(**gemini_details),
        created_at=record.created_at,
    )
