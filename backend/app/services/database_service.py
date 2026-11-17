"""Persistence service wrapping the repositories with transactions and logging."""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.cache import invalidate_analytics_cache
from app.core.exceptions import DatabaseOperationError, ResultNotFoundError
from app.core.logging import get_logger
from app.models import EvaluationLog, EvaluationResultRecord, ResponseRecord
from app.repositories import (
    EvaluationLogRepository,
    EvaluationResultRepository,
    LanguageRepository,
    ResponseRepository,
    ResultFilters,
)
from app.schemas.api import EvaluationResponse, PaginatedResults, SortField, SortOrder
from app.schemas.evaluation import EvaluationInput
from app.services.mappers import record_to_response
from app.utils.language import LANGUAGE_NAMES

if TYPE_CHECKING:
    from app.services.evaluation_service import EvaluationOutcome

logger = get_logger("database")
MS_PER_SECOND: float = 1000.0


class DatabaseService:
    """High-level persistence operations used by the API layer."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._languages = LanguageRepository(session)
        self._responses = ResponseRepository(session)
        self._results = EvaluationResultRepository(session)
        self._logs = EvaluationLogRepository(session)

    @property
    def logs(self) -> EvaluationLogRepository:
        return self._logs

    @contextmanager
    def _transaction(self, operation: str, commit: bool = True) -> Iterator[None]:
        try:
            yield
            if commit:
                self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            logger.exception("Database error during %s", operation)
            raise DatabaseOperationError(f"Database error while {operation}.") from exc
        except Exception:
            self._session.rollback()
            raise

    def ping(self) -> None:
        with self._transaction("checking connectivity", commit=False):
            self._session.execute(text("SELECT 1"))

    def save_evaluation(
        self,
        sample: EvaluationInput,
        outcome: "EvaluationOutcome",
        category: str | None = None,
    ) -> EvaluationResultRecord:
        """Insert the response, its scores and audit-log rows in one transaction."""
        started = time.perf_counter()
        result = outcome.result
        with self._transaction("saving the evaluation"):
            code = sample.language_code
            language = self._languages.get_or_create(code, LANGUAGE_NAMES.get(code, code))
            response = ResponseRecord(
                language=language,
                question=sample.question,
                response=sample.response,
                reference_answer=sample.reference_answer,
                category=category,
            )
            self._responses.add(response)
            record = self._results.add(
                EvaluationResultRecord(
                    response_record=response,
                    relevance_score=result.relevance_score,
                    correctness_score=result.correctness_score,
                    clarity_score=result.clarity_score,
                    safety_score=result.safety_score,
                    hallucination_score=result.hallucination_score,
                    duplicate_percentage=result.duplicate_percentage,
                    overall_score=result.overall_score,
                    evaluation_method=outcome.method.value,
                    warnings=list(result.warnings),
                    details=outcome.details(),
                )
            )
            insert_ms = (time.perf_counter() - started) * MS_PER_SECOND
            self._logs.add(
                EvaluationLog(
                    evaluation_result_id=record.id,
                    level="INFO",
                    event="evaluation_saved",
                    message=(
                        f"method={outcome.method.value} overall={result.overall_score} "
                        f"language={code}"
                    ),
                    latency_ms=round(insert_ms, 2),
                )
            )
            gemini = outcome.gemini
            if gemini.used:
                self._logs.add(
                    EvaluationLog(
                        evaluation_result_id=record.id,
                        level="INFO",
                        event="gemini_call",
                        message=f"model={gemini.model}",
                        latency_ms=gemini.latency_ms,
                    )
                )
            elif gemini.error:
                self._logs.add(
                    EvaluationLog(
                        evaluation_result_id=record.id,
                        level="WARNING",
                        event="gemini_fallback",
                        message=gemini.error,
                    )
                )
        invalidate_analytics_cache()
        logger.info(
            "Database insert: result_id=%s duration_ms=%.1f",
            record.id,
            (time.perf_counter() - started) * MS_PER_SECOND,
        )
        return record

    def get_result(self, result_id: int) -> EvaluationResultRecord:
        with self._transaction("reading the result", commit=False):
            record = self._results.get(result_id)
        if record is None:
            raise ResultNotFoundError(f"Evaluation result {result_id} was not found.")
        return record

    def list_results(
        self,
        filters: ResultFilters,
        page: int,
        page_size: int,
        sort_by: SortField,
        order: SortOrder,
    ) -> PaginatedResults:
        with self._transaction("listing results", commit=False):
            records, total = self._results.search(
                filters, sort_by, order, offset=(page - 1) * page_size, limit=page_size
            )
        return PaginatedResults(
            items=[record_to_response(record) for record in records],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if total else 0,
        )

    def export_results(self, filters: ResultFilters, limit: int) -> list[EvaluationResponse]:
        """Return up to ``limit`` matching evaluations (newest first) for reports."""
        with self._transaction("exporting results", commit=False):
            records, _ = self._results.search(
                filters, SortField.CREATED_AT, SortOrder.DESC, offset=0, limit=limit
            )
        return [record_to_response(record) for record in records]

    def delete_result(self, result_id: int) -> None:
        """Delete a result and its response; audit-log rows are kept."""
        record = self.get_result(result_id)
        with self._transaction("deleting the result"):
            self._logs.detach_result(result_id)
            self._responses.delete(record.response_record)
            self._logs.add(
                EvaluationLog(
                    level="INFO",
                    event="result_deleted",
                    message=f"Deleted evaluation result {result_id}",
                )
            )
        invalidate_analytics_cache()
        logger.info("Database delete: result_id=%s", result_id)

    def log_event(
        self,
        event: str,
        message: str,
        level: str = "INFO",
        latency_ms: float | None = None,
    ) -> None:
        with self._transaction("writing the audit log"):
            self._logs.add(
                EvaluationLog(event=event, message=message, level=level, latency_ms=latency_ms)
            )
