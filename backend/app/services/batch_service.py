"""Batch evaluation of CSV uploads."""

import io
import time

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, BatchFileError
from app.core.logging import get_logger
from app.schemas.api import BatchEvaluationResponse, EvaluateRequest, EvaluationResponse, FailedRecord
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationService

logger = get_logger("batch")

REQUIRED_COLUMNS: tuple[str, ...] = ("question", "response", "language")
OPTIONAL_COLUMNS: tuple[str, ...] = ("reference_answer", "category", "id")
FIRST_DATA_ROW_NUMBER: int = 2  # row 1 is the CSV header
MS_PER_SECOND: float = 1000.0
NON_GEMINI_CHUNK_SIZE: int = 50


class BatchEvaluationService:
    """Evaluates and stores every row of a CSV file."""

    def __init__(
        self,
        evaluation_service: EvaluationService,
        database: DatabaseService,
        settings: Settings | None = None,
    ) -> None:
        self._evaluation = evaluation_service
        self._database = database
        self._settings = settings or get_settings()

    def process_csv(self, content: bytes, use_gemini: bool = False) -> BatchEvaluationResponse:
        started = time.perf_counter()
        frame = self._parse_csv(content)

        scores: list[float] = []
        result_ids: list[int] = []
        failures: list[FailedRecord] = []
        pending: list[tuple[int, str | None, EvaluateRequest]] = []

        for position, (_, row) in enumerate(frame.iterrows()):
            row_number = position + FIRST_DATA_ROW_NUMBER
            record_id = row.get("id") or None
            try:
                pending.append((row_number, record_id, self._row_to_request(row, use_gemini)))
            except ValidationError as exc:
                failures.append(
                    FailedRecord(
                        row_number=row_number, record_id=record_id, error=self._format_validation(exc)
                    )
                )

        chunk_size = self._settings.gemini_concurrency if use_gemini else NON_GEMINI_CHUNK_SIZE
        for offset in range(0, len(pending), chunk_size):
            chunk = pending[offset : offset + chunk_size]
            for _row_number, _record_id, outcome in self._process_chunk(chunk, use_gemini):
                if isinstance(outcome, FailedRecord):
                    failures.append(outcome)
                else:
                    scores.append(outcome.overall_score)
                    result_ids.append(outcome.id)
        failures.sort(key=lambda item: item.row_number)

        summary = BatchEvaluationResponse(
            total_records=len(frame),
            success_count=len(scores),
            failed_count=len(failures),
            average_score=round(sum(scores) / len(scores), 2) if scores else None,
            used_gemini=use_gemini,
            failed_records=failures,
            result_ids=result_ids,
        )
        duration_ms = (time.perf_counter() - started) * MS_PER_SECOND
        logger.info(
            "Batch finished: total=%d success=%d failed=%d avg=%s duration_ms=%.1f",
            summary.total_records, summary.success_count, summary.failed_count,
            summary.average_score, duration_ms,
        )
        self._record_summary(summary, duration_ms)
        return summary

    def _process_chunk(
        self, chunk: list[tuple[int, str | None, EvaluateRequest]], use_gemini: bool
    ) -> list[tuple[int, str | None, EvaluationResponse | FailedRecord]]:
        """Evaluate a chunk (Gemini calls run concurrently) and store each row."""
        samples = [request.to_input() for _, _, request in chunk]
        try:
            outcomes = self._evaluation.evaluate_many(samples, use_gemini)
        except Exception:  # noqa: BLE001 - fall back to isolated per-row processing
            logger.exception("Chunk evaluation failed; retrying row by row")
            outcomes = [None] * len(chunk)

        results: list[tuple[int, str | None, EvaluationResponse | FailedRecord]] = []
        for (row_number, record_id, request), sample, outcome in zip(chunk, samples, outcomes):
            try:
                if outcome is None:
                    saved = self._evaluation.evaluate_and_save(request)
                else:
                    saved = self._evaluation.persist(request, sample, outcome)
                results.append((row_number, record_id, saved))
            except (AppError, SQLAlchemyError) as exc:
                message = exc.message if isinstance(exc, AppError) else "Database error."
                results.append(
                    (row_number, record_id, FailedRecord(row_number=row_number, record_id=record_id, error=message))
                )
            except Exception:  # noqa: BLE001 - one bad row must not abort the batch
                logger.exception("Unexpected error while evaluating row %d", row_number)
                results.append(
                    (
                        row_number,
                        record_id,
                        FailedRecord(
                            row_number=row_number, record_id=record_id, error="Unexpected processing error."
                        ),
                    )
                )
        return results

    def _record_summary(self, summary: BatchEvaluationResponse, duration_ms: float) -> None:
        try:
            self._database.log_event(
                "batch_completed",
                f"total={summary.total_records} success={summary.success_count} "
                f"failed={summary.failed_count} average={summary.average_score}",
                latency_ms=round(duration_ms, 2),
            )
        except AppError:
            logger.warning("Could not store the batch summary log entry.")

    def _parse_csv(self, content: bytes) -> pd.DataFrame:
        if not content.strip():
            raise BatchFileError("The uploaded CSV file is empty.")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BatchFileError("The CSV file must be UTF-8 encoded.") from exc
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise BatchFileError(f"The CSV file could not be parsed: {exc}") from exc

        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise BatchFileError(
                f"Missing required column(s): {', '.join(missing)}.",
                details={"required": list(REQUIRED_COLUMNS), "optional": list(OPTIONAL_COLUMNS)},
            )
        if frame.empty:
            raise BatchFileError("The CSV file contains no data rows.")
        if len(frame) > self._settings.max_batch_rows:
            raise BatchFileError(
                f"The CSV file has {len(frame)} rows; the maximum is {self._settings.max_batch_rows}."
            )
        return frame

    @staticmethod
    def _row_to_request(row: pd.Series, use_gemini: bool) -> EvaluateRequest:
        return EvaluateRequest(
            question=row.get("question", ""),
            response=row.get("response", ""),
            language=row.get("language", ""),
            reference_answer=row.get("reference_answer") or None,
            category=row.get("category") or None,
            use_gemini=use_gemini,
        )

    @staticmethod
    def _format_validation(error: ValidationError) -> str:
        parts = []
        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"]) or "row"
            message = str(item["msg"]).removeprefix("Value error, ")
            parts.append(f"{field}: {message}")
        return "; ".join(parts)
