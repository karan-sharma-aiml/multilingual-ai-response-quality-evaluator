"""Repository for evaluation results with filtering, sorting and pagination."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, contains_eager

from app.models import EvaluationResultRecord, Language, ResponseRecord
from app.repositories.base import BaseRepository
from app.schemas.api import SortField, SortOrder


@dataclass(frozen=True)
class ResultFilters:
    """Optional filters for result listings."""

    language_code: str | None = None
    evaluation_method: str | None = None
    category: str | None = None
    min_overall_score: float | None = None
    max_overall_score: float | None = None
    below_overall_score: float | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None  # exclusive upper bound


class EvaluationResultRepository(BaseRepository[EvaluationResultRecord]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, EvaluationResultRecord)

    def search(
        self,
        filters: ResultFilters,
        sort_by: SortField = SortField.CREATED_AT,
        order: SortOrder = SortOrder.DESC,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[EvaluationResultRecord], int]:
        """Return one page of results and the total number of matches."""
        base = self._apply_filters(
            select(EvaluationResultRecord)
            .join(EvaluationResultRecord.response_record)
            .join(ResponseRecord.language),
            filters,
        )
        total = int(
            self._session.scalar(
                self._apply_filters(
                    select(func.count(EvaluationResultRecord.id))
                    .select_from(EvaluationResultRecord)
                    .join(EvaluationResultRecord.response_record)
                    .join(ResponseRecord.language),
                    filters,
                )
            )
            or 0
        )

        column = getattr(EvaluationResultRecord, sort_by.value)
        ordering = column.asc() if order is SortOrder.ASC else column.desc()
        statement = (
            base.options(
                contains_eager(EvaluationResultRecord.response_record).contains_eager(
                    ResponseRecord.language
                )
            )
            .order_by(ordering, EvaluationResultRecord.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement).unique()), total

    @staticmethod
    def _apply_filters(statement: Select, filters: ResultFilters) -> Select:
        if filters.language_code:
            statement = statement.where(Language.code == filters.language_code)
        if filters.evaluation_method:
            statement = statement.where(
                EvaluationResultRecord.evaluation_method == filters.evaluation_method
            )
        if filters.category:
            statement = statement.where(
                func.lower(ResponseRecord.category) == filters.category.lower()
            )
        if filters.min_overall_score is not None:
            statement = statement.where(
                EvaluationResultRecord.overall_score >= filters.min_overall_score
            )
        if filters.max_overall_score is not None:
            statement = statement.where(
                EvaluationResultRecord.overall_score <= filters.max_overall_score
            )
        if filters.below_overall_score is not None:
            statement = statement.where(
                EvaluationResultRecord.overall_score < filters.below_overall_score
            )
        if filters.created_from is not None:
            statement = statement.where(EvaluationResultRecord.created_at >= filters.created_from)
        if filters.created_to is not None:
            statement = statement.where(EvaluationResultRecord.created_at < filters.created_to)
        return statement
