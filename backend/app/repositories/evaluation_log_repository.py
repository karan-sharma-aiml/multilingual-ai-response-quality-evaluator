"""Repository for the evaluation audit log."""

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import EvaluationLog
from app.repositories.base import BaseRepository


class EvaluationLogRepository(BaseRepository[EvaluationLog]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, EvaluationLog)

    def list_for_result(self, result_id: int) -> list[EvaluationLog]:
        statement = (
            select(EvaluationLog)
            .where(EvaluationLog.evaluation_result_id == result_id)
            .order_by(EvaluationLog.id)
        )
        return list(self._session.scalars(statement))

    def list_by_event(self, event: str) -> list[EvaluationLog]:
        statement = select(EvaluationLog).where(EvaluationLog.event == event).order_by(EvaluationLog.id)
        return list(self._session.scalars(statement))

    def detach_result(self, result_id: int) -> None:
        """Keep audit rows when their result is deleted."""
        self._session.execute(
            update(EvaluationLog)
            .where(EvaluationLog.evaluation_result_id == result_id)
            .values(evaluation_result_id=None)
        )
