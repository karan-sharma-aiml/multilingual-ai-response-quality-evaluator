"""Generic repository providing the common CRUD operations."""

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):  # noqa: UP046 - TypeVar bound keeps 3.11 tooling happy
    """CRUD operations for a single ORM model."""

    def __init__(self, session: Session, model: type[ModelT]) -> None:
        self._session = session
        self._model = model

    @property
    def session(self) -> Session:
        return self._session

    def add(self, entity: ModelT) -> ModelT:
        """Add and flush ``entity`` so that its primary key is populated."""
        self._session.add(entity)
        self._session.flush()
        return entity

    def get(self, entity_id: int) -> ModelT | None:
        return self._session.get(self._model, entity_id)

    def list(self, offset: int = 0, limit: int = 100) -> list[ModelT]:
        statement = select(self._model).order_by(self._model.id).offset(offset).limit(limit)
        return list(self._session.scalars(statement).unique())

    def count(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(self._model)) or 0)

    def delete(self, entity: ModelT) -> None:
        self._session.delete(entity)
        self._session.flush()
