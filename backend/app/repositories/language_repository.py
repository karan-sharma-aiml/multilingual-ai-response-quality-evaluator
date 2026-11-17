"""Repository for supported languages."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Language
from app.repositories.base import BaseRepository


class LanguageRepository(BaseRepository[Language]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Language)

    def get_by_code(self, code: str) -> Language | None:
        return self._session.scalar(select(Language).where(Language.code == code))

    def get_or_create(self, code: str, name: str) -> Language:
        existing = self.get_by_code(code)
        if existing is not None:
            return existing
        return self.add(Language(code=code, name=name))
