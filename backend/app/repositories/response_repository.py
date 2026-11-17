"""Repository for stored responses."""

from sqlalchemy.orm import Session

from app.models import ResponseRecord
from app.repositories.base import BaseRepository


class ResponseRepository(BaseRepository[ResponseRecord]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ResponseRecord)
