"""ORM model storing one evaluated response (no CRUD logic in Part 1)."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base

LANGUAGE_MAX_LENGTH: int = 32
CATEGORY_MAX_LENGTH: int = 64


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationRecord(Base):
    """Persisted evaluation of a single LLM response."""

    __tablename__ = "evaluation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    language: Mapped[str] = mapped_column(String(LANGUAGE_MAX_LENGTH), index=True)
    category: Mapped[str | None] = mapped_column(String(CATEGORY_MAX_LENGTH), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float] = mapped_column(Float)
    correctness_score: Mapped[float] = mapped_column(Float)
    clarity_score: Mapped[float] = mapped_column(Float)
    safety_score: Mapped[float] = mapped_column(Float)
    hallucination_score: Mapped[float] = mapped_column(Float)
    duplicate_percentage: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float)
    warnings: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
