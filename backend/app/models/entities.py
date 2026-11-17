"""ORM entities of the evaluation store: languages, responses, results, logs."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

LANGUAGE_CODE_LENGTH: int = 8
LANGUAGE_NAME_LENGTH: int = 64
CATEGORY_LENGTH: int = 64
METHOD_LENGTH: int = 32
LOG_LEVEL_LENGTH: int = 16
LOG_EVENT_LENGTH: int = 64


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Language(Base):
    """Supported language (seeded on start-up)."""

    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(LANGUAGE_CODE_LENGTH), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(LANGUAGE_NAME_LENGTH))


class ResponseRecord(Base):
    """An LLM response together with the question that produced it."""

    __tablename__ = "responses"
    __table_args__ = (Index("ix_responses_language_category", "language_id", "category"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    language_id: Mapped[int] = mapped_column(ForeignKey("languages.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(CATEGORY_LENGTH), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    language: Mapped[Language] = relationship(lazy="joined")
    results: Mapped[list["EvaluationResultRecord"]] = relationship(
        back_populates="response_record",
        cascade="all, delete-orphan",
    )


class EvaluationResultRecord(Base):
    """Scores produced for one response."""

    __tablename__ = "evaluation_results"
    __table_args__ = (
        Index("ix_evaluation_results_created_overall", "created_at", "overall_score"),
        Index("ix_evaluation_results_method_created", "evaluation_method", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), index=True)
    relevance_score: Mapped[float] = mapped_column(Float)
    correctness_score: Mapped[float] = mapped_column(Float)
    clarity_score: Mapped[float] = mapped_column(Float)
    safety_score: Mapped[float] = mapped_column(Float)
    hallucination_score: Mapped[float] = mapped_column(Float)
    duplicate_percentage: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float, index=True)
    evaluation_method: Mapped[str] = mapped_column(String(METHOD_LENGTH), index=True)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, index=True
    )

    response_record: Mapped[ResponseRecord] = relationship(
        back_populates="results", lazy="joined"
    )


class EvaluationLog(Base):
    """Audit trail of evaluation-related events."""

    __tablename__ = "evaluation_logs"
    __table_args__ = (Index("ix_evaluation_logs_event_created", "event", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_results.id"), nullable=True, index=True
    )
    level: Mapped[str] = mapped_column(String(LOG_LEVEL_LENGTH), default="INFO")
    event: Mapped[str] = mapped_column(String(LOG_EVENT_LENGTH), index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
