"""Pydantic v2 schemas for evaluation input and output."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.language import normalize_language
from app.utils.scoring import SCORE_MAX, SCORE_MIN


class EvaluationInput(BaseModel):
    """A single LLM response to be evaluated."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1)
    language: str = Field(min_length=1)
    response: str = Field(default="")
    reference_answer: str | None = Field(default=None)

    @field_validator("reference_answer")
    @classmethod
    def _empty_reference_to_none(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value

    @property
    def language_code(self) -> str:
        """Normalised ISO-639-1 language code (``und`` when unknown)."""
        return normalize_language(self.language)


class EvaluationResult(BaseModel):
    """Quality scores produced by the evaluation engine.

    ``hallucination_score`` is a *risk* score: 0 means no hallucination
    signals were found and 100 means the response is almost certainly
    hallucinated. All other quality scores are higher-is-better.
    """

    model_config = ConfigDict(frozen=True)

    relevance_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    correctness_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    clarity_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    safety_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    hallucination_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    duplicate_percentage: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    overall_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    warnings: list[str] = Field(default_factory=list)
