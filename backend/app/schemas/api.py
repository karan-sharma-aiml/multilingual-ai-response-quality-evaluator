"""Request and response schemas of the REST API (Pydantic v2)."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.evaluation import EvaluationInput
from app.utils.language import SUPPORTED_LANGUAGES, normalize_language
from app.utils.sanitize import sanitize_text
from app.utils.scoring import SCORE_MAX, SCORE_MIN

MAX_QUESTION_LENGTH: int = 2_000
MAX_RESPONSE_LENGTH: int = 20_000
MAX_REFERENCE_LENGTH: int = 20_000
MAX_CATEGORY_LENGTH: int = 64
MAX_PAGE_SIZE: int = 100
DEFAULT_PAGE_SIZE: int = 20


class EvaluationMethod(str, Enum):
    """How an evaluation was produced."""

    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


class SortField(str, Enum):
    ID = "id"
    CREATED_AT = "created_at"
    OVERALL_SCORE = "overall_score"
    RELEVANCE_SCORE = "relevance_score"
    CORRECTNESS_SCORE = "correctness_score"
    CLARITY_SCORE = "clarity_score"
    SAFETY_SCORE = "safety_score"
    HALLUCINATION_SCORE = "hallucination_score"
    DUPLICATE_PERCENTAGE = "duplicate_percentage"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


def _sanitize_value(value: object) -> object:
    return sanitize_text(value) if isinstance(value, str) else value


def validate_language(value: str) -> str:
    """Normalise a language name/code and reject unsupported languages."""
    code = normalize_language(value)
    if code not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unsupported language '{value}'. Supported codes: {supported} "
            "(or English, Hindi, French, Spanish, German)."
        )
    return code


class EvaluateRequest(BaseModel):
    """Body of ``POST /evaluate`` (also used for every batch row)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    response: str = Field(min_length=1, max_length=MAX_RESPONSE_LENGTH)
    language: str = Field(min_length=1)
    reference_answer: str | None = Field(default=None, max_length=MAX_REFERENCE_LENGTH)
    category: str | None = Field(default=None, max_length=MAX_CATEGORY_LENGTH)
    use_gemini: bool = Field(default=True)

    @field_validator("*", mode="before")
    @classmethod
    def _sanitize(cls, value: object) -> object:
        return _sanitize_value(value)

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return validate_language(value)

    @field_validator("reference_answer", "category")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value or None

    def to_input(self) -> EvaluationInput:
        return EvaluationInput(
            question=self.question,
            language=self.language,
            response=self.response,
            reference_answer=self.reference_answer,
        )


class CompareRequest(BaseModel):
    """Body of ``POST /compare``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    language: str = Field(min_length=1)
    response_a: str = Field(min_length=1, max_length=MAX_RESPONSE_LENGTH)
    response_b: str = Field(min_length=1, max_length=MAX_RESPONSE_LENGTH)
    reference_answer: str | None = Field(default=None, max_length=MAX_REFERENCE_LENGTH)
    category: str | None = Field(default=None, max_length=MAX_CATEGORY_LENGTH)
    use_gemini: bool = Field(default=True)

    @field_validator("*", mode="before")
    @classmethod
    def _sanitize(cls, value: object) -> object:
        return _sanitize_value(value)

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return validate_language(value)

    @field_validator("reference_answer", "category")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value or None

    def to_request(self, response: str) -> EvaluateRequest:
        return EvaluateRequest(
            question=self.question,
            response=response,
            language=self.language,
            reference_answer=self.reference_answer,
            category=self.category,
            use_gemini=self.use_gemini,
        )


class GeminiInfo(BaseModel):
    """What happened with the Gemini part of an evaluation."""

    used: bool = False
    model: str | None = None
    latency_ms: float | None = None
    reasoning: str | None = None
    scores: dict[str, float] | None = None
    error: str | None = None


class EvaluationResponse(BaseModel):
    """A stored evaluation (flattened view of response + result rows)."""

    id: int
    response_id: int
    language: str
    language_name: str
    category: str | None
    question: str
    response: str
    reference_answer: str | None
    relevance_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    correctness_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    clarity_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    safety_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    hallucination_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    duplicate_percentage: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    overall_score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    evaluation_method: EvaluationMethod
    warnings: list[str] = Field(default_factory=list)
    gemini: GeminiInfo = Field(default_factory=GeminiInfo)
    created_at: datetime


class PaginatedResults(BaseModel):
    items: list[EvaluationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DeleteResponse(BaseModel):
    id: int
    deleted: bool = True
    message: str


class FailedRecord(BaseModel):
    row_number: int
    record_id: str | None = None
    error: str


class BatchEvaluationResponse(BaseModel):
    total_records: int
    success_count: int
    failed_count: int
    average_score: float | None
    used_gemini: bool
    failed_records: list[FailedRecord] = Field(default_factory=list)
    result_ids: list[int] = Field(default_factory=list)


class MetricComparison(BaseModel):
    metric: str
    response_a: float
    response_b: float
    difference: float = Field(description="response_a minus response_b")
    winner: Literal["A", "B", "tie"]


class CompareResponse(BaseModel):
    better_response: Literal["A", "B", "tie"]
    overall_difference: float
    reason: str
    comparison: list[MetricComparison]
    response_a: EvaluationResponse
    response_b: EvaluationResponse


class ComponentHealth(BaseModel):
    status: Literal["ok", "error", "not_configured"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    database: ComponentHealth
    gemini: ComponentHealth


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
