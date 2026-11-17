"""Response schemas of the analytics API."""

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.api import EvaluationResponse


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TrendPeriod(str, Enum):
    ALL = "all"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MetricAverages(BaseModel):
    average_overall_score: float | None = None
    average_relevance: float | None = None
    average_correctness: float | None = None
    average_clarity: float | None = None
    average_safety: float | None = None
    average_hallucination: float | None = None
    average_duplicate_percentage: float | None = None


class OverviewStats(MetricAverages):
    total_evaluations: int = 0
    low_quality_count: int = 0
    low_quality_threshold: float = 50.0
    hybrid_evaluations: int = 0
    generated_at: datetime = Field(default_factory=_now)


class LanguageStats(MetricAverages):
    language: str
    language_name: str
    total_evaluations: int


class LanguageHighlight(BaseModel):
    language: str
    language_name: str
    average_score: float


class LanguageAnalytics(BaseModel):
    languages: list[LanguageStats]
    best_language: LanguageHighlight | None = None
    worst_language: LanguageHighlight | None = None


class CategoryStats(MetricAverages):
    category: str
    total_evaluations: int
    low_quality_count: int = 0


class CategoryAnalytics(BaseModel):
    categories: list[CategoryStats]
    best_category: str | None = None
    worst_category: str | None = None


class HallucinationTypeStat(BaseModel):
    type: str
    description: str
    frequency: int = Field(description="Number of evaluations in which the type was found")
    percentage: float = Field(description="Share of all detected hallucination types (sums to 100)")
    prevalence_percentage: float = Field(description="Share of all evaluations affected by the type")


class HallucinationAnalytics(BaseModel):
    total_evaluations: int
    evaluations_with_signals: int
    signal_rate_percentage: float
    average_hallucination_score: float | None
    types: list[HallucinationTypeStat]


class LowQualityResponse(BaseModel):
    threshold: float
    total: int
    page: int
    page_size: int
    total_pages: int
    items: list[EvaluationResponse]


class TrendPoint(BaseModel):
    period: str = Field(description="Day (YYYY-MM-DD), ISO week start (Monday) or month (YYYY-MM)")
    start_date: date
    evaluations: int
    average_overall_score: float
    average_relevance: float
    average_correctness: float
    average_clarity: float
    average_safety: float
    average_hallucination: float


class TrendsResponse(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    daily: list[TrendPoint] | None = None
    weekly: list[TrendPoint] | None = None
    monthly: list[TrendPoint] | None = None
