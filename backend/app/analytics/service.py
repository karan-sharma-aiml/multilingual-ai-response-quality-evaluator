"""Analytics queries: SQL aggregates plus light Python post-processing."""

import math
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
from sqlalchemy import case, func, literal_column, select
from sqlalchemy.orm import Session

from app.analytics.hallucination_types import (
    HALLUCINATION_TYPE_DESCRIPTIONS,
    extract_hallucination_types,
)
from app.core.cache import TTLCache
from app.core.config import Settings, get_settings
from app.evaluator import HallucinationDetector
from app.models import EvaluationResultRecord as Result
from app.models import Language, ResponseRecord
from app.repositories import EvaluationResultRepository, ResultFilters
from app.schemas.analytics import (
    CategoryAnalytics,
    CategoryStats,
    HallucinationAnalytics,
    HallucinationTypeStat,
    LanguageAnalytics,
    LanguageHighlight,
    LanguageStats,
    LowQualityResponse,
    OverviewStats,
    TrendPeriod,
    TrendPoint,
    TrendsResponse,
)
from app.schemas.api import SortField, SortOrder
from app.schemas.evaluation import EvaluationInput
from app.services.mappers import record_to_response
from app.utils.language import LANGUAGE_NAMES

UNCATEGORIZED: str = "Uncategorized"
PRECISION: int = 2
STREAM_BATCH_SIZE: int = 500


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), PRECISION)


class AnalyticsService:
    """Aggregated statistics over stored evaluations."""

    def __init__(
        self,
        session: Session,
        cache: TTLCache | None = None,
        settings: Settings | None = None,
        detector: HallucinationDetector | None = None,
    ) -> None:
        self._session = session
        self._cache = cache
        self._settings = settings or get_settings()
        self._detector = detector or HallucinationDetector()
        self._results = EvaluationResultRepository(session)

    def _cached(self, key: tuple, factory):
        return self._cache.get_or_set(key, factory) if self._cache is not None else factory()

    @property
    def _threshold(self) -> float:
        return self._settings.low_quality_threshold

    # ------------------------------------------------------------ overview
    def overview(self) -> OverviewStats:
        return self._cached(("overview", self._threshold), self._compute_overview)

    def _compute_overview(self) -> OverviewStats:
        row = self._session.execute(
            select(
                func.count(Result.id),
                func.avg(Result.overall_score),
                func.avg(Result.relevance_score),
                func.avg(Result.correctness_score),
                func.avg(Result.clarity_score),
                func.avg(Result.safety_score),
                func.avg(Result.hallucination_score),
                func.avg(Result.duplicate_percentage),
                func.coalesce(func.sum(case((Result.overall_score < self._threshold, 1), else_=0)), 0),
                func.coalesce(func.sum(case((Result.evaluation_method == "hybrid", 1), else_=0)), 0),
            )
        ).one()
        return OverviewStats(
            total_evaluations=int(row[0]),
            average_overall_score=_round(row[1]),
            average_relevance=_round(row[2]),
            average_correctness=_round(row[3]),
            average_clarity=_round(row[4]),
            average_safety=_round(row[5]),
            average_hallucination=_round(row[6]),
            average_duplicate_percentage=_round(row[7]),
            low_quality_count=int(row[8]),
            low_quality_threshold=self._threshold,
            hybrid_evaluations=int(row[9]),
        )

    # ----------------------------------------------------------- languages
    def languages(self) -> LanguageAnalytics:
        return self._cached(("languages",), self._compute_languages)

    def _compute_languages(self) -> LanguageAnalytics:
        average = func.avg(Result.overall_score)
        rows = self._session.execute(
            select(
                Language.code,
                Language.name,
                func.count(Result.id),
                average,
                func.avg(Result.relevance_score),
                func.avg(Result.correctness_score),
                func.avg(Result.clarity_score),
                func.avg(Result.safety_score),
                func.avg(Result.hallucination_score),
                func.avg(Result.duplicate_percentage),
            )
            .select_from(Result)
            .join(ResponseRecord, Result.response_id == ResponseRecord.id)
            .join(Language, ResponseRecord.language_id == Language.id)
            .group_by(Language.code, Language.name)
            .order_by(average.desc(), Language.code)
        ).all()
        stats = [
            LanguageStats(
                language=row[0],
                language_name=row[1] or LANGUAGE_NAMES.get(row[0], row[0]),
                total_evaluations=int(row[2]),
                average_overall_score=_round(row[3]),
                average_relevance=_round(row[4]),
                average_correctness=_round(row[5]),
                average_clarity=_round(row[6]),
                average_safety=_round(row[7]),
                average_hallucination=_round(row[8]),
                average_duplicate_percentage=_round(row[9]),
            )
            for row in rows
        ]
        if not stats:
            return LanguageAnalytics(languages=[])
        return LanguageAnalytics(
            languages=stats,
            best_language=self._highlight(stats[0]),
            worst_language=self._highlight(stats[-1]),
        )

    @staticmethod
    def _highlight(stat: LanguageStats) -> LanguageHighlight:
        return LanguageHighlight(
            language=stat.language,
            language_name=stat.language_name,
            average_score=stat.average_overall_score or 0.0,
        )

    # ---------------------------------------------------------- categories
    def categories(self) -> CategoryAnalytics:
        return self._cached(("categories", self._threshold), self._compute_categories)

    def _compute_categories(self) -> CategoryAnalytics:
        category = func.coalesce(ResponseRecord.category, literal_column(f"'{UNCATEGORIZED}'"))
        average = func.avg(Result.overall_score)
        rows = self._session.execute(
            select(
                category,
                func.count(Result.id),
                average,
                func.avg(Result.relevance_score),
                func.avg(Result.correctness_score),
                func.avg(Result.clarity_score),
                func.avg(Result.safety_score),
                func.avg(Result.hallucination_score),
                func.avg(Result.duplicate_percentage),
                func.coalesce(func.sum(case((Result.overall_score < self._threshold, 1), else_=0)), 0),
            )
            .select_from(Result)
            .join(ResponseRecord, Result.response_id == ResponseRecord.id)
            .group_by(category)
            .order_by(average.desc(), category)
        ).all()
        stats = [
            CategoryStats(
                category=row[0],
                total_evaluations=int(row[1]),
                average_overall_score=_round(row[2]),
                average_relevance=_round(row[3]),
                average_correctness=_round(row[4]),
                average_clarity=_round(row[5]),
                average_safety=_round(row[6]),
                average_hallucination=_round(row[7]),
                average_duplicate_percentage=_round(row[8]),
                low_quality_count=int(row[9]),
            )
            for row in rows
        ]
        return CategoryAnalytics(
            categories=stats,
            best_category=stats[0].category if stats else None,
            worst_category=stats[-1].category if stats else None,
        )

    # ------------------------------------------------------- hallucinations
    def hallucinations(self) -> HallucinationAnalytics:
        return self._cached(("hallucinations",), self._compute_hallucinations)

    def _compute_hallucinations(self) -> HallucinationAnalytics:
        frequency: Counter[str] = Counter()
        total = 0
        with_signals = 0
        statement = select(Result.id, Result.details).execution_options(yield_per=STREAM_BATCH_SIZE)
        for result_id, details in self._session.execute(statement):
            total += 1
            types = (details or {}).get("hallucination_types")
            if types is None:  # evaluation stored before types were recorded
                types = self._recompute_types(result_id)
            if types:
                with_signals += 1
                frequency.update(types.keys())

        total_signals = sum(frequency.values())
        stats = [
            HallucinationTypeStat(
                type=name,
                description=HALLUCINATION_TYPE_DESCRIPTIONS.get(name, name),
                frequency=count,
                percentage=round(count / total_signals * 100, PRECISION),
                prevalence_percentage=round(count / total * 100, PRECISION),
            )
            for name, count in frequency.most_common()
        ]
        average = self._session.scalar(select(func.avg(Result.hallucination_score)))
        return HallucinationAnalytics(
            total_evaluations=total,
            evaluations_with_signals=with_signals,
            signal_rate_percentage=round(with_signals / total * 100, PRECISION) if total else 0.0,
            average_hallucination_score=_round(average),
            types=stats,
        )

    def _recompute_types(self, result_id: int) -> dict[str, int]:
        record = self._results.get(result_id)
        if record is None:
            return {}
        response = record.response_record
        sample = EvaluationInput(
            question=response.question,
            language=response.language.code,
            response=response.response,
            reference_answer=response.reference_answer,
        )
        return extract_hallucination_types(self._detector.analyze(sample))

    # ---------------------------------------------------------- low quality
    def low_quality(
        self,
        page: int,
        page_size: int,
        threshold: float | None = None,
        language_code: str | None = None,
        category: str | None = None,
    ) -> LowQualityResponse:
        limit_value = self._threshold if threshold is None else threshold
        filters = ResultFilters(
            below_overall_score=limit_value, language_code=language_code, category=category
        )
        records, total = self._results.search(
            filters,
            SortField.OVERALL_SCORE,
            SortOrder.ASC,
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        return LowQualityResponse(
            threshold=limit_value,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if total else 0,
            items=[record_to_response(record) for record in records],
        )

    # --------------------------------------------------------------- trends
    def trends(
        self,
        period: TrendPeriod = TrendPeriod.ALL,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> TrendsResponse:
        key = ("trends", period.value, start_date, end_date)
        return self._cached(key, lambda: self._compute_trends(period, start_date, end_date))

    def _compute_trends(
        self, period: TrendPeriod, start_date: date | None, end_date: date | None
    ) -> TrendsResponse:
        statement = select(
            Result.created_at,
            Result.overall_score,
            Result.relevance_score,
            Result.correctness_score,
            Result.clarity_score,
            Result.safety_score,
            Result.hallucination_score,
        )
        if start_date:
            statement = statement.where(
                Result.created_at >= datetime.combine(start_date, time.min, tzinfo=timezone.utc)
            )
        if end_date:
            statement = statement.where(
                Result.created_at
                < datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
            )
        frame = pd.DataFrame(
            self._session.execute(statement).all(),
            columns=[
                "created_at", "overall", "relevance", "correctness", "clarity", "safety", "hallucination",
            ],
        )
        response = TrendsResponse(start_date=start_date, end_date=end_date)
        wanted = {
            TrendPeriod.DAILY: period in (TrendPeriod.ALL, TrendPeriod.DAILY),
            TrendPeriod.WEEKLY: period in (TrendPeriod.ALL, TrendPeriod.WEEKLY),
            TrendPeriod.MONTHLY: period in (TrendPeriod.ALL, TrendPeriod.MONTHLY),
        }
        if frame.empty:
            for name, enabled in wanted.items():
                if enabled:
                    setattr(response, name.value, [])
            return response

        days = frame["created_at"].map(self._to_utc_date)
        keys = {
            TrendPeriod.DAILY: days,
            TrendPeriod.WEEKLY: days.map(lambda day: day - timedelta(days=day.weekday())),
            TrendPeriod.MONTHLY: days.map(lambda day: day.replace(day=1)),
        }
        for name, enabled in wanted.items():
            if enabled:
                setattr(response, name.value, self._group(frame, keys[name], name))
        return response

    @staticmethod
    def _to_utc_date(value: datetime) -> date:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()

    @staticmethod
    def _group(frame: pd.DataFrame, keys: pd.Series, period: TrendPeriod) -> list[TrendPoint]:
        grouped = frame.drop(columns="created_at").groupby(keys.values)
        counts = grouped.size()
        means = grouped.mean()
        points: list[TrendPoint] = []
        for start in sorted(counts.index):
            if period is TrendPeriod.MONTHLY:
                label = start.strftime("%Y-%m")
            else:
                label = start.isoformat()
            row = means.loc[start]
            points.append(
                TrendPoint(
                    period=label,
                    start_date=start,
                    evaluations=int(counts.loc[start]),
                    average_overall_score=round(float(row["overall"]), PRECISION),
                    average_relevance=round(float(row["relevance"]), PRECISION),
                    average_correctness=round(float(row["correctness"]), PRECISION),
                    average_clarity=round(float(row["clarity"]), PRECISION),
                    average_safety=round(float(row["safety"]), PRECISION),
                    average_hallucination=round(float(row["hallucination"]), PRECISION),
                )
            )
        return points
