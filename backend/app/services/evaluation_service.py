"""Orchestrates rule-based evaluation, Gemini evaluation, merging and storage."""

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.analytics.hallucination_types import extract_hallucination_types
from app.core.config import Settings, get_settings
from app.core.exceptions import GeminiError
from app.core.logging import get_logger
from app.evaluator import (
    HallucinationDetector,
    OverallScoreEngine,
    ResponseEvaluator,
    RuleEngine,
    ScoreComponents,
)
from app.schemas.api import (
    CompareRequest,
    CompareResponse,
    EvaluateRequest,
    EvaluationMethod,
    EvaluationResponse,
    GeminiInfo,
    MetricComparison,
)
from app.schemas.evaluation import EvaluationInput, EvaluationResult
from app.services.database_service import DatabaseService
from app.services.gemini_service import GeminiEvaluation
from app.services.mappers import record_to_response
from app.utils.scoring import clamp_score

logger = get_logger("evaluation")

TIE_MARGIN: float = 1.0
METRIC_TIE_TOLERANCE: float = 0.01
REASON_HIGHLIGHTS: int = 2
MS_PER_SECOND: float = 1000.0
GEMINI_METRICS: tuple[str, ...] = ("relevance", "correctness", "clarity", "safety")

# metric attribute -> (label, higher_is_better)
COMPARISON_METRICS: dict[str, tuple[str, bool]] = {
    "relevance_score": ("relevance", True),
    "correctness_score": ("correctness", True),
    "clarity_score": ("clarity", True),
    "safety_score": ("safety", True),
    "hallucination_score": ("hallucination risk", False),
    "duplicate_percentage": ("duplicate percentage", False),
    "overall_score": ("overall score", True),
}


class GeminiEvaluator(Protocol):
    """Interface of the Gemini collaborator (allows easy mocking)."""

    def evaluate(self, sample: EvaluationInput) -> GeminiEvaluation: ...


@dataclass(frozen=True)
class EvaluationOutcome:
    """Merged evaluation plus provenance information."""

    result: EvaluationResult
    method: EvaluationMethod
    gemini: GeminiInfo
    rule_scores: dict[str, float]
    hallucination_types: dict[str, int] = field(default_factory=dict)

    def details(self) -> dict[str, Any]:
        """JSON-serialisable provenance stored with the result."""
        return {
            "gemini": self.gemini.model_dump(),
            "rule_scores": self.rule_scores,
            "hallucination_types": self.hallucination_types,
        }


@dataclass(frozen=True)
class _RulePhase:
    result: EvaluationResult
    rule_scores: dict[str, float]
    hallucination_types: dict[str, int]


class EvaluationService:
    """Runs the full evaluation workflow."""

    def __init__(
        self,
        rule_evaluator: ResponseEvaluator,
        gemini: GeminiEvaluator,
        database: DatabaseService,
        settings: Settings | None = None,
        rule_engine: RuleEngine | None = None,
        overall_engine: OverallScoreEngine | None = None,
        hallucination_detector: HallucinationDetector | None = None,
    ) -> None:
        self._rules = rule_evaluator
        self._gemini = gemini
        self._database = database
        self._settings = settings or get_settings()
        self._rule_engine = rule_engine or RuleEngine()
        self._overall = overall_engine or OverallScoreEngine()
        self._hallucination = hallucination_detector or HallucinationDetector()

    # ------------------------------------------------------------ evaluate
    def evaluate(self, sample: EvaluationInput, use_gemini: bool = True) -> EvaluationOutcome:
        """Rule engine -> Gemini (optional, with fallback) -> merged scores."""
        rule_phase = self._rule_phase(sample)
        gemini = self._gemini_phase(sample) if use_gemini else None
        return self._assemble(sample, rule_phase, gemini)

    def evaluate_many(
        self, samples: Sequence[EvaluationInput], use_gemini: bool = True
    ) -> list[EvaluationOutcome]:
        """Evaluate several samples; Gemini calls run concurrently (I/O bound)."""
        rule_phases = [self._rule_phase(sample) for sample in samples]
        if not use_gemini:
            return [self._assemble(s, p, None) for s, p in zip(samples, rule_phases)]
        workers = max(1, min(self._settings.gemini_concurrency, len(samples)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gemini") as pool:
            gemini_results = list(pool.map(self._gemini_phase, samples))
        return [
            self._assemble(sample, phase, gemini)
            for sample, phase, gemini in zip(samples, rule_phases, gemini_results)
        ]

    def _rule_phase(self, sample: EvaluationInput) -> _RulePhase:
        result = self._rules.evaluate(sample)
        rule_scores = {
            "relevance": result.relevance_score,
            "correctness": result.correctness_score,
            "clarity": result.clarity_score,
            "safety": result.safety_score,
            "hallucination": result.hallucination_score,
            "duplicate": result.duplicate_percentage,
            "overall": result.overall_score,
        }
        types = extract_hallucination_types(self._hallucination.analyze(sample))
        return _RulePhase(result, rule_scores, types)

    def _gemini_phase(self, sample: EvaluationInput) -> GeminiEvaluation | GeminiError:
        """Call Gemini; failures are returned (not raised) so callers can fall back."""
        try:
            return self._gemini.evaluate(sample)
        except GeminiError as error:
            logger.warning("Gemini unavailable, using rule-based scores: %s", error.message)
            return error

    def _assemble(
        self,
        sample: EvaluationInput,
        phase: _RulePhase,
        gemini: GeminiEvaluation | GeminiError | None,
    ) -> EvaluationOutcome:
        if gemini is None or isinstance(gemini, GeminiError):
            return EvaluationOutcome(
                result=phase.result,
                method=EvaluationMethod.RULE_BASED,
                gemini=GeminiInfo(used=False, error=gemini.message if gemini else None),
                rule_scores=phase.rule_scores,
                hallucination_types=phase.hallucination_types,
            )
        merged = self._merge(sample, phase.result, gemini)
        info = GeminiInfo(
            used=True,
            model=gemini.model,
            latency_ms=gemini.latency_ms,
            reasoning=gemini.reasoning or None,
            scores=gemini.scores(),
        )
        return EvaluationOutcome(
            result=merged,
            method=EvaluationMethod.HYBRID,
            gemini=info,
            rule_scores=phase.rule_scores,
            hallucination_types=phase.hallucination_types,
        )

    def _merge(
        self,
        sample: EvaluationInput,
        rule: EvaluationResult,
        gemini: GeminiEvaluation,
    ) -> EvaluationResult:
        """Weighted merge; safety keeps the stricter (lower) of both sources."""
        weight = self._settings.gemini_weight

        def blend(rule_score: float, gemini_score: float) -> float:
            return clamp_score(weight * gemini_score + (1.0 - weight) * rule_score)

        relevance = blend(rule.relevance_score, gemini.relevance)
        correctness = blend(rule.correctness_score, gemini.correctness)
        clarity = blend(rule.clarity_score, gemini.clarity)
        safety = clamp_score(min(rule.safety_score, gemini.safety))

        warnings = self._rule_engine.evaluate(sample)
        overall = self._overall.calculate(
            ScoreComponents(
                relevance=relevance,
                correctness=correctness,
                clarity=clarity,
                safety=safety,
                hallucination=rule.hallucination_score,
                duplicate_percentage=rule.duplicate_percentage,
            ),
            warnings,
        )
        return EvaluationResult(
            relevance_score=relevance,
            correctness_score=correctness,
            clarity_score=clarity,
            safety_score=safety,
            hallucination_score=rule.hallucination_score,
            duplicate_percentage=rule.duplicate_percentage,
            overall_score=overall,
            warnings=list(rule.warnings),
        )

    # ------------------------------------------------------------- storage
    def evaluate_and_save(self, request: EvaluateRequest) -> EvaluationResponse:
        """Full workflow of ``POST /evaluate``."""
        sample = request.to_input()
        outcome = self.evaluate(sample, request.use_gemini)
        return self.persist(request, sample, outcome)

    def persist(
        self, request: EvaluateRequest, sample: EvaluationInput, outcome: EvaluationOutcome
    ) -> EvaluationResponse:
        """Store an already computed outcome and return the API representation."""
        started = time.perf_counter()
        record = self._database.save_evaluation(sample, outcome, request.category)
        logger.info(
            "Evaluation stored: id=%s method=%s overall=%.2f save_ms=%.1f",
            record.id,
            outcome.method.value,
            outcome.result.overall_score,
            (time.perf_counter() - started) * MS_PER_SECOND,
        )
        return record_to_response(record)

    # ------------------------------------------------------------- compare
    def compare(self, request: CompareRequest) -> CompareResponse:
        """Evaluate two responses to the same question and pick the better one."""
        requests = [request.to_request(request.response_a), request.to_request(request.response_b)]
        samples = [item.to_input() for item in requests]
        outcomes = self.evaluate_many(samples, request.use_gemini)
        first, second = (
            self.persist(item, sample, outcome)
            for item, sample, outcome in zip(requests, samples, outcomes)
        )

        comparison = [
            self._compare_metric(attribute, first, second) for attribute in COMPARISON_METRICS
        ]
        difference = round(first.overall_score - second.overall_score, 2)
        better: Literal["A", "B", "tie"]
        if abs(difference) < TIE_MARGIN:
            better = "tie"
        else:
            better = "A" if difference > 0 else "B"
        return CompareResponse(
            better_response=better,
            overall_difference=difference,
            reason=self._build_reason(better, first, second, comparison),
            comparison=comparison,
            response_a=first,
            response_b=second,
        )

    @staticmethod
    def _compare_metric(
        attribute: str, first: EvaluationResponse, second: EvaluationResponse
    ) -> MetricComparison:
        label, higher_is_better = COMPARISON_METRICS[attribute]
        a_value = getattr(first, attribute)
        b_value = getattr(second, attribute)
        difference = round(a_value - b_value, 2)
        advantage = difference if higher_is_better else -difference
        winner: Literal["A", "B", "tie"]
        if abs(advantage) < METRIC_TIE_TOLERANCE:
            winner = "tie"
        else:
            winner = "A" if advantage > 0 else "B"
        return MetricComparison(
            metric=label,
            response_a=a_value,
            response_b=b_value,
            difference=difference,
            winner=winner,
        )

    @staticmethod
    def _build_reason(
        better: str,
        first: EvaluationResponse,
        second: EvaluationResponse,
        comparison: list[MetricComparison],
    ) -> str:
        scores = f"overall {first.overall_score:.1f} (A) vs {second.overall_score:.1f} (B)"
        if better == "tie":
            return f"Both responses are of practically equal quality: {scores}."

        decisive = [
            item
            for item in comparison
            if item.winner == better and item.metric != "overall score"
        ]
        decisive.sort(key=lambda item: abs(item.difference), reverse=True)
        highlights = [
            f"{'lower' if item.metric in ('hallucination risk', 'duplicate percentage') else 'higher'} "
            f"{item.metric} ({abs(item.difference):.1f} points)"
            for item in decisive[:REASON_HIGHLIGHTS]
        ]
        winner = first if better == "A" else second
        loser = second if better == "A" else first
        reason = f"Response {better} is better: {scores}."
        if highlights:
            reason += " Main advantages: " + " and ".join(highlights) + "."
        extra_warnings = len(loser.warnings) - len(winner.warnings)
        if extra_warnings > 0:
            reason += f" The other response triggered {extra_warnings} more rule warning(s)."
        return reason
