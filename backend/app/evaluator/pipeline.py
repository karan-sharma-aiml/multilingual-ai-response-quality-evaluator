"""Facade orchestrating every evaluator into a single evaluation call."""

from collections.abc import Iterable

from app.core.logging import get_logger
from app.evaluator.base import BaseEvaluator
from app.evaluator.clarity import ClarityEvaluator
from app.evaluator.correctness import CorrectnessEvaluator
from app.evaluator.duplicate_detector import DuplicateDetector
from app.evaluator.hallucination import HallucinationDetector
from app.evaluator.overall_score import OverallScoreEngine, ScoreComponents
from app.evaluator.relevance import RelevanceEvaluator
from app.evaluator.rule_engine import RuleEngine
from app.evaluator.safety import SafetyEvaluator
from app.schemas.evaluation import EvaluationInput, EvaluationResult

logger = get_logger("pipeline")


class ResponseEvaluator:
    """Runs all evaluators for a sample and builds an ``EvaluationResult``.

    Every collaborator can be injected (Dependency Inversion), which lets later
    parts of the project swap or extend individual evaluators.
    """

    def __init__(
        self,
        relevance: BaseEvaluator | None = None,
        correctness: BaseEvaluator | None = None,
        clarity: BaseEvaluator | None = None,
        safety: BaseEvaluator | None = None,
        hallucination: BaseEvaluator | None = None,
        duplicate: BaseEvaluator | None = None,
        rule_engine: RuleEngine | None = None,
        overall: OverallScoreEngine | None = None,
    ) -> None:
        self._relevance = relevance or RelevanceEvaluator()
        self._correctness = correctness or CorrectnessEvaluator()
        self._clarity = clarity or ClarityEvaluator()
        self._safety = safety or SafetyEvaluator()
        self._hallucination = hallucination or HallucinationDetector()
        self._duplicate = duplicate or DuplicateDetector()
        self._rule_engine = rule_engine or RuleEngine()
        self._overall = overall or OverallScoreEngine()

    def evaluate(self, sample: EvaluationInput) -> EvaluationResult:
        """Evaluate one sample."""
        components = ScoreComponents(
            relevance=self._relevance.evaluate(sample),
            correctness=self._correctness.evaluate(sample),
            clarity=self._clarity.evaluate(sample),
            safety=self._safety.evaluate(sample),
            hallucination=self._hallucination.evaluate(sample),
            duplicate_percentage=self._duplicate.evaluate(sample),
        )
        warnings = self._rule_engine.evaluate(sample)
        overall_score = self._overall.calculate(components, warnings)
        logger.debug(
            "Evaluated sample (language=%s) overall=%.2f warnings=%d",
            sample.language,
            overall_score,
            len(warnings),
        )
        return EvaluationResult(
            relevance_score=components.relevance,
            correctness_score=components.correctness,
            clarity_score=components.clarity,
            safety_score=components.safety,
            hallucination_score=components.hallucination,
            duplicate_percentage=components.duplicate_percentage,
            overall_score=overall_score,
            warnings=self._rule_engine.messages(warnings),
        )

    def evaluate_many(self, samples: Iterable[EvaluationInput]) -> list[EvaluationResult]:
        """Evaluate several samples, preserving order."""
        return [self.evaluate(sample) for sample in samples]
