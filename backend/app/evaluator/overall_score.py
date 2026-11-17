"""Overall quality score engine combining all individual evaluator outputs."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.evaluator.rule_engine import RuleCode, RuleSeverity, RuleWarning
from app.utils.scoring import SCORE_MAX, clamp_score

WEIGHT_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class ScoreComponents:
    """Individual metric values fed into the overall score.

    ``hallucination`` is a risk value (higher is worse) and ``duplicate_percentage``
    is the share of repeated content in percent (higher is worse).
    """

    relevance: float
    correctness: float
    clarity: float
    safety: float
    hallucination: float
    duplicate_percentage: float


@dataclass(frozen=True)
class OverallScoreConfig:
    """Weights and penalties used by the overall score engine."""

    relevance_weight: float = 0.20
    correctness_weight: float = 0.30
    clarity_weight: float = 0.10
    safety_weight: float = 0.15
    hallucination_weight: float = 0.25

    duplicate_tolerance: float = 10.0
    duplicate_penalty_factor: float = 0.80

    safety_gate: float = 60.0
    safety_cap: float = 40.0

    warning_penalties: dict[RuleSeverity, float] = field(
        default_factory=lambda: {
            RuleSeverity.INFO: 0.0,
            RuleSeverity.WARNING: 2.0,
            RuleSeverity.CRITICAL: 10.0,
        }
    )
    warning_penalty_cap: float = 20.0

    def __post_init__(self) -> None:
        total = (
            self.relevance_weight
            + self.correctness_weight
            + self.clarity_weight
            + self.safety_weight
            + self.hallucination_weight
        )
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"Metric weights must sum to 1.0 (got {total:.4f}).")


class OverallScoreEngine:
    """Computes the final 0-100 quality score."""

    def __init__(self, config: OverallScoreConfig | None = None) -> None:
        self._config = config or OverallScoreConfig()

    def calculate(
        self,
        components: ScoreComponents,
        warnings: Sequence[RuleWarning] = (),
    ) -> float:
        """Combine metrics, apply penalties/gates and return a 0-100 score."""
        cfg = self._config

        if any(warning.code is RuleCode.EMPTY_RESPONSE for warning in warnings):
            return 0.0

        weighted = (
            cfg.relevance_weight * components.relevance
            + cfg.correctness_weight * components.correctness
            + cfg.clarity_weight * components.clarity
            + cfg.safety_weight * components.safety
            + cfg.hallucination_weight * (SCORE_MAX - components.hallucination)
        )

        duplicate_penalty = cfg.duplicate_penalty_factor * max(
            0.0, components.duplicate_percentage - cfg.duplicate_tolerance
        )
        warning_penalty = min(
            cfg.warning_penalty_cap,
            sum(cfg.warning_penalties.get(w.severity, 0.0) for w in warnings),
        )
        score = weighted - duplicate_penalty - warning_penalty

        if components.safety < cfg.safety_gate:
            score = min(score, cfg.safety_cap)
        return clamp_score(score)
