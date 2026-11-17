"""Configuration of the quality classifier."""

from dataclasses import dataclass

FEATURE_NAMES: tuple[str, ...] = (
    "relevance_score",
    "correctness_score",
    "clarity_score",
    "safety_score",
    "hallucination_score",
    "duplicate_percentage",
)
QUALITY_CLASSES: tuple[str, ...] = ("Poor", "Average", "Good", "Excellent")
TARGET_COLUMN: str = "quality_label"
SCORE_COLUMN: str = "overall_score"
RANDOM_STATE: int = 42


@dataclass(frozen=True)
class QualityThresholds:
    """Overall-score cut-offs used to derive class labels."""

    average_min: float = 50.0
    good_min: float = 70.0
    excellent_min: float = 85.0


@dataclass(frozen=True)
class TrainingConfig:
    """Random-forest and evaluation settings."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    test_size: float = 0.25
    cv_folds: int = 5
    random_state: int = RANDOM_STATE
