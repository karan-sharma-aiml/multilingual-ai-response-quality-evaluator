"""Quality label helpers."""

import pandas as pd

from app.ml.config import QUALITY_CLASSES, SCORE_COLUMN, TARGET_COLUMN, QualityThresholds


def label_from_score(score: float, thresholds: QualityThresholds | None = None) -> str:
    """Map an overall score (0-100) to Excellent / Good / Average / Poor."""
    limits = thresholds or QualityThresholds()
    if score >= limits.excellent_min:
        return "Excellent"
    if score >= limits.good_min:
        return "Good"
    if score >= limits.average_min:
        return "Average"
    return "Poor"


def add_labels(frame: pd.DataFrame, thresholds: QualityThresholds | None = None) -> pd.DataFrame:
    """Return a copy of ``frame`` with the ``quality_label`` column."""
    labelled = frame.copy()
    labelled[TARGET_COLUMN] = labelled[SCORE_COLUMN].map(lambda s: label_from_score(s, thresholds))
    labelled[TARGET_COLUMN] = pd.Categorical(labelled[TARGET_COLUMN], categories=QUALITY_CLASSES)
    return labelled
