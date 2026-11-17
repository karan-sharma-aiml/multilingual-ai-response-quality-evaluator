"""Numeric helpers shared by all evaluators."""

from collections.abc import Mapping

SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0
SCORE_PRECISION: int = 2


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Restrict ``value`` to the closed interval ``[lower, upper]``."""
    return max(lower, min(upper, value))


def clamp_score(value: float) -> float:
    """Clamp to the 0-100 score range and round for stable output."""
    return round(clamp(value, SCORE_MIN, SCORE_MAX), SCORE_PRECISION)


def ratio_to_score(ratio: float) -> float:
    """Convert a 0-1 ratio into a 0-100 score."""
    return clamp_score(ratio * SCORE_MAX)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is zero."""
    return numerator / denominator if denominator else default


def linear_band_score(
    value: float,
    ideal_low: float,
    ideal_high: float,
    hard_low: float,
    hard_high: float,
) -> float:
    """Score 1.0 inside the ideal band and decay linearly to 0 at hard limits."""
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return clamp(safe_divide(value - hard_low, ideal_low - hard_low))
    return clamp(safe_divide(hard_high - value, hard_high - ideal_high))


def weighted_average(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Weighted mean of ``values`` using the matching keys in ``weights``."""
    total_weight = sum(weights[key] for key in values)
    weighted_sum = sum(values[key] * weights[key] for key in values)
    return safe_divide(weighted_sum, total_weight)
