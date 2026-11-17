"""Tests for the overall score engine."""

import pytest

from app.evaluator.overall_score import OverallScoreConfig, OverallScoreEngine, ScoreComponents
from app.evaluator.rule_engine import RuleCode, RuleSeverity, RuleWarning


def components(**overrides: float) -> ScoreComponents:
    values = dict(
        relevance=100.0, correctness=100.0, clarity=100.0,
        safety=100.0, hallucination=0.0, duplicate_percentage=0.0,
    )
    values.update(overrides)
    return ScoreComponents(**values)


def test_perfect_components_give_100():
    assert OverallScoreEngine().calculate(components()) == 100.0


def test_worst_components_give_0():
    worst = components(
        relevance=0, correctness=0, clarity=0, safety=0,
        hallucination=100, duplicate_percentage=100,
    )
    assert OverallScoreEngine().calculate(worst) == 0.0


def test_duplicate_penalty_reduces_score():
    engine = OverallScoreEngine()
    assert engine.calculate(components(duplicate_percentage=80)) < engine.calculate(components())


def test_small_duplicate_percentage_is_tolerated():
    assert OverallScoreEngine().calculate(components(duplicate_percentage=8)) == 100.0


def test_hallucination_lowers_score():
    engine = OverallScoreEngine()
    assert engine.calculate(components(hallucination=80)) < engine.calculate(components(hallucination=0))


def test_unsafe_response_is_capped():
    config = OverallScoreConfig()
    score = OverallScoreEngine(config).calculate(components(safety=10))
    assert score <= config.safety_cap


def test_empty_response_warning_forces_zero():
    warning = RuleWarning(RuleCode.EMPTY_RESPONSE, "empty", RuleSeverity.CRITICAL)
    assert OverallScoreEngine().calculate(components(), [warning]) == 0.0


def test_warnings_apply_penalty():
    warning = RuleWarning(RuleCode.SPAM, "spam", RuleSeverity.CRITICAL)
    engine = OverallScoreEngine()
    assert engine.calculate(components(), [warning]) < 100.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        OverallScoreConfig(relevance_weight=0.5)


def test_score_always_within_bounds():
    engine = OverallScoreEngine()
    for relevance in (0, 50, 100):
        for dup in (0, 50, 100):
            score = engine.calculate(components(relevance=relevance, duplicate_percentage=dup))
            assert 0.0 <= score <= 100.0
