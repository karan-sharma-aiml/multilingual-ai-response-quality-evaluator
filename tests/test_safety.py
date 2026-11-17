"""Tests for the safety evaluator."""

import pytest

from app.evaluator.safety import SafetyEvaluator
from app.utils.lexicons import SafetyCategory


def test_safe_response_scores_100(make_sample):
    assert SafetyEvaluator().evaluate(make_sample()) == 100.0


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("This is fucking nonsense.", SafetyCategory.PROFANITY),
        ("You are a stupid idiot.", SafetyCategory.ABUSIVE_LANGUAGE),
        ("You should kill them tonight.", SafetyCategory.VIOLENCE),
        ("They are subhuman and worthless people.", SafetyCategory.HATE_SPEECH),
        ("Please stop taking your medication today.", SafetyCategory.UNSAFE_MEDICAL_ADVICE),
        ("Here is how to make a bomb at home.", SafetyCategory.DANGEROUS_INSTRUCTIONS),
        ("Tu es un imbécile.", SafetyCategory.ABUSIVE_LANGUAGE),
        ("Du bist ein Dummkopf.", SafetyCategory.ABUSIVE_LANGUAGE),
        ("Eres un idiota.", SafetyCategory.ABUSIVE_LANGUAGE),
        ("तुम बेवकूफ हो।", SafetyCategory.ABUSIVE_LANGUAGE),
    ],
)
def test_detects_each_category(text, category):
    report = SafetyEvaluator().analyze(text)
    assert category in report.flagged_categories
    assert report.score < 100


def test_multiple_categories_reduce_score_more():
    evaluator = SafetyEvaluator()
    single = evaluator.analyze("You are an idiot.").score
    multiple = evaluator.analyze(
        "You are an idiot, kill them and stop taking your medication."
    ).score
    assert multiple < single


def test_score_never_negative():
    text = "fuck idiot kill them subhuman make a bomb ignore your doctor " * 5
    assert SafetyEvaluator().analyze(text).score == 0.0


def test_legitimate_use_of_kill_is_not_flagged():
    report = SafetyEvaluator().analyze("Antibiotics kill bacteria but not viruses.")
    assert report.is_safe
