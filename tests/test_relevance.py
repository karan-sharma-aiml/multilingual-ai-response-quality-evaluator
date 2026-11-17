"""Tests for the relevance evaluator."""

from app.evaluator.relevance import RelevanceEvaluator


def test_relevant_response_scores_high(make_sample):
    score = RelevanceEvaluator().evaluate(make_sample())
    assert 60 <= score <= 100


def test_irrelevant_response_scores_low(make_sample):
    sample = make_sample(response="Bananas are yellow fruit that monkeys enjoy eating daily.")
    assert RelevanceEvaluator().evaluate(sample) < 20


def test_relevant_beats_irrelevant(make_sample):
    evaluator = RelevanceEvaluator()
    good = evaluator.evaluate(make_sample())
    bad = evaluator.evaluate(make_sample(response="Bananas are yellow fruit."))
    assert good > bad


def test_works_without_reference(make_sample):
    score = RelevanceEvaluator().evaluate(make_sample(reference_answer=None))
    assert 0 <= score <= 100
    assert score > 50


def test_empty_response_scores_zero(make_sample):
    assert RelevanceEvaluator().evaluate(make_sample(response="")) == 0.0


def test_hindi_relevance(make_sample):
    sample = make_sample(
        question="पानी का रासायनिक सूत्र क्या है?",
        language="Hindi",
        response="पानी का रासायनिक सूत्र H2O है। इसमें हाइड्रोजन के दो परमाणु और ऑक्सीजन का एक परमाणु होता है।",
        reference_answer="पानी का रासायनिक सूत्र H2O है, जिसमें हाइड्रोजन के दो परमाणु और ऑक्सीजन का एक परमाणु होता है।",
    )
    assert RelevanceEvaluator().evaluate(sample) > 70
