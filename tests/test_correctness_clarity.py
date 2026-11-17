"""Tests for correctness and clarity evaluators."""

from app.evaluator.clarity import ClarityEvaluator
from app.evaluator.correctness import CorrectnessEvaluator


def test_correct_response_beats_wrong_response(make_sample):
    evaluator = CorrectnessEvaluator()
    correct = evaluator.evaluate(make_sample())
    wrong = evaluator.evaluate(
        make_sample(response="Cloud computing is a weather technique that predicts rainfall.")
    )
    assert correct > wrong + 20


def test_number_mismatch_is_penalised(make_sample):
    evaluator = CorrectnessEvaluator()
    kwargs = dict(
        question="Who was the first US President?",
        reference_answer="George Washington was the first President, serving from 1789 to 1797.",
    )
    right = evaluator.evaluate(make_sample(response="George Washington served from 1789 to 1797 as first President.", **kwargs))
    wrong = evaluator.evaluate(make_sample(response="George Washington served from 1801 to 1805 as first President.", **kwargs))
    assert right > wrong


def test_heuristic_mode_is_bounded_and_conservative(make_sample):
    score = CorrectnessEvaluator().evaluate(make_sample(reference_answer=None))
    assert 20 <= score <= 85


def test_uncertain_response_scores_lower_without_reference(make_sample):
    evaluator = CorrectnessEvaluator()
    sure = evaluator.evaluate(make_sample(reference_answer=None))
    unsure = evaluator.evaluate(
        make_sample(reference_answer=None, response="Maybe cloud computing is something, I don't know, perhaps.")
    )
    assert sure > unsure


def test_clear_text_beats_messy_text(make_sample):
    evaluator = ClarityEvaluator()
    clear = evaluator.evaluate(make_sample())
    messy = evaluator.evaluate(
        make_sample(response="cloud cloud cloud  cloud , is is the the cloud cloud cloud (thing")
    )
    assert clear > messy + 20


def test_clarity_empty_is_zero(make_sample):
    assert ClarityEvaluator().evaluate(make_sample(response="")) == 0.0


def test_clarity_penalises_very_short_response(make_sample):
    evaluator = ClarityEvaluator()
    assert evaluator.evaluate(make_sample(response="Yes.")) < evaluator.evaluate(make_sample())
