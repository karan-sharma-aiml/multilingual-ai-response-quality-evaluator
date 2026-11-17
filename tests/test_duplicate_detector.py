"""Tests for the duplicate detector."""

from app.evaluator.duplicate_detector import DuplicateDetector


def test_unique_text_has_no_duplicates(make_sample):
    assert DuplicateDetector().evaluate(make_sample()) < 5


def test_repeated_sentences_are_detected():
    sentence = "The cloud stores data on remote servers."
    report = DuplicateDetector().analyze(" ".join([sentence] * 6), "en")
    assert report.repeated_sentence_ratio > 0.8
    assert report.percentage > 70


def test_looping_output_is_detected():
    text = "alpha beta gamma delta epsilon zeta " * 8
    report = DuplicateDetector().analyze(text, "en")
    assert report.is_looping
    assert report.percentage > 60


def test_repeated_words_are_detected():
    report = DuplicateDetector().analyze("very very very very very good good good good", "en")
    assert report.repeated_word_ratio > 0.4


def test_empty_text_is_zero():
    assert DuplicateDetector().analyze("", "en").percentage == 0.0


def test_percentage_is_bounded():
    report = DuplicateDetector().analyze("yes " * 500, "en")
    assert 0.0 <= report.percentage <= 100.0
