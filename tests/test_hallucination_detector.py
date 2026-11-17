"""Tests for the hallucination detector."""

from app.evaluator.hallucination import HallucinationDetector


def test_grounded_response_has_low_risk(make_sample):
    assert HallucinationDetector().evaluate(make_sample()) < 15


def test_fake_url_is_detected(make_sample):
    sample = make_sample(
        response="Cloud computing is great. See https://www.cloud-facts-2024.example.net/study/8837 for details."
    )
    report = HallucinationDetector().analyze(sample)
    assert report.fake_urls
    assert report.score >= 25


def test_url_present_in_reference_is_trusted(make_sample):
    sample = make_sample(
        response="Details are at https://docs.example.org/cloud for reference.",
        reference_answer="Cloud computing is described at https://docs.example.org/cloud in detail.",
    )
    assert not HallucinationDetector().analyze(sample).fake_urls


def test_random_statistics_are_detected(make_sample):
    sample = make_sample(response="Cloud computing is used by 87.4% of companies and 14,000 firms.")
    report = HallucinationDetector().analyze(sample)
    assert len(report.unsupported_statistics) == 2


def test_supported_statistic_is_not_flagged(make_sample):
    sample = make_sample(
        response="About 78% of the atmosphere is nitrogen.",
        reference_answer="The atmosphere is about 78% nitrogen.",
        question="What is the atmosphere made of?",
    )
    assert not HallucinationDetector().analyze(sample).unsupported_statistics


def test_unsupported_claims_are_detected(make_sample):
    sample = make_sample(response="According to a 2023 study, scientists have proven this fact.")
    assert HallucinationDetector().analyze(sample).unsupported_claims


def test_contradictory_statements_are_detected(make_sample):
    sample = make_sample(
        response="The cloud stores your business data on remote servers. "
        "The cloud does not store your business data on remote servers."
    )
    assert HallucinationDetector().analyze(sample).contradictions


def test_made_up_entities_are_detected(make_sample):
    sample = make_sample(
        response="Dr. Marcus Velloran of the International Institute of Verified Knowledge proved it."
    )
    report = HallucinationDetector().analyze(sample)
    assert report.made_up_entities


def test_hallucinated_response_scores_higher_than_grounded(make_sample):
    detector = HallucinationDetector()
    grounded = detector.evaluate(make_sample())
    hallucinated = detector.evaluate(
        make_sample(
            response="According to a 2023 study by the International Institute of Verified "
            "Knowledge, 87.4% of experts agree. Dr. Marcus Velloran proved it. "
            "Visit https://www.verified-facts-1.example.net/study/1."
        )
    )
    assert hallucinated > grounded + 40


def test_score_is_bounded(make_sample):
    sample = make_sample(response="https://a1.example.com https://b2.example.com https://c3.example.com " * 5)
    assert 0 <= HallucinationDetector().evaluate(sample) <= 100
