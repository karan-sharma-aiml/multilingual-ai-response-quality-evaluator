"""Tests for the Gemini integration using a mocked HTTP transport."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiAPIError,
    GeminiInvalidResponseError,
    GeminiNotConfiguredError,
    GeminiRateLimitError,
    GeminiTimeoutError,
)
from app.evaluator import ResponseEvaluator
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationService
from app.services.gemini_service import GeminiEvaluationService

GOOD_JSON = {"relevance": 92, "correctness": 85, "clarity": 78, "safety": 100, "reasoning": "Accurate."}


def gemini_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def build_service(handler, **settings_kwargs) -> tuple[GeminiEvaluationService, list[float]]:
    sleeps: list[float] = []
    options = dict(
        gemini_api_key="secret-key",
        gemini_max_retries=2,
        gemini_retry_backoff_seconds=1.0,
        gemini_max_retry_delay_seconds=5.0,
    )
    options.update(settings_kwargs)
    settings = Settings(_env_file=None, **options)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiEvaluationService(settings, client=client, sleep=sleeps.append), sleeps


def test_successful_evaluation_and_request_shape(make_sample):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.headers["x-goog-api-key"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=gemini_payload(json.dumps(GOOD_JSON)))

    service, sleeps = build_service(handler)
    result = service.evaluate(make_sample())

    assert result.scores() == {"relevance": 92, "correctness": 85, "clarity": 78, "safety": 100}
    assert result.reasoning == "Accurate."
    assert result.attempts == 1 and result.latency_ms >= 0
    assert ":generateContent" in captured["url"]
    assert captured["key"] == "secret-key"
    assert "secret-key" not in captured["url"]
    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "What is cloud computing?" in prompt and "<reference>" in prompt
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert sleeps == []


def test_json_wrapped_in_code_fence_is_accepted(make_sample):
    text = "```json\n" + json.dumps(GOOD_JSON) + "\n```"
    service, _ = build_service(lambda r: httpx.Response(200, json=gemini_payload(text)))
    assert service.evaluate(make_sample()).relevance == 92


def test_json_with_surrounding_text_is_accepted(make_sample):
    text = "Here you go: " + json.dumps(GOOD_JSON) + " Thanks!"
    service, _ = build_service(lambda r: httpx.Response(200, json=gemini_payload(text)))
    assert service.evaluate(make_sample()).clarity == 78


def test_invalid_json_is_retried_then_raises(make_sample):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=gemini_payload("definitely not json"))

    service, sleeps = build_service(handler)
    with pytest.raises(GeminiInvalidResponseError):
        service.evaluate(make_sample())
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_invalid_json_recovers_on_retry(make_sample):
    responses = iter([gemini_payload("oops"), gemini_payload(json.dumps(GOOD_JSON))])
    service, sleeps = build_service(lambda r: httpx.Response(200, json=next(responses)))
    result = service.evaluate(make_sample())
    assert result.attempts == 2 and len(sleeps) == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"relevance": 120, "correctness": 1, "clarity": 1, "safety": 1},
        {"relevance": 50, "correctness": 50, "clarity": 50},
        [1, 2, 3],
    ],
)
def test_schema_violations_are_rejected(make_sample, bad):
    service, _ = build_service(
        lambda r: httpx.Response(200, json=gemini_payload(json.dumps(bad))), gemini_max_retries=0
    )
    with pytest.raises(GeminiInvalidResponseError):
        service.evaluate(make_sample())


def test_timeout_is_retried_then_raises(make_sample):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    service, sleeps = build_service(handler)
    with pytest.raises(GeminiTimeoutError):
        service.evaluate(make_sample())
    assert len(sleeps) == 2


def test_rate_limit_honours_retry_after_and_caps_delay(make_sample):
    service, sleeps = build_service(
        lambda r: httpx.Response(429, headers={"Retry-After": "60"}, json={})
    )
    with pytest.raises(GeminiRateLimitError) as info:
        service.evaluate(make_sample())
    assert info.value.retry_after == 60.0
    assert sleeps == [5.0, 5.0]  # capped by gemini_max_retry_delay_seconds


def test_rate_limit_recovers(make_sample):
    responses = iter(
        [httpx.Response(429, json={}), httpx.Response(200, json=gemini_payload(json.dumps(GOOD_JSON)))]
    )
    service, _ = build_service(lambda r: next(responses))
    assert service.evaluate(make_sample()).attempts == 2


def test_server_error_is_retried(make_sample):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={})

    service, _ = build_service(handler)
    with pytest.raises(GeminiAPIError):
        service.evaluate(make_sample())
    assert len(calls) == 3


def test_client_error_is_not_retried(make_sample):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    service, sleeps = build_service(handler)
    with pytest.raises(GeminiAPIError, match="API key not valid"):
        service.evaluate(make_sample())
    assert len(calls) == 1 and sleeps == []


def test_network_error_is_wrapped(make_sample):
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    service, _ = build_service(handler, gemini_max_retries=0)
    with pytest.raises(GeminiAPIError):
        service.evaluate(make_sample())


def test_blocked_prompt_is_not_retried(make_sample):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})

    service, _ = build_service(handler)
    with pytest.raises(GeminiInvalidResponseError, match="blocked"):
        service.evaluate(make_sample())
    assert len(calls) == 1


def test_missing_api_key_raises(make_sample):
    service = GeminiEvaluationService(Settings(gemini_api_key=None, _env_file=None))
    assert service.is_configured is False
    with pytest.raises(GeminiNotConfiguredError):
        service.evaluate(make_sample())


def test_long_inputs_are_truncated_in_prompt(make_sample):
    captured = {}

    def handler(request):
        captured["prompt"] = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        return httpx.Response(200, json=gemini_payload(json.dumps(GOOD_JSON)))

    service, _ = build_service(handler, gemini_max_input_chars=200)
    service.evaluate(make_sample(response="word " * 5000))
    assert len(captured["prompt"]) < 1000


def test_end_to_end_merge_with_mocked_http(db_session, make_sample):
    service, _ = build_service(
        lambda r: httpx.Response(200, json=gemini_payload(json.dumps(GOOD_JSON)))
    )
    settings = Settings(gemini_weight=0.5, _env_file=None)
    evaluation = EvaluationService(
        ResponseEvaluator(), service, DatabaseService(db_session), settings
    )
    sample = make_sample()
    rule = ResponseEvaluator().evaluate(sample)
    outcome = evaluation.evaluate(sample, use_gemini=True)

    assert outcome.result.relevance_score == round(0.5 * 92 + 0.5 * rule.relevance_score, 2)
    assert outcome.result.correctness_score == round(0.5 * 85 + 0.5 * rule.correctness_score, 2)
    assert outcome.result.safety_score == min(100, rule.safety_score)
    assert outcome.result.hallucination_score == rule.hallucination_score
    assert outcome.gemini.used and outcome.gemini.scores["relevance"] == 92


def test_safety_takes_the_stricter_score(db_session, make_sample, fake_gemini):
    fake_gemini.scores["safety"] = 10.0
    evaluation = EvaluationService(
        ResponseEvaluator(), fake_gemini, DatabaseService(db_session), Settings(_env_file=None)
    )
    assert evaluation.evaluate(make_sample()).result.safety_score == 10.0
