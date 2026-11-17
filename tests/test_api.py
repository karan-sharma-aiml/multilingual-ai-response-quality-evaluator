"""Tests for the REST API."""

from app.core.exceptions import GeminiRateLimitError, GeminiTimeoutError


def post_evaluate(client, payload, **overrides):
    return client.post("/evaluate", json={**payload, **overrides})


# ------------------------------------------------------------------ health
def test_health_endpoint(client):
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["database"]["status"] == "ok"
    assert body["gemini"]["status"] == "ok"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_swagger_and_openapi_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/evaluate", "/batch-evaluate", "/results", "/results/{result_id}", "/compare", "/health"):
        assert path in paths


def test_cors_headers(client):
    response = client.options(
        "/evaluate",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "*"


# ---------------------------------------------------------------- evaluate
def test_evaluate_merges_gemini_and_rule_scores(client, evaluate_payload, fake_gemini):
    rule = post_evaluate(client, evaluate_payload, use_gemini=False).json()
    hybrid = post_evaluate(client, evaluate_payload).json()

    assert hybrid["evaluation_method"] == "hybrid"
    assert rule["evaluation_method"] == "rule_based"
    # weight 0.6 for Gemini, 0.4 for the rule engine
    expected = round(0.6 * 90.0 + 0.4 * rule["relevance_score"], 2)
    assert hybrid["relevance_score"] == expected
    assert hybrid["gemini"]["used"] is True
    assert hybrid["gemini"]["reasoning"] == "Mocked reasoning."
    assert hybrid["gemini"]["latency_ms"] == 12.5
    assert hybrid["language"] == "en"
    assert hybrid["language_name"] == "English"
    assert fake_gemini.calls == 1


def test_evaluate_stores_all_fields(client, evaluate_payload):
    body = post_evaluate(client, evaluate_payload).json()
    for field in (
        "id", "question", "response", "reference_answer", "language",
        "relevance_score", "correctness_score", "clarity_score", "safety_score",
        "hallucination_score", "duplicate_percentage", "overall_score",
        "evaluation_method", "created_at",
    ):
        assert body[field] is not None
    assert client.get(f"/results/{body['id']}").json()["question"] == evaluate_payload["question"]


def test_evaluate_status_code_is_201(client, evaluate_payload):
    assert post_evaluate(client, evaluate_payload).status_code == 201


def test_gemini_failure_falls_back_to_rule_based(client, evaluate_payload, fake_gemini):
    fake_gemini.error = GeminiTimeoutError("Gemini request timed out after 20s.")
    body = post_evaluate(client, evaluate_payload).json()
    assert body["evaluation_method"] == "rule_based"
    assert body["gemini"]["used"] is False
    assert "timed out" in body["gemini"]["error"]


def test_rate_limit_falls_back_to_rule_based(client, evaluate_payload, fake_gemini):
    fake_gemini.error = GeminiRateLimitError("Gemini rate limit exceeded.")
    assert post_evaluate(client, evaluate_payload).json()["evaluation_method"] == "rule_based"


def test_unsafe_response_is_capped(client, evaluate_payload):
    body = post_evaluate(
        client, evaluate_payload, response="You stupid idiot, kill them and make a bomb now."
    ).json()
    assert body["safety_score"] < 60
    assert body["overall_score"] <= 40


# -------------------------------------------------------------- validation
def test_missing_fields_return_422(client):
    response = client.post("/evaluate", json={"question": "Only a question?"})
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "validation_error"
    fields = {item["field"] for item in body["error"]["details"]}
    assert {"response", "language"} <= fields


def test_invalid_language_returns_422(client, evaluate_payload):
    response = post_evaluate(client, evaluate_payload, language="Klingon")
    assert response.status_code == 422
    assert "Unsupported language" in response.json()["error"]["details"][0]["message"]


def test_empty_response_returns_422(client, evaluate_payload):
    assert post_evaluate(client, evaluate_payload, response="   ").status_code == 422


def test_maximum_length_returns_422(client, evaluate_payload):
    assert post_evaluate(client, evaluate_payload, response="a" * 20_001).status_code == 422
    assert post_evaluate(client, evaluate_payload, question="q" * 2_001).status_code == 422


def test_unknown_field_is_rejected(client, evaluate_payload):
    assert post_evaluate(client, evaluate_payload, unexpected="x").status_code == 422


def test_language_code_is_accepted(client, evaluate_payload):
    body = post_evaluate(client, evaluate_payload, language="fr").json()
    assert body["language"] == "fr"


# ----------------------------------------------------------------- results
def seed(client, payload, count=5):
    ids = []
    for index in range(count):
        language = ["English", "French", "German"][index % 3]
        ids.append(
            post_evaluate(
                client, payload, language=language, category="Technology" if index % 2 else "Travel",
                use_gemini=index % 2 == 0,
            ).json()["id"]
        )
    return ids


def test_results_pagination(client, evaluate_payload):
    seed(client, evaluate_payload, 5)
    page = client.get("/results", params={"page": 2, "page_size": 2}).json()
    assert page["total"] == 5
    assert page["total_pages"] == 3
    assert len(page["items"]) == 2
    assert page["page"] == 2


def test_results_sorting(client, evaluate_payload):
    post_evaluate(client, evaluate_payload, use_gemini=False)
    post_evaluate(client, evaluate_payload, response="Bananas are yellow fruit.", use_gemini=False)
    ascending = client.get("/results", params={"sort_by": "overall_score", "order": "asc"}).json()
    scores = [item["overall_score"] for item in ascending["items"]]
    assert scores == sorted(scores)
    descending = client.get("/results", params={"sort_by": "overall_score", "order": "desc"}).json()
    assert [i["overall_score"] for i in descending["items"]] == sorted(scores, reverse=True)


def test_results_filtering(client, evaluate_payload):
    seed(client, evaluate_payload, 6)
    french = client.get("/results", params={"language": "French"}).json()
    assert french["total"] == 2
    assert {item["language"] for item in french["items"]} == {"fr"}
    hybrid = client.get("/results", params={"evaluation_method": "hybrid"}).json()
    assert {item["evaluation_method"] for item in hybrid["items"]} == {"hybrid"}
    category = client.get("/results", params={"category": "technology"}).json()
    assert category["total"] == 3
    high = client.get("/results", params={"min_overall_score": 101}).status_code
    assert high == 422


def test_results_invalid_filters(client):
    assert client.get("/results", params={"language": "Klingon"}).status_code == 422
    assert client.get("/results", params={"min_overall_score": 80, "max_overall_score": 10}).status_code == 422
    assert client.get("/results", params={"page_size": 1000}).status_code == 422
    assert client.get("/results", params={"sort_by": "nope"}).status_code == 422


def test_get_result_not_found(client):
    response = client.get("/results/9999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_delete_result(client, evaluate_payload):
    created = post_evaluate(client, evaluate_payload).json()
    deleted = client.delete(f"/results/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/results/{created['id']}").status_code == 404
    assert client.delete(f"/results/{created['id']}").status_code == 404


def test_unknown_route_uses_error_format(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ----------------------------------------------------------------- compare
def compare_payload(evaluate_payload, **overrides):
    payload = {
        "question": evaluate_payload["question"],
        "language": "English",
        "reference_answer": evaluate_payload["reference_answer"],
        "response_a": evaluate_payload["response"],
        "response_b": "Cloud computing is a weather technique that predicts rainfall using cameras.",
        "use_gemini": False,
    }
    payload.update(overrides)
    return payload


def test_compare_picks_better_response(client, evaluate_payload):
    response = client.post("/compare", json=compare_payload(evaluate_payload))
    body = response.json()
    assert response.status_code == 200
    assert body["better_response"] == "A"
    assert body["overall_difference"] > 0
    assert body["response_a"]["overall_score"] > body["response_b"]["overall_score"]
    assert body["reason"].startswith("Response A is better")
    assert {item["metric"] for item in body["comparison"]} >= {"relevance", "correctness", "hallucination risk"}
    assert client.get("/results").json()["total"] == 2


def test_compare_swapped_picks_b(client, evaluate_payload):
    payload = compare_payload(evaluate_payload)
    payload["response_a"], payload["response_b"] = payload["response_b"], payload["response_a"]
    assert client.post("/compare", json=payload).json()["better_response"] == "B"


def test_compare_identical_responses_is_tie(client, evaluate_payload):
    payload = compare_payload(evaluate_payload, response_b=evaluate_payload["response"])
    body = client.post("/compare", json=payload).json()
    assert body["better_response"] == "tie"
    assert "equal quality" in body["reason"]


def test_compare_validation(client, evaluate_payload):
    payload = compare_payload(evaluate_payload)
    del payload["response_b"]
    assert client.post("/compare", json=payload).status_code == 422
