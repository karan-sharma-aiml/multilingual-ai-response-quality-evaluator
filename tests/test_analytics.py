"""Tests for the analytics API and service."""

from datetime import date, datetime, timedelta, timezone

import pytest

from app.analytics.service import AnalyticsService
from app.core.cache import TTLCache
from app.models import EvaluationResultRecord
from app.schemas.analytics import TrendPeriod


def all_results(client):
    return client.get("/results", params={"page_size": 100}).json()["items"]


def mean(values):
    return sum(values) / len(values)


def test_overview_empty(client):
    body = client.get("/analytics/overview").json()
    assert body["total_evaluations"] == 0
    assert body["average_overall_score"] is None


def test_overview_matches_results(seeded_client):
    items = all_results(seeded_client)
    body = seeded_client.get("/analytics/overview").json()
    assert body["total_evaluations"] == len(items) == 6
    assert body["average_overall_score"] == pytest.approx(mean([i["overall_score"] for i in items]), abs=0.01)
    assert body["average_relevance"] == pytest.approx(mean([i["relevance_score"] for i in items]), abs=0.01)
    assert body["average_correctness"] == pytest.approx(mean([i["correctness_score"] for i in items]), abs=0.01)
    assert body["average_clarity"] == pytest.approx(mean([i["clarity_score"] for i in items]), abs=0.01)
    assert body["average_safety"] == pytest.approx(mean([i["safety_score"] for i in items]), abs=0.01)
    assert body["average_hallucination"] == pytest.approx(mean([i["hallucination_score"] for i in items]), abs=0.01)
    assert body["low_quality_count"] == sum(1 for i in items if i["overall_score"] < 50)


def test_languages(seeded_client):
    items = all_results(seeded_client)
    body = seeded_client.get("/analytics/languages").json()
    by_code = {row["language"]: row for row in body["languages"]}
    assert set(by_code) == {"en", "fr", "de"}
    assert by_code["en"]["total_evaluations"] == 4
    expected = {
        code: mean([i["overall_score"] for i in items if i["language"] == code]) for code in by_code
    }
    assert by_code["fr"]["average_overall_score"] == pytest.approx(expected["fr"], abs=0.01)
    assert body["best_language"]["language"] == max(expected, key=expected.get)
    assert body["worst_language"]["language"] == min(expected, key=expected.get)
    assert body["best_language"]["average_score"] >= body["worst_language"]["average_score"]


def test_categories(seeded_client):
    body = seeded_client.get("/analytics/categories").json()
    stats = {row["category"]: row for row in body["categories"]}
    assert set(stats) == {"Technology", "Science", "Travel", "Uncategorized"}
    assert stats["Technology"]["total_evaluations"] == 3
    assert stats["Uncategorized"]["total_evaluations"] == 1
    averages = [row["average_overall_score"] for row in body["categories"]]
    assert averages == sorted(averages, reverse=True)
    assert body["best_category"] == body["categories"][0]["category"]


def test_hallucination_types(seeded_client):
    body = seeded_client.get("/analytics/hallucinations").json()
    types = {row["type"]: row for row in body["types"]}
    assert {"fake_urls", "unsupported_statistics", "unsupported_claims", "made_up_entities"} <= set(types)
    assert types["fake_urls"]["frequency"] == 1
    assert body["evaluations_with_signals"] >= 1
    assert sum(row["percentage"] for row in body["types"]) == pytest.approx(100, abs=0.1)
    frequencies = [row["frequency"] for row in body["types"]]
    assert frequencies == sorted(frequencies, reverse=True)
    assert types["fake_urls"]["prevalence_percentage"] == pytest.approx(1 / 6 * 100, abs=0.01)


def test_hallucination_types_recomputed_for_legacy_rows(seeded_client, db_engine):
    from sqlalchemy.orm import Session

    with Session(db_engine) as session:
        for record in session.query(EvaluationResultRecord):
            record.details = {}  # simulate rows stored before types were recorded
        session.commit()
    body = seeded_client.get("/analytics/hallucinations").json()
    assert "fake_urls" in {row["type"] for row in body["types"]}


def test_low_quality(seeded_client):
    items = all_results(seeded_client)
    expected = sorted(i["overall_score"] for i in items if i["overall_score"] < 50)
    body = seeded_client.get("/analytics/low-quality").json()
    assert body["threshold"] == 50
    assert body["total"] == len(expected) >= 2
    assert all(item["overall_score"] < 50 for item in body["items"])
    assert [i["overall_score"] for i in body["items"]] == expected  # worst first
    strict = seeded_client.get("/analytics/low-quality", params={"threshold": 0}).json()
    assert strict["total"] == 0
    paged = seeded_client.get("/analytics/low-quality", params={"page_size": 1, "page": 2}).json()
    assert len(paged["items"]) == 1 and paged["total_pages"] == body["total"]


def test_low_quality_filters_and_validation(seeded_client):
    french = seeded_client.get("/analytics/low-quality", params={"threshold": 101}).status_code
    assert french == 422
    assert seeded_client.get("/analytics/low-quality", params={"language": "Klingon"}).status_code == 422
    english = seeded_client.get("/analytics/low-quality", params={"language": "en"}).json()
    assert {i["language"] for i in english["items"]} <= {"en"}


def test_trends_all_periods(seeded_client):
    body = seeded_client.get("/analytics/trends").json()
    for period in ("daily", "weekly", "monthly"):
        points = body[period]
        assert sum(p["evaluations"] for p in points) == 6
    today = datetime.now(timezone.utc).date()
    assert body["daily"][-1]["period"] == today.isoformat()
    week_start = today - timedelta(days=today.weekday())
    assert body["weekly"][-1]["start_date"] == week_start.isoformat()
    assert body["monthly"][-1]["period"] == today.strftime("%Y-%m")
    assert body["daily"][0]["average_overall_score"] == pytest.approx(
        mean([i["overall_score"] for i in all_results(seeded_client)]), abs=0.01
    )


def test_trends_period_and_date_filters(seeded_client):
    only = seeded_client.get("/analytics/trends", params={"period": "weekly"}).json()
    assert only["weekly"] and only["daily"] is None and only["monthly"] is None
    future = (date.today() + timedelta(days=30)).isoformat()
    empty = seeded_client.get("/analytics/trends", params={"start_date": future}).json()
    assert empty["daily"] == []
    bad = seeded_client.get("/analytics/trends", params={"start_date": "2030-01-02", "end_date": "2030-01-01"})
    assert bad.status_code == 422


def test_trends_empty_database(client):
    assert client.get("/analytics/trends").json()["daily"] == []


def test_analytics_are_cached_and_invalidated(db_session, seeded_client):
    cache = TTLCache(60, 10)
    service = AnalyticsService(db_session, cache=cache)
    first = service.overview()
    second = service.overview()
    assert first is second and cache.hits == 1 and cache.misses == 1
    seeded_client.post(
        "/evaluate",
        json={"question": "What is gravity?", "response": "A force that pulls masses together.",
              "language": "en", "use_gemini": False},
    )
    # The API writes through DatabaseService, which clears the shared analytics cache.
    assert seeded_client.get("/analytics/overview").json()["total_evaluations"] == 7


def test_service_without_cache(db_session):
    assert AnalyticsService(db_session, cache=None).overview().total_evaluations == 0
    assert AnalyticsService(db_session).trends(TrendPeriod.DAILY).daily == []
