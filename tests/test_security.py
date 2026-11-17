"""Tests for authentication, rate limiting, CORS, headers and sanitisation."""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db, get_gemini_service
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.core.security import is_valid_api_key
from app.database.session import create_session_factory
from app.main import create_app
from app.utils.sanitize import sanitize_text

PAYLOAD = {"question": "What is gravity?", "response": "A force that attracts masses.", "language": "en", "use_gemini": False}


@pytest.fixture
def make_client(db_engine, fake_gemini):
    def build(**options) -> TestClient:
        settings = Settings(_env_file=None, **options)
        app = create_app(settings=settings, initialize_db=False)
        factory = create_session_factory(db_engine)

        def override_db():
            session = factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_gemini_service] = lambda: fake_gemini
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app, raise_server_exceptions=False)

    return build


# ------------------------------------------------------------------ API key
def test_api_key_required_when_configured(make_client):
    client = make_client(api_keys="alpha-key,beta-key")
    assert client.get("/results").status_code == 401
    assert client.post("/evaluate", json=PAYLOAD).json()["error"]["code"] == "unauthorized"
    assert client.get("/results", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/results", headers={"X-API-Key": "beta-key"}).status_code == 200
    assert client.post("/evaluate", json=PAYLOAD, headers={"X-API-Key": "alpha-key"}).status_code == 201


@pytest.mark.parametrize("path", ["/analytics/overview", "/reports/csv", "/results/1", "/analytics/trends"])
def test_every_protected_router_requires_key(make_client, path):
    assert make_client(api_keys="k").get(path).status_code == 401


def test_health_and_docs_stay_public(make_client):
    client = make_client(api_keys="k")
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200


def test_auth_disabled_without_keys(make_client):
    assert make_client().get("/results").status_code == 200


def test_openapi_declares_api_key_scheme(make_client):
    schemes = make_client(api_keys="k").get("/openapi.json").json()["components"]["securitySchemes"]
    assert any(s.get("name") == "X-API-Key" for s in schemes.values())


def test_key_comparison_helper():
    assert is_valid_api_key("abc", ["abc"]) and not is_valid_api_key("abd", ["abc"])
    assert not is_valid_api_key(None, ["abc"]) and not is_valid_api_key("", ["abc"])
    assert is_valid_api_key("klüssel", ["klüssel"])


# ------------------------------------------------------------- rate limiting
def test_rate_limit_returns_429_with_headers(make_client):
    client = make_client(rate_limit_requests=3, rate_limit_window_seconds=60)
    statuses = [client.get("/results").status_code for _ in range(5)]
    assert statuses == [200, 200, 200, 429, 429]
    blocked = client.get("/results")
    assert blocked.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(blocked.headers["Retry-After"]) >= 1
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limit_headers_on_success_and_exempt_health(make_client):
    client = make_client(rate_limit_requests=2)
    response = client.get("/results")
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"
    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_rate_limit_is_per_api_key(make_client):
    client = make_client(rate_limit_requests=1)
    assert client.get("/results", headers={"X-API-Key": "a"}).status_code == 200
    assert client.get("/results", headers={"X-API-Key": "a"}).status_code == 429
    assert client.get("/results", headers={"X-API-Key": "b"}).status_code == 200


def test_rate_limit_can_be_disabled(make_client):
    client = make_client(rate_limit_enabled=False, rate_limit_requests=1)
    assert all(client.get("/results").status_code == 200 for _ in range(5))


def test_sliding_window_expires():
    limiter = SlidingWindowLimiter(2, 10)
    assert limiter.check("k", now=0)[0] and limiter.check("k", now=1)[0]
    allowed, remaining, reset = limiter.check("k", now=2)
    assert not allowed and remaining == 0 and reset == pytest.approx(8)
    assert limiter.check("k", now=11)[0]


# --------------------------------------------------- CORS / headers / config
def test_security_headers_present(make_client):
    headers = make_client().get("/results").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in headers


def test_hsts_in_production(make_client):
    client = make_client(app_env="production", cors_origins="https://app.example.com")
    assert "max-age" in client.get("/health").headers["strict-transport-security"]


def test_cors_restricted_to_configured_origins(make_client):
    client = make_client(cors_origins="https://app.example.com")
    allowed = client.get("/health", headers={"Origin": "https://app.example.com"})
    denied = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-origin" not in denied.headers


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        Settings(_env_file=None, app_env="production", cors_origins="*")
    assert Settings(_env_file=None, app_env="production", cors_origins="https://a.example").is_production


def test_invalid_environment_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_env="staging-ish")


def test_docs_can_be_disabled(make_client):
    client = make_client(enable_docs=False)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_key_list_parsing():
    assert Settings(_env_file=None, api_keys=" a , b,,c ").api_key_list == ["a", "b", "c"]
    assert Settings(_env_file=None).api_key_list == []


# --------------------------------------------------------------- sanitising
def test_sanitize_removes_control_and_invisible_characters():
    assert sanitize_text("he\x00llo\x07 wor\u200bld\ufeff") == "hello world"
    assert sanitize_text("a\u202eb") == "ab"
    assert sanitize_text("line1\r\nline2\tend") == "line1\nline2\tend"


def test_sanitize_preserves_hindi_joiners_and_normalises():
    text = "क्\u200dष"  # contains a zero-width joiner, meaningful in Devanagari
    assert sanitize_text(text) == text
    assert sanitize_text("e\u0301") == "\u00e9"


def test_api_sanitises_and_rejects_control_only_input(make_client):
    client = make_client()
    body = client.post("/evaluate", json={**PAYLOAD, "response": "Gravity\x00 is a force.\u202e"}).json()
    assert body["response"] == "Gravity is a force."
    assert client.post("/evaluate", json={**PAYLOAD, "response": "\x00\u200b\ufeff"}).status_code == 422


def test_overlong_category_query_rejected(make_client):
    assert make_client().get("/results", params={"category": "x" * 200}).status_code == 422
