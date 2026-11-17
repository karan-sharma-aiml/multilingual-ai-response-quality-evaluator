"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.evaluation import EvaluationInput  # noqa: E402

GOOD_RESPONSE = (
    "Cloud computing means renting computing resources like servers, storage, "
    "databases, and software over the internet instead of owning physical hardware. "
    "Companies pay only for what they use."
)
REFERENCE = (
    "Cloud computing delivers computing services such as servers, storage, "
    "databases, and software over the internet on a pay-as-you-go basis."
)
QUESTION = "What is cloud computing?"


@pytest.fixture
def make_sample():
    """Factory building ``EvaluationInput`` objects with sensible defaults."""

    def _make(
        response: str = GOOD_RESPONSE,
        question: str = QUESTION,
        language: str = "English",
        reference_answer: str | None = REFERENCE,
    ) -> EvaluationInput:
        return EvaluationInput(
            question=question,
            language=language,
            response=response,
            reference_answer=reference_answer,
        )

    return _make


# --------------------------------------------------------------------------
# Part 2 fixtures: in-memory database, fake Gemini and API test client
# --------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402

from app.api.dependencies import get_db, get_gemini_service  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.exceptions import GeminiError  # noqa: E402
from app.database.init_db import initialize_database  # noqa: E402
from app.database.session import create_db_engine, create_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.gemini_service import GeminiEvaluation  # noqa: E402


class FakeGemini:
    """Stand-in for ``GeminiEvaluationService`` used in tests."""

    def __init__(self, relevance=90.0, correctness=80.0, clarity=70.0, safety=100.0):
        self.scores = dict(
            relevance=relevance, correctness=correctness, clarity=clarity, safety=safety
        )
        self.error: GeminiError | None = None
        self.calls = 0
        self.is_configured = True
        self.model = "fake-gemini"

    def evaluate(self, sample):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return GeminiEvaluation(
            **self.scores,
            reasoning="Mocked reasoning.",
            latency_ms=12.5,
            model=self.model,
            attempts=1,
        )

    def close(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_weight=0.6, max_batch_rows=20, _env_file=None)


@pytest.fixture
def db_engine():
    engine = create_db_engine("sqlite:///:memory:", echo=False)
    initialize_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    session = create_session_factory(db_engine)()
    yield session
    session.close()


@pytest.fixture
def fake_gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def client(db_engine, fake_gemini, settings):
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


EVALUATE_PAYLOAD = {
    "question": QUESTION,
    "response": GOOD_RESPONSE,
    "language": "English",
    "reference_answer": REFERENCE,
}


@pytest.fixture
def evaluate_payload() -> dict:
    return dict(EVALUATE_PAYLOAD)


# --------------------------------------------------------------------------
# Part 3 fixtures: cache reset and a client pre-loaded with varied evaluations
# --------------------------------------------------------------------------
from app.core.cache import get_analytics_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_analytics_cache():
    get_analytics_cache().clear()
    yield
    get_analytics_cache().clear()


SEED_CASES = [
    # (language, category, response, use_gemini)
    ("English", "Technology", GOOD_RESPONSE, False),
    ("English", "Technology", "Bananas are yellow fruit that monkeys enjoy.", False),
    ("French", "Science", "Le ciel est bleu car l'atmosphère diffuse la lumière bleue du Soleil.", False),
    ("German", "Travel", "Ja.", False),
    (
        "English",
        "Technology",
        "Cloud computing is great. According to a 2023 study, 87.4% of experts agree. "
        "Dr. Marcus Velloran proved it. See https://www.verified-facts-1.example.net/study/1.",
        False,
    ),
    ("English", None, "You stupid idiot, kill them and make a bomb at home.", False),
]


@pytest.fixture
def seeded_client(client):
    """Client with six stored evaluations (good, off-topic, short, hallucinated, unsafe)."""
    for language, category, response, use_gemini in SEED_CASES:
        payload = {**EVALUATE_PAYLOAD, "language": language, "response": response, "use_gemini": use_gemini}
        if category:
            payload["category"] = category
        assert client.post("/evaluate", json=payload).status_code == 201
    return client
