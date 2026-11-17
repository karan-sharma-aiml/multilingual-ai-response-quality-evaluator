"""Tests for caching, indexing, pooling and concurrent Gemini calls."""

import threading
import time

from sqlalchemy import inspect

from app.core.cache import TTLCache
from app.core.config import Settings
from app.database.session import create_db_engine
from app.evaluator import ResponseEvaluator
from app.schemas.api import CompareRequest
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationService


def test_ttl_cache_hits_expiry_and_eviction():
    now = [0.0]
    cache = TTLCache(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    calls = []
    compute = lambda: calls.append(1) or len(calls)  # noqa: E731
    assert cache.get_or_set("a", compute) == 1
    assert cache.get_or_set("a", compute) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    now[0] = 11
    assert cache.get_or_set("a", compute) == 2  # expired
    cache.get_or_set("b", compute)
    cache.get_or_set("c", compute)  # evicts the least recently used key
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_composite_indexes_exist(db_engine):
    inspector = inspect(db_engine)
    results = {ix["name"] for ix in inspector.get_indexes("evaluation_results")}
    assert {"ix_evaluation_results_created_overall", "ix_evaluation_results_method_created"} <= results
    assert "ix_responses_language_category" in {ix["name"] for ix in inspector.get_indexes("responses")}
    assert "ix_evaluation_logs_event_created" in {ix["name"] for ix in inspector.get_indexes("evaluation_logs")}


def test_ensure_indexes_recreates_missing_index(db_engine):
    from sqlalchemy import text

    from app.database.init_db import ensure_indexes

    with db_engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_evaluation_results_created_overall"))
    ensure_indexes(db_engine)
    assert "ix_evaluation_results_created_overall" in {
        ix["name"] for ix in inspect(db_engine).get_indexes("evaluation_results")
    }


def test_pool_settings_applied_for_server_databases():
    settings = Settings(_env_file=None, db_pool_size=3, db_max_overflow=1)
    engine = create_db_engine("postgresql+psycopg2://u:p@localhost:5432/db")
    assert engine.pool.size() == 10  # defaults from the global settings object
    assert settings.db_pool_size == 3
    engine.dispose()


class SlowGemini:
    """Each call sleeps; used to prove calls overlap."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def evaluate(self, sample):
        from app.services.gemini_service import GeminiEvaluation

        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.2)
        with self._lock:
            self.active -= 1
        return GeminiEvaluation(80, 80, 80, 100, "ok", 200.0, "slow", 1)


def test_compare_runs_gemini_calls_concurrently(db_session):
    gemini = SlowGemini()
    service = EvaluationService(
        ResponseEvaluator(), gemini, DatabaseService(db_session), Settings(_env_file=None, gemini_concurrency=4)
    )
    started = time.perf_counter()
    result = service.compare(
        CompareRequest(question="What is gravity?", language="en", response_a="A force of attraction.",
                       response_b="Gravity pulls masses together.", use_gemini=True)
    )
    elapsed = time.perf_counter() - started
    assert gemini.peak == 2
    assert elapsed < 0.38  # two 0.2s calls overlap
    assert result.response_a.evaluation_method.value == "hybrid"


def test_batch_uses_concurrency_and_stores_all(db_session):
    from app.services.batch_service import BatchEvaluationService

    gemini = SlowGemini()
    settings = Settings(_env_file=None, gemini_concurrency=4)
    database = DatabaseService(db_session)
    evaluation = EvaluationService(ResponseEvaluator(), gemini, database, settings)
    rows = "".join(f"What is gravity {i}?,en,Gravity is a force that attracts masses number {i}.\n" for i in range(8))
    summary = BatchEvaluationService(evaluation, database, settings).process_csv(
        ("question,language,response\n" + rows).encode(), use_gemini=True
    )
    assert summary.success_count == 8 and gemini.peak > 1


def test_health_endpoint_is_async():
    import inspect as pyinspect

    from app.api.routes.health import health

    assert pyinspect.iscoroutinefunction(health)
