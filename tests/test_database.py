"""Tests for the ORM models, repositories and DatabaseService."""

import pytest
from sqlalchemy import inspect

from app.core.exceptions import DatabaseOperationError, ResultNotFoundError
from app.database.init_db import initialize_database, seed_languages
from app.evaluator import ResponseEvaluator
from app.repositories import (
    EvaluationLogRepository,
    EvaluationResultRepository,
    LanguageRepository,
    ResultFilters,
)
from app.schemas.api import EvaluationMethod, GeminiInfo, SortField, SortOrder
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationOutcome


@pytest.fixture
def outcome_factory(make_sample):
    evaluator = ResponseEvaluator()

    def build(sample=None, method=EvaluationMethod.RULE_BASED, gemini=None):
        sample = sample or make_sample()
        return sample, EvaluationOutcome(
            result=evaluator.evaluate(sample),
            method=method,
            gemini=gemini or GeminiInfo(used=False),
            rule_scores={},
        )

    return build


def test_all_tables_exist(db_engine):
    tables = set(inspect(db_engine).get_table_names())
    assert {"languages", "responses", "evaluation_results", "evaluation_logs"} <= tables


def test_languages_are_seeded_once(db_session):
    repository = LanguageRepository(db_session)
    assert repository.count() == 5
    assert seed_languages(db_session) == 0
    assert repository.get_by_code("hi").name == "Hindi"


def test_initialize_database_is_idempotent(db_engine):
    initialize_database(db_engine)
    initialize_database(db_engine)


def test_save_evaluation_persists_all_tables(db_session, outcome_factory):
    sample, outcome = outcome_factory()
    service = DatabaseService(db_session)
    record = service.save_evaluation(sample, outcome, category="Technology")

    assert record.id is not None
    assert record.response_record.question == sample.question
    assert record.response_record.language.code == "en"
    assert record.response_record.category == "Technology"
    assert record.evaluation_method == "rule_based"
    assert record.created_at is not None
    logs = EvaluationLogRepository(db_session).list_for_result(record.id)
    assert [log.event for log in logs] == ["evaluation_saved"]
    assert logs[0].latency_ms is not None


def test_gemini_events_are_logged(db_session, outcome_factory):
    service = DatabaseService(db_session)
    sample, used = outcome_factory(
        method=EvaluationMethod.HYBRID, gemini=GeminiInfo(used=True, model="m", latency_ms=42.0)
    )
    record = service.save_evaluation(sample, used)
    events = {log.event: log for log in service.logs.list_for_result(record.id)}
    assert events["gemini_call"].latency_ms == 42.0

    sample, failed = outcome_factory(gemini=GeminiInfo(used=False, error="boom"))
    record = service.save_evaluation(sample, failed)
    events = {log.event for log in service.logs.list_for_result(record.id)}
    assert "gemini_fallback" in events


def test_list_results_filters_and_sorting(db_session, outcome_factory, make_sample):
    service = DatabaseService(db_session)
    service.save_evaluation(*outcome_factory(make_sample(language="English")), category="A")
    service.save_evaluation(
        *outcome_factory(make_sample(response="Bananas are yellow.", language="French")), category="B"
    )
    repo = EvaluationResultRepository(db_session)

    items, total = repo.search(ResultFilters(language_code="fr"))
    assert total == 1 and items[0].response_record.language.code == "fr"
    items, total = repo.search(ResultFilters(category="a"))
    assert total == 1
    items, total = repo.search(ResultFilters(), SortField.OVERALL_SCORE, SortOrder.ASC)
    assert total == 2 and items[0].overall_score <= items[1].overall_score
    items, total = repo.search(ResultFilters(), offset=1, limit=1)
    assert total == 2 and len(items) == 1
    page = service.list_results(ResultFilters(), 1, 1, SortField.ID, SortOrder.ASC)
    assert page.total_pages == 2


def test_delete_removes_rows_but_keeps_logs(db_session, outcome_factory):
    service = DatabaseService(db_session)
    record = service.save_evaluation(*outcome_factory())
    result_id = record.id
    service.delete_result(result_id)

    with pytest.raises(ResultNotFoundError):
        service.get_result(result_id)
    repository = EvaluationResultRepository(db_session)
    assert repository.count() == 0
    kept = service.logs.list_by_event("evaluation_saved")
    assert kept and kept[0].evaluation_result_id is None
    assert service.logs.list_by_event("result_deleted")


def test_get_missing_result_raises(db_session):
    with pytest.raises(ResultNotFoundError):
        DatabaseService(db_session).get_result(123)


def test_database_errors_are_wrapped(db_session, outcome_factory):
    from sqlalchemy import text

    sample, outcome = outcome_factory()
    db_session.execute(text("DROP TABLE evaluation_logs"))
    with pytest.raises(DatabaseOperationError):
        DatabaseService(db_session).save_evaluation(sample, outcome)
