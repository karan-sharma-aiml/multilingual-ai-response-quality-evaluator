"""Tests for CSV batch evaluation (service and endpoint)."""

import io

import pytest

from app.core.exceptions import BatchFileError
from app.evaluator import ResponseEvaluator
from app.services.batch_service import BatchEvaluationService
from app.services.database_service import DatabaseService
from app.services.evaluation_service import EvaluationService

HEADER = "id,language,question,response,reference_answer,category\n"
GOOD_ROW = (
    '1,English,What is cloud computing?,"Cloud computing delivers servers and storage over the '
    'internet. Companies pay only for what they use.","Cloud computing delivers services over the internet.",Technology\n'
)
BAD_LANGUAGE_ROW = "2,Klingon,What is cloud computing?,Some answer here for you.,,Technology\n"
EMPTY_RESPONSE_ROW = "3,French,Qu'est-ce que le cloud ?,,,Technology\n"
FRENCH_ROW = (
    "4,French,Pourquoi le ciel est-il bleu ?,La lumière bleue est diffusée par l'atmosphère.,,Science\n"
)


def csv_bytes(*rows: str, header: str = HEADER) -> bytes:
    return (header + "".join(rows)).encode("utf-8")


@pytest.fixture
def batch_service(db_session, fake_gemini, settings):
    database = DatabaseService(db_session)
    evaluation = EvaluationService(ResponseEvaluator(), fake_gemini, database, settings)
    return BatchEvaluationService(evaluation, database, settings)


def test_batch_service_reports_success_and_failures(batch_service):
    summary = batch_service.process_csv(
        csv_bytes(GOOD_ROW, BAD_LANGUAGE_ROW, EMPTY_RESPONSE_ROW, FRENCH_ROW)
    )
    assert summary.total_records == 4
    assert summary.success_count == 2
    assert summary.failed_count == 2
    assert summary.average_score is not None and 0 < summary.average_score <= 100
    assert len(summary.result_ids) == 2
    failed = {item.row_number: item for item in summary.failed_records}
    assert set(failed) == {3, 4}
    assert failed[3].record_id == "2"
    assert "Unsupported language" in failed[3].error
    assert "response" in failed[4].error


def test_batch_results_are_stored_and_summary_logged(batch_service, db_session):
    summary = batch_service.process_csv(csv_bytes(GOOD_ROW, FRENCH_ROW))
    service = DatabaseService(db_session)
    for result_id in summary.result_ids:
        assert service.get_result(result_id).evaluation_method == "rule_based"
    assert service.logs.list_by_event("batch_completed")


def test_batch_with_gemini_uses_hybrid(batch_service, fake_gemini):
    summary = batch_service.process_csv(csv_bytes(GOOD_ROW), use_gemini=True)
    assert summary.used_gemini and fake_gemini.calls == 1


def test_batch_gemini_failure_does_not_fail_rows(batch_service, fake_gemini):
    from app.core.exceptions import GeminiTimeoutError

    fake_gemini.error = GeminiTimeoutError("timeout")
    summary = batch_service.process_csv(csv_bytes(GOOD_ROW, FRENCH_ROW), use_gemini=True)
    assert summary.success_count == 2 and summary.failed_count == 0


def test_batch_all_rows_failing_gives_null_average(batch_service):
    summary = batch_service.process_csv(csv_bytes(BAD_LANGUAGE_ROW))
    assert summary.success_count == 0 and summary.average_score is None


def test_batch_minimal_columns_and_bom(batch_service):
    content = ("\ufeffQuestion,LANGUAGE,Response\n"
               "What is gravity?,en,Gravity is the force that attracts objects with mass.\n").encode("utf-8")
    assert batch_service.process_csv(content).success_count == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"question,language\nq,en\n", "Missing required column"),
        (b"question,language,response\n", "no data rows"),
        ("question,language,response\nq,en,\xe9\n".encode("latin-1"), "UTF-8"),
    ],
)
def test_batch_file_errors(batch_service, content, message):
    with pytest.raises(BatchFileError, match=message):
        batch_service.process_csv(content)


def test_batch_row_limit(batch_service):
    rows = "".join("What is gravity?,en,Gravity attracts masses together.\n" for _ in range(21))
    with pytest.raises(BatchFileError, match="maximum"):
        batch_service.process_csv(("question,language,response\n" + rows).encode())


# ---------------------------------------------------------------- endpoint
def upload(client, content: bytes, filename="data.csv", **form):
    return client.post(
        "/batch-evaluate",
        files={"file": (filename, io.BytesIO(content), "text/csv")},
        data=form,
    )


def test_batch_endpoint(client):
    response = upload(client, csv_bytes(GOOD_ROW, BAD_LANGUAGE_ROW, FRENCH_ROW))
    body = response.json()
    assert response.status_code == 200
    assert body["success_count"] == 2
    assert body["failed_count"] == 1
    assert body["failed_records"][0]["row_number"] == 3
    assert body["average_score"] > 0
    assert client.get("/results").json()["total"] == 2


def test_batch_endpoint_with_gemini_flag(client, fake_gemini):
    body = upload(client, csv_bytes(GOOD_ROW), use_gemini="true").json()
    assert body["used_gemini"] is True and fake_gemini.calls == 1


def test_batch_endpoint_rejects_non_csv(client):
    response = upload(client, b"hello", filename="data.txt")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_batch_file"


def test_batch_endpoint_rejects_bad_structure(client):
    response = upload(client, b"foo,bar\n1,2\n")
    assert response.status_code == 400
    assert "Missing required column" in response.json()["error"]["message"]


def test_batch_endpoint_requires_file(client):
    assert client.post("/batch-evaluate").status_code == 422


def test_batch_endpoint_enforces_size_limit(client, settings):
    settings.max_upload_bytes = 50
    response = upload(client, csv_bytes(GOOD_ROW))
    assert response.status_code == 413
