"""Tests for CSV / Excel / PDF report generation."""

import io

import pandas as pd
from openpyxl import load_workbook

from app.utils.sanitize import escape_spreadsheet_formula


def test_csv_report(seeded_client):
    response = seeded_client.get("/reports/csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="evaluation_report_' in response.headers["content-disposition"]
    frame = pd.read_csv(io.StringIO(response.content.decode("utf-8-sig")))
    assert len(frame) == 6
    for column in ("Question", "Language", "Relevance", "Overall Score", "Timestamp (UTC)",
                   "Gemini Relevance", "Gemini Safety", "Method"):
        assert column in frame.columns
    assert set(frame["Language"]) == {"English", "French", "German"}


def test_csv_includes_gemini_scores(client, fake_gemini, evaluate_payload):
    client.post("/evaluate", json=evaluate_payload)
    frame = pd.read_csv(io.StringIO(client.get("/reports/csv").content.decode("utf-8-sig")))
    assert frame.loc[0, "Gemini Relevance"] == 90
    assert frame.loc[0, "Gemini Reasoning"] == "Mocked reasoning."
    assert frame.loc[0, "Method"] == "hybrid"


def test_csv_filters_and_limit(seeded_client):
    french = pd.read_csv(io.StringIO(seeded_client.get("/reports/csv", params={"language": "fr"}).content.decode("utf-8-sig")))
    assert len(french) == 1
    limited = pd.read_csv(io.StringIO(seeded_client.get("/reports/csv", params={"limit": 2}).content.decode("utf-8-sig")))
    assert len(limited) == 2
    assert seeded_client.get("/reports/csv", params={"language": "Klingon"}).status_code == 422


def test_csv_neutralises_formula_injection(client, evaluate_payload):
    client.post("/evaluate", json={**evaluate_payload, "use_gemini": False, "response": "=HYPERLINK(\"http://evil\") is a formula"})
    text = client.get("/reports/csv").content.decode("utf-8-sig")
    frame = pd.read_csv(io.StringIO(text))
    assert frame.loc[0, "Response"].startswith("'=")


def test_escape_helper():
    assert escape_spreadsheet_formula("=1+1") == "'=1+1"
    assert escape_spreadsheet_formula("@SUM(A1)") == "'@SUM(A1)"
    assert escape_spreadsheet_formula("-5 degrees") == "'-5 degrees"
    assert escape_spreadsheet_formula("normal text") == "normal text"


def test_excel_report(seeded_client):
    response = seeded_client.get("/reports/excel")
    assert response.status_code == 200
    assert "spreadsheetml" in response.headers["content-type"]
    workbook = load_workbook(io.BytesIO(response.content))
    assert workbook.sheetnames == ["Evaluations", "Summary"]
    sheet = workbook["Evaluations"]
    assert sheet.max_row == 7
    assert sheet["A1"].value == "ID" and sheet.freeze_panes == "A2"
    summary = {row[0].value: row[1].value for row in workbook["Summary"].iter_rows(min_row=2)}
    assert summary["Total evaluations"] == 6


def test_pdf_report(seeded_client):
    response = seeded_client.get("/reports/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 1500


def test_pdf_handles_hindi_text_and_empty_data(client, evaluate_payload):
    assert client.get("/reports/pdf").content.startswith(b"%PDF")  # empty report
    client.post(
        "/evaluate",
        json={"question": "पानी का रासायनिक सूत्र क्या है?", "language": "Hindi",
              "response": "पानी का रासायनिक सूत्र H2O है।", "use_gemini": False},
    )
    assert client.get("/reports/pdf").content.startswith(b"%PDF")
    excel = load_workbook(io.BytesIO(client.get("/reports/excel").content))
    assert "पानी" in excel["Evaluations"]["E2"].value


def test_report_filters_by_date(seeded_client):
    empty = seeded_client.get("/reports/csv", params={"start_date": "2999-01-01"}).content.decode("utf-8-sig")
    assert len(pd.read_csv(io.StringIO(empty))) == 0
