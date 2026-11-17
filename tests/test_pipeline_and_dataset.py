"""Integration tests: pipeline, dataset, database metadata and settings."""

import pandas as pd
import pytest
from sqlalchemy import inspect

from app.core.config import Settings
from app.database import create_db_engine, create_tables
from app.evaluator import ResponseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.dataset import REQUIRED_COLUMNS, load_dataset, row_to_input
from app.utils.paths import SAMPLE_DATASET_PATH


def test_pipeline_returns_complete_result(make_sample):
    result = ResponseEvaluator().evaluate(make_sample())
    assert 0 <= result.overall_score <= 100
    assert result.hallucination_score < 20
    assert result.safety_score == 100


def test_pipeline_ranks_good_above_unsafe_and_empty(make_sample):
    evaluator = ResponseEvaluator()
    good = evaluator.evaluate(make_sample()).overall_score
    unsafe = evaluator.evaluate(
        make_sample(response="You are a stupid idiot, kill them and make a bomb at home now.")
    ).overall_score
    empty = evaluator.evaluate(make_sample(response="")).overall_score
    assert good > unsafe > empty or (good > unsafe and empty == 0.0)


def test_sample_dataset_shape_and_coverage():
    frame = load_dataset(SAMPLE_DATASET_PATH)
    assert len(frame) >= 100
    assert tuple(frame.columns) == REQUIRED_COLUMNS
    assert set(frame["language"]) == {"English", "Hindi", "French", "Spanish", "German"}
    assert set(frame["category"]) == {
        "Technology", "History", "Science", "Business", "Health", "Education", "Travel",
    }


def test_every_dataset_row_is_evaluable():
    frame = load_dataset(SAMPLE_DATASET_PATH).head(15)
    evaluator = ResponseEvaluator()
    for _, row in frame.iterrows():
        assert 0 <= evaluator.evaluate(row_to_input(row)).overall_score <= 100


def test_load_dataset_validates_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_empty_reference_becomes_none():
    sample = EvaluationInput(question="q?", language="en", response="r", reference_answer="  ")
    assert sample.reference_answer is None


def test_settings_validate_log_level():
    assert Settings(log_level="warning").log_level == "WARNING"
    with pytest.raises(ValueError):
        Settings(log_level="LOUD")


def test_init_db_creates_table():
    engine = create_db_engine("sqlite:///:memory:", echo=False)
    create_tables(engine)
    assert "evaluation_records" in inspect(engine).get_table_names()
