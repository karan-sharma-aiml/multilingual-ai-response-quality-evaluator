"""Tests for the ML pipeline and charts."""

import numpy as np
import pandas as pd
import pytest

from app.analytics.charts import CHART_FILES, ChartGenerator
from app.ml.config import FEATURE_NAMES, QualityThresholds
from app.ml.labels import add_labels, label_from_score
from app.ml.predict import ModelNotFoundError, QualityPredictor
from app.ml.predict import main as predict_main
from app.ml.train_model import main as train_main
from app.ml.train_model import save_model, save_reports, train


def synthetic_frame(rows: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(0, 100, size=(rows, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    frame["overall_score"] = (
        0.25 * frame["relevance_score"] + 0.35 * frame["correctness_score"] + 0.15 * frame["clarity_score"]
        + 0.1 * frame["safety_score"] + 0.15 * (100 - frame["hallucination_score"])
        - 0.5 * frame["duplicate_percentage"].clip(lower=10)
    ).clip(0, 100)
    frame["language"] = rng.choice(["English", "French", "Hindi"], size=rows)
    frame["category"] = rng.choice(["Technology", "Travel"], size=rows)
    return frame


@pytest.mark.parametrize(
    ("score", "label"),
    [(100, "Excellent"), (85, "Excellent"), (84.99, "Good"), (70, "Good"), (69.9, "Average"), (50, "Average"), (49.9, "Poor"), (0, "Poor")],
)
def test_label_thresholds(score, label):
    assert label_from_score(score) == label


def test_custom_thresholds_and_label_column():
    frame = add_labels(pd.DataFrame({"overall_score": [10, 90]}), QualityThresholds(average_min=5, good_min=20, excellent_min=60))
    assert list(frame["quality_label"]) == ["Average", "Excellent"]


def test_training_produces_metrics_and_artifacts(tmp_path):
    result = train(synthetic_frame())
    assert result.metrics["accuracy"] > 0.7
    for key in ("precision_weighted", "recall_weighted", "f1_weighted", "cv_accuracy_mean"):
        assert 0 <= result.metrics[key] <= 1
    assert result.confusion.shape == (len(result.labels), len(result.labels))
    assert "precision" in result.report_text
    assert set(result.feature_importances) == set(FEATURE_NAMES)

    model_path = save_model(result, QualityThresholds(), tmp_path / "model.joblib")
    reports = save_reports(result, tmp_path / "reports")
    assert model_path.is_file()
    assert all(path.is_file() for path in reports.values())


def test_predictor_roundtrip(tmp_path):
    result = train(synthetic_frame())
    path = save_model(result, QualityThresholds(), tmp_path / "model.joblib")
    predictor = QualityPredictor.load(path)
    high = predictor.predict(dict(relevance=95, correctness=95, clarity=95, safety=100, hallucination=0, duplicate=0))
    low = predictor.predict(dict(relevance=5, correctness=5, clarity=10, safety=40, hallucination=90, duplicate=80))
    order = ["Poor", "Average", "Good", "Excellent"]
    assert order.index(high.label) > order.index(low.label)
    assert low.label == "Poor"
    assert sum(high.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert predictor.predict_many([{"relevance_score": 50, "correctness_score": 50, "clarity_score": 50,
                                    "safety_score": 50, "hallucination_score": 50, "duplicate_percentage": 50}])


def test_predictor_validation_and_missing_model(tmp_path):
    with pytest.raises(ModelNotFoundError):
        QualityPredictor.load(tmp_path / "missing.joblib")
    path = save_model(train(synthetic_frame()), QualityThresholds(), tmp_path / "m.joblib")
    predictor = QualityPredictor.load(path)
    with pytest.raises(ValueError, match="Missing feature"):
        predictor.predict({"relevance": 1})
    with pytest.raises(ValueError, match="between 0 and 100"):
        predictor.predict(dict(relevance=500, correctness=1, clarity=1, safety=1, hallucination=1, duplicate=1))


def test_training_rejects_bad_data():
    with pytest.raises(ValueError, match="At least"):
        train(synthetic_frame(rows=5))
    constant = synthetic_frame()
    constant["overall_score"] = 10.0
    with pytest.raises(ValueError, match="two quality classes"):
        train(constant)


def test_train_and_predict_cli(tmp_path, capsys):
    csv_path = tmp_path / "scores.csv"
    synthetic_frame().to_csv(csv_path, index=False)
    model = tmp_path / "cli.joblib"
    assert train_main(["--source", "csv", "--csv", str(csv_path), "--model-path", str(model),
                       "--reports-dir", str(tmp_path / "r")]) == 0
    assert "Classification report" in capsys.readouterr().out
    argv = ["--model-path", str(model), "--json",
            '{"relevance":90,"correctness":90,"clarity":90,"safety":100,"hallucination":0,"duplicate":0}']
    assert predict_main(argv) == 0
    assert '"label"' in capsys.readouterr().out
    assert predict_main(["--model-path", str(tmp_path / "none.joblib"), "--json", "{}"]) == 1


def test_charts_are_generated(tmp_path):
    paths = ChartGenerator(tmp_path / "charts").generate_all(synthetic_frame(60))
    assert set(paths) == set(CHART_FILES)
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 5_000


def test_charts_reject_empty_frame(tmp_path):
    with pytest.raises(ValueError):
        ChartGenerator(tmp_path).generate_all(synthetic_frame(0))
