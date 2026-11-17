"""Train the Random Forest quality classifier.

Labels are derived from the rule-based / hybrid overall score (weak
supervision), so the model learns to reproduce the scoring policy from the six
metric values. Retrain on human-labelled data to obtain an independent judge.

Usage (from ``backend/``)::

    python -m app.ml.train_model                      # sample dataset
    python -m app.ml.train_model --source csv --csv reports/sample_results.csv
    python -m app.ml.train_model --source db
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from app.analytics.frames import load_frame_from_csv, load_frame_from_dataset, load_frame_from_db  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.ml.config import (  # noqa: E402
    FEATURE_NAMES,
    QUALITY_CLASSES,
    TARGET_COLUMN,
    QualityThresholds,
    TrainingConfig,
)
from app.ml.labels import add_labels  # noqa: E402
from app.utils.paths import ML_REPORTS_DIR, MODEL_PATH  # noqa: E402

logger = get_logger("ml.train")
MIN_SAMPLES: int = 20


@dataclass
class TrainingResult:
    """Everything produced by a training run."""

    model: RandomForestClassifier
    metrics: dict[str, Any]
    report_text: str
    confusion: np.ndarray
    labels: list[str]
    feature_importances: dict[str, float]
    model_path: Path | None = None


def train(
    frame: pd.DataFrame,
    thresholds: QualityThresholds | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train and evaluate a classifier on a frame of metric scores."""
    settings = config or TrainingConfig()
    labelled = add_labels(frame, thresholds)
    if len(labelled) < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} samples are required (got {len(labelled)}).")

    features = labelled[list(FEATURE_NAMES)].astype(float)
    target = labelled[TARGET_COLUMN].astype(str)
    counts = target.value_counts()
    if len(counts) < 2:
        raise ValueError("Training data must contain at least two quality classes.")

    stratify = target if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=settings.test_size,
        random_state=settings.random_state, stratify=stratify,
    )
    model = RandomForestClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        min_samples_leaf=settings.min_samples_leaf,
        class_weight="balanced",
        random_state=settings.random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    present = [name for name in QUALITY_CLASSES if name in set(target)]
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_weighted": round(
            float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "class_distribution": {name: int(counts.get(name, 0)) for name in present},
    }
    folds = min(settings.cv_folds, int(counts.min()))
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=settings.random_state)
        scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=settings.n_estimators, class_weight="balanced",
                random_state=settings.random_state, n_jobs=1,
            ),
            features, target, cv=cv, scoring="accuracy",
        )
        metrics["cv_accuracy_mean"] = round(float(scores.mean()), 4)
        metrics["cv_accuracy_std"] = round(float(scores.std()), 4)
        metrics["cv_folds"] = folds

    matrix = confusion_matrix(y_test, predictions, labels=present)
    report_text = classification_report(y_test, predictions, labels=present, zero_division=0)
    metrics["classification_report"] = classification_report(
        y_test, predictions, labels=present, zero_division=0, output_dict=True
    )
    importances = {
        name: round(float(value), 4) for name, value in zip(FEATURE_NAMES, model.feature_importances_)
    }
    return TrainingResult(model, metrics, report_text, matrix, present, importances)


def save_model(result: TrainingResult, thresholds: QualityThresholds, path: Path = MODEL_PATH) -> Path:
    """Persist the model and its metadata with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_names": list(FEATURE_NAMES),
            "classes": list(result.model.classes_),
            "thresholds": asdict(thresholds),
            "metrics": {k: v for k, v in result.metrics.items() if k != "classification_report"},
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
        },
        path,
    )
    result.model_path = path
    return path


def save_reports(result: TrainingResult, directory: Path = ML_REPORTS_DIR) -> dict[str, Path]:
    """Write metrics JSON, the classification report and a confusion-matrix plot."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": directory / "metrics.json",
        "report": directory / "classification_report.txt",
        "confusion_matrix": directory / "confusion_matrix.png",
    }
    payload = {**result.metrics, "feature_importances": result.feature_importances,
               "confusion_matrix": {"labels": result.labels, "matrix": result.confusion.tolist()}}
    paths["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["report"].write_text(result.report_text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(result.confusion, cmap="Blues")
    ax.set_xticks(range(len(result.labels)), result.labels, rotation=30, ha="right")
    ax.set_yticks(range(len(result.labels)), result.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Quality classifier - confusion matrix")
    threshold = result.confusion.max() / 2 if result.confusion.size else 0
    for (row, col), value in np.ndenumerate(result.confusion):
        ax.text(col, row, int(value), ha="center", va="center",
                color="white" if value > threshold else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(paths["confusion_matrix"], dpi=150)
    plt.close(fig)
    return paths


def load_training_frame(source: str, csv_path: Path | None) -> pd.DataFrame:
    """Load metric scores from the sample dataset, a CSV file or the database."""
    if source == "csv":
        if csv_path is None:
            raise ValueError("--csv is required when --source csv")
        return load_frame_from_csv(csv_path)
    if source == "db":
        from app.database.session import create_db_engine, create_session_factory

        with create_session_factory(create_db_engine())() as session:
            return load_frame_from_db(session)
    return load_frame_from_dataset()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the quality classifier.")
    parser.add_argument("--source", choices=("dataset", "csv", "db"), default="dataset")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=ML_REPORTS_DIR)
    args = parser.parse_args(argv)

    thresholds = QualityThresholds()
    frame = load_training_frame(args.source, args.csv)
    result = train(frame, thresholds)
    model_path = save_model(result, thresholds, args.model_path)
    save_reports(result, args.reports_dir)

    print(f"Samples: {len(frame)}  classes: {result.metrics['class_distribution']}")
    for key in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"):
        print(f"{key:>20}: {result.metrics[key]:.4f}")
    if "cv_accuracy_mean" in result.metrics:
        mean, std = result.metrics["cv_accuracy_mean"], result.metrics["cv_accuracy_std"]
        print(f"{'cv_accuracy':>20}: {mean:.4f} +/- {std:.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):", result.labels)
    print(result.confusion)
    print("\nClassification report:\n" + result.report_text)
    print(f"Model saved to {model_path}")
    logger.info("Model trained: accuracy=%.4f path=%s", result.metrics["accuracy"], model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
