"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd

from app.schemas.evaluation import EvaluationInput

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "language",
    "question",
    "response",
    "reference_answer",
    "category",
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset and verify that all required columns exist."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return frame


def row_to_input(row: pd.Series) -> EvaluationInput:
    """Convert a dataset row into a validated ``EvaluationInput``."""
    return EvaluationInput(
        question=row["question"],
        language=row["language"],
        response=row["response"],
        reference_answer=row["reference_answer"] or None,
    )
