"""Pandas frames of evaluation scores shared by charts and the ML pipeline."""

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluator import ResponseEvaluator
from app.models import EvaluationResultRecord
from app.utils.dataset import load_dataset, row_to_input
from app.utils.paths import SAMPLE_DATASET_PATH

SCORE_COLUMNS: tuple[str, ...] = (
    "relevance_score",
    "correctness_score",
    "clarity_score",
    "safety_score",
    "hallucination_score",
    "duplicate_percentage",
    "overall_score",
)
FRAME_COLUMNS: tuple[str, ...] = ("id", "language", "category", *SCORE_COLUMNS)


def load_frame_from_dataset(path: Path = SAMPLE_DATASET_PATH) -> pd.DataFrame:
    """Evaluate every row of a dataset CSV with the rule-based engine."""
    evaluator = ResponseEvaluator()
    records: list[dict[str, object]] = []
    for _, row in load_dataset(path).iterrows():
        result = evaluator.evaluate(row_to_input(row))
        records.append(
            {
                "id": row["id"],
                "language": row["language"],
                "category": row["category"] or "Uncategorized",
                **{column: getattr(result, column) for column in SCORE_COLUMNS},
            }
        )
    return pd.DataFrame.from_records(records, columns=list(FRAME_COLUMNS))


def load_frame_from_csv(path: Path) -> pd.DataFrame:
    """Load a results CSV (e.g. ``reports/sample_results.csv``)."""
    frame = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing score columns: {', '.join(missing)}")
    if "language" not in frame.columns:
        frame["language"] = "Unknown"
    if "category" not in frame.columns:
        frame["category"] = "Uncategorized"
    return frame


def load_frame_from_db(session: Session) -> pd.DataFrame:
    """Load stored evaluations from the database."""
    statement = select(EvaluationResultRecord)
    records = list(session.scalars(statement).unique())
    rows = [
        {
            "id": record.id,
            "language": record.response_record.language.name,
            "category": record.response_record.category or "Uncategorized",
            **{column: getattr(record, column) for column in SCORE_COLUMNS},
        }
        for record in records
    ]
    return pd.DataFrame.from_records(rows, columns=list(FRAME_COLUMNS))
