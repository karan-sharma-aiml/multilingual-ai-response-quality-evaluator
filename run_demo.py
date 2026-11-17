"""Demo runner: evaluates the sample dataset and writes a results report.

Usage:
    python run_demo.py
    python run_demo.py --dataset datasets/sample_dataset.csv --output reports/sample_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.evaluator import ResponseEvaluator  # noqa: E402
from app.utils.dataset import load_dataset, row_to_input  # noqa: E402
from app.utils.paths import SAMPLE_DATASET_PATH, SAMPLE_RESULTS_PATH  # noqa: E402

SCORE_COLUMNS: tuple[str, ...] = (
    "relevance_score",
    "correctness_score",
    "clarity_score",
    "safety_score",
    "hallucination_score",
    "duplicate_percentage",
    "overall_score",
)
DISPLAY_COLUMNS: dict[str, str] = {
    "id": "ID",
    "language": "Lang",
    "category": "Category",
    "relevance_score": "Relev",
    "correctness_score": "Correct",
    "clarity_score": "Clarity",
    "safety_score": "Safety",
    "hallucination_score": "Halluc",
    "duplicate_percentage": "Dup%",
    "overall_score": "Overall",
    "warning_count": "Warn",
}
WARNING_SEPARATOR: str = " | "
FLOAT_FORMAT: str = "{:.2f}".format


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the sample dataset.")
    parser.add_argument("--dataset", type=Path, default=SAMPLE_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=SAMPLE_RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    return parser.parse_args(argv)


def evaluate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each row and return the original columns plus all scores."""
    evaluator = ResponseEvaluator()
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        result = evaluator.evaluate(row_to_input(row))
        record = {
            "id": row["id"],
            "language": row["language"],
            "category": row["category"],
            "response_preview": " ".join(row["response"].split())[:80],
            **{column: getattr(result, column) for column in SCORE_COLUMNS},
            "warning_count": len(result.warnings),
            "warnings": WARNING_SEPARATOR.join(result.warnings),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def print_report(results: pd.DataFrame) -> None:
    """Print the formatted per-record table and per-language summary."""
    table = results[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS)
    print("\n" + "=" * 100)
    print("MULTILINGUAL AI RESPONSE QUALITY EVALUATION")
    print("=" * 100)
    print(table.to_string(index=False, float_format=FLOAT_FORMAT, justify="center"))

    summary = results.groupby("language")[list(SCORE_COLUMNS)].mean().round(2)
    summary.columns = [DISPLAY_COLUMNS[column] for column in SCORE_COLUMNS]
    print("\n" + "-" * 100)
    print("AVERAGE SCORES BY LANGUAGE")
    print("-" * 100)
    print(summary.to_string())

    flagged = int((results["warning_count"] > 0).sum())
    print(f"\nRecords evaluated : {len(results)}")
    print(f"Records with warnings : {flagged}")
    print(f"Mean overall score : {results['overall_score'].mean():.2f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    logger = get_logger("run_demo")

    frame = load_dataset(args.dataset)
    if args.limit:
        frame = frame.head(args.limit)
    logger.info("Loaded %d records from %s", len(frame), args.dataset)

    results = evaluate_dataset(frame)
    print_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, encoding="utf-8")
    logger.info("Saved results to %s", args.output)
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
