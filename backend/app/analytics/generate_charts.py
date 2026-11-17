"""CLI: generate all charts into ``reports/charts/``.

Usage (from ``backend/``)::

    python -m app.analytics.generate_charts                 # sample dataset
    python -m app.analytics.generate_charts --source db
    python -m app.analytics.generate_charts --source csv --csv ../reports/sample_results.csv
"""

import argparse
import sys
from pathlib import Path

from app.analytics.charts import ChartGenerator
from app.analytics.frames import load_frame_from_csv, load_frame_from_dataset, load_frame_from_db
from app.core.logging import get_logger
from app.utils.paths import CHARTS_DIR

logger = get_logger("analytics.charts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate analytics charts.")
    parser.add_argument("--source", choices=("dataset", "csv", "db"), default="dataset")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=CHARTS_DIR)
    args = parser.parse_args(argv)

    if args.source == "csv":
        if args.csv is None:
            parser.error("--csv is required with --source csv")
        frame = load_frame_from_csv(args.csv)
    elif args.source == "db":
        from app.database.session import create_db_engine, create_session_factory

        with create_session_factory(create_db_engine())() as session:
            frame = load_frame_from_db(session)
    else:
        frame = load_frame_from_dataset()

    try:
        paths = ChartGenerator(args.output).generate_all(frame)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"{name}: {path}")
    logger.info("Generated %d charts in %s", len(paths), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
