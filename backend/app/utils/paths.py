"""Filesystem locations shared across the project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
BACKEND_DIR: Path = PROJECT_ROOT / "backend"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs"
ENV_FILE: Path = PROJECT_ROOT / ".env"

SAMPLE_DATASET_PATH: Path = DATASETS_DIR / "sample_dataset.csv"
SAMPLE_RESULTS_PATH: Path = REPORTS_DIR / "sample_results.csv"

ML_ARTIFACTS_DIR: Path = PROJECT_ROOT / "ml_artifacts"
ML_REPORTS_DIR: Path = REPORTS_DIR / "ml"
CHARTS_DIR: Path = REPORTS_DIR / "charts"
MODEL_PATH: Path = ML_ARTIFACTS_DIR / "quality_classifier.joblib"
