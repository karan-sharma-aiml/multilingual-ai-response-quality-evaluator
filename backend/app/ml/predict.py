"""Predict the quality class of a response from its metric scores.

Usage (from ``backend/``)::

    python -m app.ml.predict --relevance 90 --correctness 85 --clarity 80 \
        --safety 100 --hallucination 0 --duplicate 0

Only load model files you trained yourself: joblib files can execute code.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from app.utils.paths import MODEL_PATH

ALIASES: dict[str, str] = {
    "relevance": "relevance_score",
    "correctness": "correctness_score",
    "clarity": "clarity_score",
    "safety": "safety_score",
    "hallucination": "hallucination_score",
    "duplicate": "duplicate_percentage",
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model exists yet."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class QualityPredictor:
    """Wraps the persisted classifier."""

    def __init__(self, bundle: dict) -> None:
        self._model = bundle["model"]
        self._features: list[str] = list(bundle["feature_names"])
        self.metadata = {k: v for k, v in bundle.items() if k != "model"}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "QualityPredictor":
        if not path.is_file():
            raise ModelNotFoundError(
                f"No trained model at {path}. Run: python -m app.ml.train_model"
            )
        return cls(joblib.load(path))

    def _to_frame(self, rows: list[Mapping[str, float]]) -> pd.DataFrame:
        normalized = [{ALIASES.get(k, k): v for k, v in row.items()} for row in rows]
        frame = pd.DataFrame(normalized)
        missing = [name for name in self._features if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature(s): {', '.join(missing)}")
        frame = frame[self._features].astype(float)
        if ((frame < 0) | (frame > 100)).any().any():
            raise ValueError("All features must be between 0 and 100.")
        return frame

    def predict_many(self, rows: list[Mapping[str, float]]) -> list[Prediction]:
        frame = self._to_frame(rows)
        probabilities = self._model.predict_proba(frame)
        classes = list(self._model.classes_)
        predictions = []
        for row in probabilities:
            best = int(row.argmax())
            predictions.append(
                Prediction(
                    label=str(classes[best]),
                    confidence=round(float(row[best]), 4),
                    probabilities={str(c): round(float(p), 4) for c, p in zip(classes, row)},
                )
            )
        return predictions

    def predict(self, features: Mapping[str, float]) -> Prediction:
        return self.predict_many([features])[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict response quality class.")
    for alias in ALIASES:
        parser.add_argument(f"--{alias}", type=float)
    parser.add_argument("--json", help="JSON object with the six features")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)

    if args.json:
        features = json.loads(args.json)
    else:
        features = {alias: getattr(args, alias) for alias in ALIASES}
        if any(value is None for value in features.values()):
            parser.error("provide all of " + ", ".join(f"--{a}" for a in ALIASES) + " or --json")
    try:
        prediction = QualityPredictor.load(args.model_path).predict(features)
    except (ModelNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"label": prediction.label, "confidence": prediction.confidence,
                      "probabilities": prediction.probabilities}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
