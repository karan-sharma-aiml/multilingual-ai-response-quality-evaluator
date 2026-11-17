"""Abstract base class implemented by every score-producing evaluator."""

from abc import ABC, abstractmethod

from app.schemas.evaluation import EvaluationInput


class BaseEvaluator(ABC):
    """Contract for evaluators that map a sample to a 0-100 number."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, sample: EvaluationInput) -> float:
        """Return a score in the range 0-100 for ``sample``."""
