"""Relevance evaluator based on TF-IDF vectors and cosine similarity."""

from dataclasses import dataclass, field

from app.evaluator.base import BaseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.scoring import clamp, ratio_to_score
from app.utils.similarity import SimilarityConfig, tfidf_similarities
from app.utils.text import tokenize


@dataclass(frozen=True)
class RelevanceConfig:
    """Tunable parameters of the relevance evaluator."""

    question_saturation: float = 0.30
    reference_saturation: float = 0.55
    question_weight: float = 0.35
    reference_weight: float = 0.65
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)


class RelevanceEvaluator(BaseEvaluator):
    """Scores how on-topic a response is for its question and reference."""

    name = "relevance"

    def __init__(self, config: RelevanceConfig | None = None) -> None:
        self._config = config or RelevanceConfig()

    def evaluate(self, sample: EvaluationInput) -> float:
        if not tokenize(sample.response):
            return 0.0

        cfg = self._config
        texts = [sample.question]
        if sample.reference_answer:
            texts.append(sample.reference_answer)

        similarities = tfidf_similarities(
            sample.response, texts, sample.language_code, cfg.similarity
        )
        question_score = clamp(similarities[0] / cfg.question_saturation)
        if not sample.reference_answer:
            return ratio_to_score(question_score)

        reference_score = clamp(similarities[1] / cfg.reference_saturation)
        total_weight = cfg.question_weight + cfg.reference_weight
        blended = (
            cfg.question_weight * question_score
            + cfg.reference_weight * reference_score
        ) / total_weight
        return ratio_to_score(blended)
