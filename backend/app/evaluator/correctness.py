"""Correctness evaluator: reference similarity or heuristic estimation."""

import re
from dataclasses import dataclass, field

from app.evaluator.base import BaseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.lexicons import ABSOLUTE_CLAIM_TERMS, UNCERTAINTY_PHRASES
from app.utils.phrase_matcher import PhraseMatcher
from app.utils.scoring import clamp, clamp_score, ratio_to_score, safe_divide
from app.utils.similarity import SimilarityConfig, tfidf_similarities
from app.utils.text import (
    content_tokens,
    extract_numbers,
    light_stem,
    split_sentences,
    tokenize,
)

_CAPITALIZED_WORD = re.compile(r"\b[A-ZÀ-ÖØ-Ý][\w'’-]+")


@dataclass(frozen=True)
class CorrectnessConfig:
    """Tunable parameters of the correctness evaluator."""

    similarity_saturation: float = 0.60
    cosine_weight: float = 0.40
    recall_weight: float = 0.40
    number_weight: float = 0.20

    heuristic_floor: float = 20.0
    heuristic_ceiling: float = 85.0
    adequate_word_count: int = 20
    specificity_target: int = 3
    coverage_weight: float = 0.30
    length_weight: float = 0.30
    specificity_weight: float = 0.20
    diversity_weight: float = 0.20
    uncertainty_penalty: float = 8.0
    uncertainty_penalty_cap: float = 24.0
    absolute_claim_penalty: float = 4.0
    absolute_claim_penalty_cap: float = 12.0

    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)


class CorrectnessEvaluator(BaseEvaluator):
    """Estimates factual correctness.

    With a reference answer the score blends TF-IDF similarity, recall of the
    reference's key terms and agreement on numeric facts. Without a reference
    a conservative heuristic estimate (capped below 100) is produced.
    """

    name = "correctness"

    def __init__(self, config: CorrectnessConfig | None = None) -> None:
        self._config = config or CorrectnessConfig()
        self._uncertainty = PhraseMatcher(UNCERTAINTY_PHRASES)
        self._absolutes = PhraseMatcher(ABSOLUTE_CLAIM_TERMS)

    def evaluate(self, sample: EvaluationInput) -> float:
        response_tokens = tokenize(sample.response)
        if not response_tokens:
            return 0.0
        if sample.reference_answer:
            return self._score_with_reference(sample, response_tokens)
        return self._score_heuristically(sample, response_tokens)

    def _score_with_reference(
        self, sample: EvaluationInput, response_tokens: list[str]
    ) -> float:
        cfg = self._config
        language = sample.language_code
        reference = sample.reference_answer or ""

        similarity = tfidf_similarities(
            sample.response, [reference], language, cfg.similarity
        )[0]
        cosine_score = clamp(similarity / cfg.similarity_saturation)

        reference_stems = {
            light_stem(token)
            for token in content_tokens(tokenize(reference), language)
        }
        response_stems = {
            light_stem(token) for token in content_tokens(response_tokens, language)
        }
        recall_score = safe_divide(
            len(reference_stems & response_stems), len(reference_stems)
        )

        reference_numbers = set(extract_numbers(reference))
        weights = {
            "cosine": cfg.cosine_weight,
            "recall": cfg.recall_weight,
        }
        components = {"cosine": cosine_score, "recall": recall_score}
        if reference_numbers:
            response_numbers = set(extract_numbers(sample.response))
            components["numbers"] = safe_divide(
                len(reference_numbers & response_numbers), len(reference_numbers)
            )
            weights["numbers"] = cfg.number_weight

        total_weight = sum(weights.values())
        blended = sum(components[key] * weights[key] for key in components)
        return ratio_to_score(blended / total_weight)

    def _score_heuristically(
        self, sample: EvaluationInput, response_tokens: list[str]
    ) -> float:
        cfg = self._config
        language = sample.language_code
        total_tokens = len(response_tokens)

        question_stems = {
            light_stem(token)
            for token in content_tokens(tokenize(sample.question), language)
        }
        response_stems = {light_stem(token) for token in response_tokens}
        coverage = (
            safe_divide(len(question_stems & response_stems), len(question_stems))
            if question_stems
            else 0.5
        )
        length_adequacy = clamp(total_tokens / cfg.adequate_word_count)
        specificity = clamp(self._count_specifics(sample.response) / cfg.specificity_target)
        diversity = clamp(len(set(response_tokens)) / total_tokens)

        base_ratio = (
            cfg.coverage_weight * coverage
            + cfg.length_weight * length_adequacy
            + cfg.specificity_weight * specificity
            + cfg.diversity_weight * diversity
        )
        score = cfg.heuristic_floor + (
            cfg.heuristic_ceiling - cfg.heuristic_floor
        ) * clamp(base_ratio)

        uncertainty_hits = len(self._uncertainty.find(response_tokens))
        score -= min(
            cfg.uncertainty_penalty_cap, uncertainty_hits * cfg.uncertainty_penalty
        )
        absolute_hits = len(self._absolutes.find(response_tokens))
        score -= min(
            cfg.absolute_claim_penalty_cap,
            absolute_hits * cfg.absolute_claim_penalty,
        )
        return clamp_score(score)

    @staticmethod
    def _count_specifics(text: str) -> int:
        """Count numbers and mid-sentence capitalised words (names, places)."""
        count = len(extract_numbers(text))
        for sentence in split_sentences(text):
            words = sentence.split(maxsplit=1)
            remainder = words[1] if len(words) > 1 else ""
            count += len(_CAPITALIZED_WORD.findall(remainder))
        return count
