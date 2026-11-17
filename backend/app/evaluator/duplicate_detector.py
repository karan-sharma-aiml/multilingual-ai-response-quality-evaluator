"""Duplicate detector: repeated words, repeated sentences and looping output."""

from dataclasses import dataclass

from app.evaluator.base import BaseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.scoring import clamp, ratio_to_score, safe_divide
from app.utils.text import (
    content_tokens,
    ngrams,
    normalize_sentence,
    split_sentences,
    tokenize,
)


@dataclass(frozen=True)
class DuplicateConfig:
    """Tunable parameters of the duplicate detector."""

    sentence_weight: float = 0.35
    word_weight: float = 0.15
    ngram_weight: float = 0.30
    loop_weight: float = 0.20
    ngram_size: int = 4
    natural_word_repeat_baseline: float = 0.25
    min_content_tokens: int = 5
    min_loop_tokens: int = 8
    max_loop_period: int = 200
    loop_match_threshold: float = 0.80
    dominant_signal_weight: float = 0.90


@dataclass(frozen=True)
class DuplicateReport:
    """Detailed outcome of a duplicate analysis (all ratios in 0-1)."""

    percentage: float
    repeated_word_ratio: float
    repeated_sentence_ratio: float
    repeated_ngram_ratio: float
    loop_ratio: float

    @property
    def is_looping(self) -> bool:
        return self.loop_ratio > 0.0


class DuplicateDetector(BaseEvaluator):
    """Returns the duplicate percentage (0 = unique, 100 = fully repeated)."""

    name = "duplicate"

    def __init__(self, config: DuplicateConfig | None = None) -> None:
        self._config = config or DuplicateConfig()

    def analyze(self, text: str, language: str = "und") -> DuplicateReport:
        cfg = self._config
        tokens = tokenize(text)
        if not tokens:
            return DuplicateReport(0.0, 0.0, 0.0, 0.0, 0.0)

        word_ratio = self._repeated_word_ratio(tokens, language)
        sentence_ratio = self._repeated_sentence_ratio(text)
        ngram_ratio = self._repeated_ngram_ratio(tokens)
        loop_ratio = self._loop_ratio(tokens)

        total_weight = (
            cfg.sentence_weight + cfg.word_weight + cfg.ngram_weight + cfg.loop_weight
        )
        combined = (
            cfg.sentence_weight * sentence_ratio
            + cfg.word_weight * word_ratio
            + cfg.ngram_weight * ngram_ratio
            + cfg.loop_weight * loop_ratio
        ) / total_weight
        # A strong looping / n-gram signal on its own must not be diluted by
        # the other (naturally low) components.
        combined = max(
            combined, cfg.dominant_signal_weight * max(ngram_ratio, loop_ratio)
        )
        return DuplicateReport(
            percentage=ratio_to_score(combined),
            repeated_word_ratio=word_ratio,
            repeated_sentence_ratio=sentence_ratio,
            repeated_ngram_ratio=ngram_ratio,
            loop_ratio=loop_ratio,
        )

    def evaluate(self, sample: EvaluationInput) -> float:
        return self.analyze(sample.response, sample.language_code).percentage

    def _repeated_word_ratio(self, tokens: list[str], language: str) -> float:
        cfg = self._config
        consecutive = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        consecutive_ratio = safe_divide(consecutive, len(tokens))

        content = content_tokens(tokens, language)
        if len(content) < cfg.min_content_tokens:
            return clamp(consecutive_ratio)
        repeat_ratio = 1.0 - len(set(content)) / len(content)
        baseline = cfg.natural_word_repeat_baseline
        excess = clamp(safe_divide(repeat_ratio - baseline, 1.0 - baseline))
        return max(excess, clamp(consecutive_ratio))

    @staticmethod
    def _repeated_sentence_ratio(text: str) -> float:
        sentences = [
            normalized
            for normalized in (normalize_sentence(s) for s in split_sentences(text))
            if normalized
        ]
        if len(sentences) < 2:
            return 0.0
        return (len(sentences) - len(set(sentences))) / len(sentences)

    def _repeated_ngram_ratio(self, tokens: list[str]) -> float:
        grams = ngrams(tokens, self._config.ngram_size)
        if not grams:
            return 0.0
        return (len(grams) - len(set(grams))) / len(grams)

    def _loop_ratio(self, tokens: list[str]) -> float:
        """Share of tokens that merely replay the sequence one period earlier."""
        cfg = self._config
        total = len(tokens)
        if total < cfg.min_loop_tokens:
            return 0.0

        best_ratio = 0.0
        for period in range(1, min(total // 2, cfg.max_loop_period) + 1):
            comparable = total - period
            matches = sum(
                1 for index in range(period, total) if tokens[index] == tokens[index - period]
            )
            if safe_divide(matches, comparable) >= cfg.loop_match_threshold:
                best_ratio = max(best_ratio, matches / total)
        return best_ratio
