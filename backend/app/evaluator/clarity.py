"""Clarity evaluator: sentence length, repetition, readability, grammar, diversity."""

import re
from dataclasses import dataclass, field

from app.evaluator.base import BaseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.scoring import (
    clamp,
    linear_band_score,
    ratio_to_score,
    safe_divide,
    weighted_average,
)
from app.utils.text import (
    TERMINAL_PUNCTUATION,
    has_cased_letters,
    ngrams,
    split_sentences,
    tokenize,
)

_DOUBLE_SPACE = re.compile(r"[^\S\n]{2,}")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s[,.;:!?]")
_LOWERCASE_PRONOUN_I = re.compile(r"(?<![\w'’])i(?![\w'’])")
_CLOSING_CHARS = "\"'”’)]"


@dataclass(frozen=True)
class ClarityConfig:
    """Tunable parameters of the clarity evaluator."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "sentence_length": 0.20,
            "repetition": 0.20,
            "readability": 0.20,
            "grammar": 0.20,
            "diversity": 0.20,
        }
    )
    ideal_sentence_words: tuple[int, int] = (8, 25)
    hard_sentence_words: tuple[int, int] = (1, 60)
    repetition_ngram_size: int = 3
    repetition_scale: float = 2.0
    default_lix_ideal_max: float = 45.0
    lix_ideal_max: dict[str, float] = field(
        default_factory=lambda: {"de": 55.0, "hi": 55.0}
    )
    lix_hard_max: float = 85.0
    long_word_length: dict[str, int] = field(
        default_factory=lambda: {"en": 6, "fr": 7, "es": 7, "de": 9, "hi": 8}
    )
    default_long_word_length: int = 7
    diversity_window: int = 50
    diversity_step: int = 10
    diversity_target: float = 0.60
    full_confidence_words: int = 8
    min_confidence_factor: float = 0.5
    spaced_punctuation_languages: frozenset[str] = frozenset({"fr"})


class ClarityEvaluator(BaseEvaluator):
    """Scores how clear and well-formed a response is."""

    name = "clarity"

    def __init__(self, config: ClarityConfig | None = None) -> None:
        self._config = config or ClarityConfig()

    def evaluate(self, sample: EvaluationInput) -> float:
        text = sample.response
        tokens = tokenize(text)
        if not tokens:
            return 0.0

        cfg = self._config
        language = sample.language_code
        sentences = split_sentences(text)
        components = {
            "sentence_length": self._sentence_length_score(tokens, sentences),
            "repetition": self._repetition_score(tokens),
            "readability": self._readability_score(tokens, sentences, language),
            "grammar": self._grammar_score(text, sentences, language),
            "diversity": self._diversity_score(tokens),
        }
        quality = weighted_average(components, cfg.weights)

        confidence = clamp(len(tokens) / cfg.full_confidence_words)
        factor = cfg.min_confidence_factor + (1 - cfg.min_confidence_factor) * confidence
        return ratio_to_score(quality * factor)

    def _sentence_length_score(self, tokens: list[str], sentences: list[str]) -> float:
        cfg = self._config
        average = safe_divide(len(tokens), max(len(sentences), 1))
        return linear_band_score(
            average,
            cfg.ideal_sentence_words[0],
            cfg.ideal_sentence_words[1],
            cfg.hard_sentence_words[0],
            cfg.hard_sentence_words[1],
        )

    def _repetition_score(self, tokens: list[str]) -> float:
        cfg = self._config
        grams = ngrams(tokens, cfg.repetition_ngram_size)
        if not grams:
            return 1.0
        repeated_ratio = (len(grams) - len(set(grams))) / len(grams)
        return clamp(1.0 - repeated_ratio * cfg.repetition_scale)

    def _readability_score(
        self, tokens: list[str], sentences: list[str], language: str
    ) -> float:
        """LIX readability index (language independent) mapped to 0-1."""
        cfg = self._config
        threshold = cfg.long_word_length.get(language, cfg.default_long_word_length)
        long_words = sum(1 for token in tokens if len(token) > threshold)
        words_per_sentence = safe_divide(len(tokens), max(len(sentences), 1))
        lix = words_per_sentence + 100.0 * safe_divide(long_words, len(tokens))
        ideal_max = cfg.lix_ideal_max.get(language, cfg.default_lix_ideal_max)
        return linear_band_score(lix, 0.0, ideal_max, -1.0, cfg.lix_hard_max)

    def _diversity_score(self, tokens: list[str]) -> float:
        cfg = self._config
        window = cfg.diversity_window
        if len(tokens) <= window:
            ratio = len(set(tokens)) / len(tokens)
        else:
            ratios = [
                len(set(tokens[start : start + window])) / window
                for start in range(0, len(tokens) - window + 1, cfg.diversity_step)
            ]
            ratio = sum(ratios) / len(ratios)
        return clamp(ratio / cfg.diversity_target)

    def _grammar_score(self, text: str, sentences: list[str], language: str) -> float:
        """Fraction of applicable grammar heuristics that pass."""
        cfg = self._config
        stripped = text.strip()
        checks: list[bool] = []
        cased = has_cased_letters(stripped)

        if cased:
            first_letter = next((c for c in stripped if c.isalpha()), "")
            checks.append(not first_letter.islower())
            later = [s for s in sentences[1:] if s and s[0].isalpha()]
            checks.append(not any(s[0].islower() for s in later))

        trimmed = stripped.rstrip(_CLOSING_CHARS)
        checks.append(bool(trimmed) and trimmed[-1] in TERMINAL_PUNCTUATION)
        checks.append(_DOUBLE_SPACE.search(text) is None)
        if language not in cfg.spaced_punctuation_languages:
            checks.append(_SPACE_BEFORE_PUNCTUATION.search(text) is None)

        tokens = tokenize(text)
        checks.append(all(a != b for a, b in zip(tokens, tokens[1:])))
        checks.append(text.count("(") == text.count(")"))
        checks.append(text.count('"') % 2 == 0)
        if language == "en":
            checks.append(_LOWERCASE_PRONOUN_I.search(text) is None)

        return safe_divide(sum(checks), len(checks))
