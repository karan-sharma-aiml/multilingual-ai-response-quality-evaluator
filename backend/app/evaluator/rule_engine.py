"""Rule engine producing human-readable warnings for structural problems."""

import re
from dataclasses import dataclass
from enum import Enum

from app.schemas.evaluation import EvaluationInput
from app.utils.lexicons import SPAM_PHRASES
from app.utils.phrase_matcher import PhraseMatcher
from app.utils.scoring import safe_divide
from app.utils.text import (
    TERMINAL_PUNCTUATION,
    content_tokens,
    extract_urls,
    has_cased_letters,
    is_word_char,
    split_sentences,
    tokenize,
)

_REPEATED_PUNCTUATION = re.compile(r"(?:([!?,;:*\-_=~#])\1{2,}|\.{4,})")
_CLOSING_CHARS = "\"'”’)]"
_INTERNAL_PUNCTUATION = ",;:" + TERMINAL_PUNCTUATION


class RuleSeverity(str, Enum):
    """Severity of a rule violation."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RuleCode(str, Enum):
    """Stable identifiers of the rules implemented by the engine."""

    EMPTY_RESPONSE = "empty_response"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    SPAM = "spam"
    REPEATED_PUNCTUATION = "repeated_punctuation"
    REPEATED_WORDS = "repeated_words"
    MISSING_PUNCTUATION = "missing_punctuation"
    CAPITALIZATION_ISSUE = "capitalization_issue"
    SPECIAL_CHARACTER_SPAM = "special_character_spam"


@dataclass(frozen=True)
class RuleWarning:
    """A single rule violation."""

    code: RuleCode
    message: str
    severity: RuleSeverity = RuleSeverity.WARNING

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code.value}: {self.message}"


@dataclass(frozen=True)
class RuleEngineConfig:
    """Thresholds used by the rule engine."""

    min_words: int = 5
    max_words: int = 1200
    max_urls: int = 2
    repeated_word_run: int = 3
    repeated_word_ratio: float = 0.35
    repeated_word_min_tokens: int = 8
    no_punctuation_min_words: int = 15
    shouting_ratio: float = 0.60
    shouting_min_letters: int = 10
    lowercase_sentence_ratio: float = 0.50
    lowercase_sentence_min_count: int = 2
    special_char_ratio: float = 0.10
    special_char_run: int = 5
    allowed_symbols: str = ".,;:!?'\"()-–—।॥…%/@&+$€₹"


class RuleEngine:
    """Applies deterministic quality rules to a response."""

    def __init__(self, config: RuleEngineConfig | None = None) -> None:
        self._config = config or RuleEngineConfig()
        self._spam_matcher = PhraseMatcher(SPAM_PHRASES)

    def evaluate(self, sample: EvaluationInput) -> list[RuleWarning]:
        """Return all rule violations found in ``sample.response``."""
        return self.evaluate_text(sample.response, sample.language_code)

    def evaluate_text(self, text: str, language: str = "und") -> list[RuleWarning]:
        stripped = text.strip()
        if not stripped:
            return [
                RuleWarning(
                    RuleCode.EMPTY_RESPONSE,
                    "The response is empty.",
                    RuleSeverity.CRITICAL,
                )
            ]

        tokens = tokenize(stripped)
        warnings: list[RuleWarning] = []
        warnings.extend(self._length_rules(len(tokens)))
        warnings.extend(self._spam_rules(stripped, tokens))
        warnings.extend(self._punctuation_rules(stripped, len(tokens)))
        warnings.extend(self._repeated_word_rules(tokens, language))
        warnings.extend(self._capitalization_rules(stripped))
        warnings.extend(self._special_character_rules(stripped))
        return warnings

    @staticmethod
    def messages(warnings: list[RuleWarning]) -> list[str]:
        """Render warnings as plain strings."""
        return [str(warning) for warning in warnings]

    def _length_rules(self, word_total: int) -> list[RuleWarning]:
        cfg = self._config
        if word_total < cfg.min_words:
            return [
                RuleWarning(
                    RuleCode.TOO_SHORT,
                    f"Response has {word_total} words; at least {cfg.min_words} expected.",
                )
            ]
        if word_total > cfg.max_words:
            return [
                RuleWarning(
                    RuleCode.TOO_LONG,
                    f"Response has {word_total} words; the limit is {cfg.max_words}.",
                )
            ]
        return []

    def _spam_rules(self, text: str, tokens: list[str]) -> list[RuleWarning]:
        cfg = self._config
        phrases = self._spam_matcher.find(tokens)
        url_total = len(extract_urls(text))
        if phrases:
            return [
                RuleWarning(
                    RuleCode.SPAM,
                    f"Promotional or spam phrases detected: {', '.join(sorted(set(phrases)))}.",
                    RuleSeverity.CRITICAL,
                )
            ]
        if url_total > cfg.max_urls:
            return [
                RuleWarning(
                    RuleCode.SPAM,
                    f"Response contains {url_total} URLs (maximum {cfg.max_urls}).",
                    RuleSeverity.CRITICAL,
                )
            ]
        return []

    def _punctuation_rules(self, text: str, word_total: int) -> list[RuleWarning]:
        cfg = self._config
        warnings: list[RuleWarning] = []
        if _REPEATED_PUNCTUATION.search(text):
            warnings.append(
                RuleWarning(
                    RuleCode.REPEATED_PUNCTUATION,
                    "Excessively repeated punctuation marks found.",
                )
            )

        trimmed = text.rstrip(_CLOSING_CHARS)
        lacks_terminal = not trimmed or trimmed[-1] not in TERMINAL_PUNCTUATION
        lacks_any = word_total >= cfg.no_punctuation_min_words and not any(
            char in _INTERNAL_PUNCTUATION for char in text
        )
        if lacks_any:
            warnings.append(
                RuleWarning(
                    RuleCode.MISSING_PUNCTUATION,
                    "A long response without any sentence punctuation.",
                )
            )
        elif lacks_terminal:
            warnings.append(
                RuleWarning(
                    RuleCode.MISSING_PUNCTUATION,
                    "The response does not end with terminal punctuation.",
                    RuleSeverity.INFO,
                )
            )
        return warnings

    def _repeated_word_rules(self, tokens: list[str], language: str) -> list[RuleWarning]:
        cfg = self._config
        run_length = 1
        longest_run = 1
        repeated_token = ""
        for previous, current in zip(tokens, tokens[1:]):
            run_length = run_length + 1 if previous == current else 1
            if run_length > longest_run:
                longest_run, repeated_token = run_length, current
        if longest_run >= cfg.repeated_word_run:
            return [
                RuleWarning(
                    RuleCode.REPEATED_WORDS,
                    f"The word '{repeated_token}' is repeated {longest_run} times in a row.",
                )
            ]

        content = content_tokens(tokens, language)
        if len(content) >= cfg.repeated_word_min_tokens:
            counts: dict[str, int] = {}
            for token in content:
                counts[token] = counts.get(token, 0) + 1
            word, highest = max(counts.items(), key=lambda item: item[1])
            if safe_divide(highest, len(content)) > cfg.repeated_word_ratio:
                return [
                    RuleWarning(
                        RuleCode.REPEATED_WORDS,
                        f"The word '{word}' dominates the response ({highest} occurrences).",
                    )
                ]
        return []

    def _capitalization_rules(self, text: str) -> list[RuleWarning]:
        cfg = self._config
        if not has_cased_letters(text):
            return []

        letters = [char for char in text if char.isalpha() and (char.isupper() or char.islower())]
        first_letter = next((char for char in text if char.isalpha()), "")
        if first_letter.islower():
            return [
                RuleWarning(
                    RuleCode.CAPITALIZATION_ISSUE,
                    "The response does not start with a capital letter.",
                    RuleSeverity.INFO,
                )
            ]

        uppercase_ratio = safe_divide(sum(1 for c in letters if c.isupper()), len(letters))
        if len(letters) >= cfg.shouting_min_letters and uppercase_ratio > cfg.shouting_ratio:
            return [
                RuleWarning(
                    RuleCode.CAPITALIZATION_ISSUE,
                    "Excessive use of capital letters.",
                )
            ]

        sentences = [s for s in split_sentences(text) if s[:1].isalpha()]
        lowercase_starts = sum(1 for s in sentences if s[0].islower())
        if (
            len(sentences) >= cfg.lowercase_sentence_min_count
            and safe_divide(lowercase_starts, len(sentences)) >= cfg.lowercase_sentence_ratio
        ):
            return [
                RuleWarning(
                    RuleCode.CAPITALIZATION_ISSUE,
                    "Many sentences start with a lowercase letter.",
                    RuleSeverity.INFO,
                )
            ]
        return []

    def _special_character_rules(self, text: str) -> list[RuleWarning]:
        cfg = self._config
        symbols = 0
        run = 0
        longest_run = 0
        visible = 0
        for char in text:
            if char.isspace():
                run = 0
                continue
            visible += 1
            is_symbol = not is_word_char(char) and char not in cfg.allowed_symbols
            if is_symbol:
                symbols += 1
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0
        if longest_run >= cfg.special_char_run or safe_divide(symbols, visible) > cfg.special_char_ratio:
            return [
                RuleWarning(
                    RuleCode.SPECIAL_CHARACTER_SPAM,
                    "Response contains an unusual amount of special characters.",
                    RuleSeverity.CRITICAL,
                )
            ]
        return []
