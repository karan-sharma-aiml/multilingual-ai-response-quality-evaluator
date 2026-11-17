"""Unicode-aware text processing helpers (tokenisation, sentences, numbers)."""

import re
import unicodedata
from functools import lru_cache

from app.utils.lexicons import SENTENCE_ABBREVIATIONS, STOPWORDS

WORD_CATEGORY_PREFIXES: tuple[str, ...] = ("L", "M", "N")
MIN_CONTENT_TOKEN_LENGTH: int = 2
LIGHT_STEM_LENGTH: int = 5
TERMINAL_PUNCTUATION: str = ".!?।॥…"
TOKEN_CACHE_SIZE: int = 8192
URL_TRAILING_PUNCTUATION: str = ".,;:!?"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥…])\s+|\n+")
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_ABBREVIATION_END = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(a) for a in SENTENCE_ABBREVIATIONS) + r")\.$",
    re.IGNORECASE,
)


def is_word_char(char: str) -> bool:
    """Return True for letters, combining marks and digits in any script."""
    return unicodedata.category(char).startswith(WORD_CATEGORY_PREFIXES)


@lru_cache(maxsize=TOKEN_CACHE_SIZE)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    buffer: list[str] = []
    for char in text.casefold():
        if is_word_char(char):
            buffer.append(char)
        elif buffer:
            tokens.append("".join(buffer))
            buffer.clear()
    if buffer:
        tokens.append("".join(buffer))
    return tuple(tokens)


def tokenize(text: str) -> list[str]:
    """Split text into lower-cased word tokens (works for Devanagari too)."""
    return list(_tokenize_cached(text))


def word_count(text: str) -> int:
    """Number of word tokens in ``text``."""
    return len(_tokenize_cached(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common abbreviations intact."""
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _ABBREVIATION_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def normalize_sentence(sentence: str) -> str:
    """Canonical form of a sentence used for duplicate comparison."""
    return " ".join(tokenize(sentence))


def content_tokens(tokens: list[str], language: str) -> list[str]:
    """Remove stopwords and one-character tokens."""
    stopwords = STOPWORDS.get(language, frozenset())
    return [
        token
        for token in tokens
        if len(token) >= MIN_CONTENT_TOKEN_LENGTH and token not in stopwords
    ]


def light_stem(token: str) -> str:
    """Language-agnostic prefix stem to tolerate inflection."""
    return token[:LIGHT_STEM_LENGTH]


def ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    """All contiguous n-grams of ``size`` tokens."""
    if size <= 0 or len(tokens) < size:
        return []
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def extract_numbers(text: str) -> list[str]:
    """Extract numeric literals, normalising decimal separators to a dot."""
    numbers: list[str] = []
    for match in _NUMBER_PATTERN.findall(text):
        numbers.append(match.replace(",", ".").rstrip("."))
    return numbers


def extract_urls(text: str) -> list[str]:
    """Extract URLs, stripping trailing sentence punctuation."""
    return [
        match.rstrip(URL_TRAILING_PUNCTUATION) for match in _URL_PATTERN.findall(text)
    ]


def has_cased_letters(text: str) -> bool:
    """True when the text contains letters that have an upper/lower case."""
    return any(char.isupper() or char.islower() for char in text)
