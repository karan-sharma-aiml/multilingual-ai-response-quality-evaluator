"""TF-IDF cosine similarity utilities used by several evaluators."""

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.text import content_tokens, ngrams, tokenize


@dataclass(frozen=True)
class SimilarityConfig:
    """Parameters of the blended word/character TF-IDF similarity."""

    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    word_weight: float = 0.6
    char_weight: float = 0.4


def _word_analyzer(
    language: str, ngram_range: tuple[int, int]
) -> Callable[[str], list[str]]:
    def analyze(document: str) -> list[str]:
        tokens = content_tokens(tokenize(document), language)
        grams: list[str] = []
        for size in range(ngram_range[0], ngram_range[1] + 1):
            grams.extend(" ".join(gram) for gram in ngrams(tokens, size))
        return grams

    return analyze


def _fit_and_compare(
    vectorizer: TfidfVectorizer, documents: list[str]
) -> list[float]:
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [0.0] * (len(documents) - 1)
    similarities = cosine_similarity(matrix[0], matrix[1:]).ravel()
    return [float(value) for value in similarities]


def tfidf_similarities(
    target: str,
    others: list[str],
    language: str,
    config: SimilarityConfig | None = None,
) -> list[float]:
    """Cosine similarity (0-1) between ``target`` and every text in ``others``.

    A word-level TF-IDF (stopwords removed, uni+bi-grams) is blended with a
    character n-gram TF-IDF, which makes the metric robust to inflection and
    to scripts where whitespace tokenisation is weak.
    """
    settings = config or SimilarityConfig()
    if not others:
        return []
    if not tokenize(target):
        return [0.0] * len(others)

    documents = [target, *others]
    word_vectorizer = TfidfVectorizer(
        analyzer=_word_analyzer(language, settings.word_ngram_range)
    )
    word_scores = _fit_and_compare(word_vectorizer, documents)

    char_documents = [" ".join(content_tokens(tokenize(doc), language)) for doc in documents]
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=settings.char_ngram_range,
        lowercase=False,
    )
    char_scores = _fit_and_compare(char_vectorizer, char_documents)

    total_weight = settings.word_weight + settings.char_weight
    return [
        (settings.word_weight * word + settings.char_weight * char) / total_weight
        for word, char in zip(word_scores, char_scores)
    ]
