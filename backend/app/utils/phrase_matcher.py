"""Token-based multi-word phrase matching."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from app.utils.text import tokenize


class PhraseMatcher:
    """Find known phrases inside an already tokenised text.

    Phrases are tokenised with the same tokenizer as the input, which makes
    matching independent of punctuation, casing and script.
    """

    def __init__(self, phrases: Iterable[str]) -> None:
        by_first_token: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for phrase in phrases:
            phrase_tokens = tuple(tokenize(phrase))
            if phrase_tokens:
                by_first_token[phrase_tokens[0]].append(phrase_tokens)
        self._index: dict[str, list[tuple[str, ...]]] = {
            first: sorted(items, key=len, reverse=True)
            for first, items in by_first_token.items()
        }

    def find(self, tokens: Sequence[str]) -> list[str]:
        """Return every non-overlapping phrase occurrence (longest match first)."""
        matches: list[str] = []
        position = 0
        total = len(tokens)
        while position < total:
            matched_length = 0
            for phrase in self._index.get(tokens[position], ()):
                length = len(phrase)
                if tuple(tokens[position : position + length]) == phrase:
                    matches.append(" ".join(phrase))
                    matched_length = length
                    break
            position += matched_length or 1
        return matches
