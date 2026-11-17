"""Hallucination detector (risk score: 0 = grounded, 100 = highly hallucinated)."""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.evaluator.base import BaseEvaluator
from app.schemas.evaluation import EvaluationInput
from app.utils.lexicons import (
    ENTITY_MARKERS,
    HINDI_TITLE_PATTERN,
    NEGATIONS,
    SUSPICIOUS_TLDS,
    TRUSTED_DOMAINS,
    UNSUPPORTED_CLAIM_PATTERNS,
)
from app.utils.scoring import SCORE_MAX, clamp, clamp_score, safe_divide
from app.utils.text import (
    content_tokens,
    extract_numbers,
    extract_urls,
    light_stem,
    split_sentences,
    tokenize,
)

_PERCENT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)*)\s*(?:%|percent|per\s?cent|prozent|pour\s?cent|por\s?ciento|प्रतिशत)",
    re.IGNORECASE,
)
_GROUPED_NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+\b")
_ENTITY_WORD = r"[A-ZÀ-ÖØ-Ý][\w'’\-]*\.?"
_ENTITY_CONNECTOR = r"(?:of|de|del|der|für|des|du|la|le|y|et|and|und)"
_ENTITY_PATTERN = re.compile(
    rf"{_ENTITY_WORD}(?:\s+(?:{_ENTITY_CONNECTOR}\s+)?{_ENTITY_WORD})*"
)
_HINDI_TITLE = re.compile(HINDI_TITLE_PATTERN)
_CLAIM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in UNSUPPORTED_CLAIM_PATTERNS
)
_ENTITY_SCAN_LANGUAGES: frozenset[str] = frozenset({"en", "fr", "es"})


@dataclass(frozen=True)
class HallucinationConfig:
    """Tunable parameters of the hallucination detector."""

    url_penalty: float = 25.0
    suspicious_url_penalty: float = 35.0
    url_cap: float = 60.0
    statistic_penalty: float = 12.0
    statistic_cap: float = 36.0
    claim_penalty: float = 10.0
    claim_cap: float = 30.0
    contradiction_penalty: float = 25.0
    contradiction_cap: float = 50.0
    marker_entity_penalty: float = 12.0
    marker_entity_cap: float = 36.0
    plain_entity_penalty: float = 3.0
    plain_entity_cap: float = 6.0
    ungrounded_penalty: float = 25.0
    ungrounded_tolerance: float = 0.70
    unverifiable_factor: float = 0.5
    contradiction_overlap: float = 0.60
    contradiction_min_terms: int = 3
    max_contradiction_sentences: int = 60


@dataclass(frozen=True)
class HallucinationReport:
    """Detailed outcome of a hallucination analysis."""

    score: float
    fake_urls: list[str] = field(default_factory=list)
    unsupported_statistics: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    made_up_entities: list[str] = field(default_factory=list)
    ungrounded_ratio: float = 0.0

    @property
    def has_signals(self) -> bool:
        return bool(
            self.fake_urls
            or self.unsupported_statistics
            or self.unsupported_claims
            or self.contradictions
            or self.made_up_entities
        )


class HallucinationDetector(BaseEvaluator):
    """Rule-based detector of typical hallucination signals."""

    name = "hallucination"

    def __init__(self, config: HallucinationConfig | None = None) -> None:
        self._config = config or HallucinationConfig()

    def analyze(self, sample: EvaluationInput) -> HallucinationReport:
        cfg = self._config
        response = sample.response
        if not tokenize(response):
            return HallucinationReport(score=0.0)

        language = sample.language_code
        has_reference = bool(sample.reference_answer)
        support_text = f"{sample.question} {sample.reference_answer or ''}"
        verifiability = 1.0 if has_reference else cfg.unverifiable_factor

        fake_urls, url_penalty = self._check_urls(response, support_text)
        statistics = self._find_unsupported_statistics(response, support_text)
        claims = self._find_unsupported_claims(response)
        contradictions = self._find_contradictions(response, language)
        marker_entities, plain_entities = self._find_made_up_entities(
            response, support_text, language
        )
        ungrounded = (
            self._ungrounded_ratio(response, support_text, language)
            if has_reference
            else 0.0
        )

        penalty = min(cfg.url_cap, url_penalty)
        penalty += min(
            cfg.statistic_cap,
            len(statistics) * cfg.statistic_penalty * verifiability,
        )
        penalty += min(cfg.claim_cap, len(claims) * cfg.claim_penalty)
        penalty += min(
            cfg.contradiction_cap, len(contradictions) * cfg.contradiction_penalty
        )
        penalty += min(
            cfg.marker_entity_cap,
            len(marker_entities) * cfg.marker_entity_penalty * verifiability,
        )
        penalty += min(
            cfg.plain_entity_cap,
            len(plain_entities) * cfg.plain_entity_penalty * verifiability,
        )
        penalty += cfg.ungrounded_penalty * clamp(
            safe_divide(
                ungrounded - cfg.ungrounded_tolerance, 1.0 - cfg.ungrounded_tolerance
            )
        )

        return HallucinationReport(
            score=clamp_score(min(penalty, SCORE_MAX)),
            fake_urls=fake_urls,
            unsupported_statistics=statistics,
            unsupported_claims=claims,
            contradictions=contradictions,
            made_up_entities=marker_entities + plain_entities,
            ungrounded_ratio=round(ungrounded, 4),
        )

    def evaluate(self, sample: EvaluationInput) -> float:
        return self.analyze(sample).score

    # ------------------------------------------------------------------ URLs
    def _check_urls(self, response: str, support_text: str) -> tuple[list[str], float]:
        cfg = self._config
        known_urls = {self._normalize_url(u) for u in extract_urls(support_text)}
        known_domains = {self._domain(u) for u in extract_urls(support_text)}
        fake: list[str] = []
        penalty = 0.0
        for url in extract_urls(response):
            domain = self._domain(url)
            if self._normalize_url(url) in known_urls or domain in known_domains:
                continue
            if self._registrable_domain(domain) in TRUSTED_DOMAINS:
                continue
            fake.append(url)
            penalty += (
                cfg.suspicious_url_penalty
                if self._is_suspicious_domain(domain)
                else cfg.url_penalty
            )
        return fake, penalty

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.lower().rstrip("/")

    @staticmethod
    def _domain(url: str) -> str:
        candidate = url if "://" in url else f"http://{url}"
        host = (urlparse(candidate).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host

    @staticmethod
    def _registrable_domain(domain: str) -> str:
        return ".".join(domain.split(".")[-2:])

    @staticmethod
    def _is_suspicious_domain(domain: str) -> bool:
        labels = domain.split(".")
        has_digits = any(char.isdigit() for char in domain)
        return (
            labels[-1] in SUSPICIOUS_TLDS
            or has_digits
            or domain.count("-") >= 2
        )

    # ------------------------------------------------------------ statistics
    @staticmethod
    def _find_unsupported_statistics(response: str, support_text: str) -> list[str]:
        supported = set(extract_numbers(support_text))
        found: list[str] = []
        for match in _PERCENT_PATTERN.finditer(response):
            number = match.group(1).replace(",", ".")
            if number not in supported:
                found.append(match.group(0).strip())
        for match in _GROUPED_NUMBER_PATTERN.finditer(response):
            number = match.group(0).replace(",", ".")
            if number not in supported:
                found.append(match.group(0))
        return found

    # ---------------------------------------------------------------- claims
    @staticmethod
    def _find_unsupported_claims(response: str) -> list[str]:
        claims: list[str] = []
        for pattern in _CLAIM_PATTERNS:
            claims.extend(match.group(0).strip() for match in pattern.finditer(response))
        return claims

    #---------------------------------------------------------- contradictions
    def _find_contradictions(
        self, response: str, language: str
    ) -> list[tuple[str, str]]:
        cfg = self._config
        negations = NEGATIONS.get(language, frozenset().union(*NEGATIONS.values()))
        sentences = split_sentences(response)[: cfg.max_contradiction_sentences]

        profiles: list[tuple[str, bool, set[str]]] = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            negated = any(token in negations for token in tokens)
            terms = {
                light_stem(token)
                for token in content_tokens(tokens, language)
                if token not in negations
            }
            profiles.append((sentence, negated, terms))

        contradictions: list[tuple[str, str]] = []
        for i, (first, first_negated, first_terms) in enumerate(profiles):
            for second, second_negated, second_terms in profiles[i + 1 :]:
                if first_negated == second_negated:
                    continue
                if min(len(first_terms), len(second_terms)) < cfg.contradiction_min_terms:
                    continue
                union = first_terms | second_terms
                overlap = safe_divide(len(first_terms & second_terms), len(union))
                if overlap >= cfg.contradiction_overlap:
                    contradictions.append((first, second))
        return contradictions

    # -------------------------------------------------------------- entities
    def _find_made_up_entities(
        self, response: str, support_text: str, language: str
    ) -> tuple[list[str], list[str]]:
        support_lower = support_text.casefold()
        support_tokens = set(tokenize(support_text))
        marker_entities: list[str] = []
        plain_entities: list[str] = []

        for sentence in split_sentences(response):
            for candidate in self._entity_candidates(sentence, language):
                words = [word.rstrip(".") for word in candidate.split()]
                if candidate.casefold() in support_lower:
                    continue
                if all(tokenize(word)[0:1] and tokenize(word)[0] in support_tokens for word in words if tokenize(word)):
                    continue
                has_marker = any(word.casefold() in ENTITY_MARKERS for word in words)
                if has_marker or _HINDI_TITLE.fullmatch(candidate):
                    marker_entities.append(candidate)
                elif len(words) >= 2 and language in _ENTITY_SCAN_LANGUAGES:
                    plain_entities.append(candidate)
        return marker_entities, plain_entities

    @staticmethod
    def _entity_candidates(sentence: str, language: str) -> list[str]:
        candidates = [m.group(0).strip().rstrip(".") for m in _HINDI_TITLE.finditer(sentence)]

        first_word, _, remainder = sentence.partition(" ")
        keeps_first_word = first_word.rstrip(".").casefold() in ENTITY_MARKERS
        scan_text = sentence if keeps_first_word else remainder
        for match in _ENTITY_PATTERN.finditer(scan_text):
            candidate = match.group(0).strip().rstrip(".")
            words = candidate.split()
            has_marker = any(w.rstrip(".").casefold() in ENTITY_MARKERS for w in words)
            if has_marker or (len(words) >= 2 and language in _ENTITY_SCAN_LANGUAGES):
                candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------- grounding
    @staticmethod
    def _ungrounded_ratio(response: str, support_text: str, language: str) -> float:
        response_terms = [
            light_stem(t) for t in content_tokens(tokenize(response), language)
        ]
        if not response_terms:
            return 0.0
        support_terms = {
            light_stem(t) for t in content_tokens(tokenize(support_text), language)
        }
        ungrounded = sum(1 for term in response_terms if term not in support_terms)
        return ungrounded / len(response_terms)
