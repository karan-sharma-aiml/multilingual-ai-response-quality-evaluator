"""Mapping of hallucination detector findings to named hallucination types."""

from app.evaluator.hallucination import HallucinationConfig, HallucinationReport

HALLUCINATION_TYPE_DESCRIPTIONS: dict[str, str] = {
    "fake_urls": "Fabricated or unverifiable URLs",
    "unsupported_statistics": "Statistics not backed by the question or reference",
    "unsupported_claims": "Authority claims without a source (e.g. 'studies show')",
    "contradictions": "Contradictory statements within the response",
    "made_up_entities": "Invented people, titles or institutions",
    "ungrounded_content": "Content largely absent from the question and reference",
}


def extract_hallucination_types(report: HallucinationReport) -> dict[str, int]:
    """Return ``{type: occurrences}`` for every type found in ``report``."""
    tolerance = HallucinationConfig().ungrounded_tolerance
    counts = {
        "fake_urls": len(report.fake_urls),
        "unsupported_statistics": len(report.unsupported_statistics),
        "unsupported_claims": len(report.unsupported_claims),
        "contradictions": len(report.contradictions),
        "made_up_entities": len(report.made_up_entities),
        "ungrounded_content": 1 if report.ungrounded_ratio > tolerance else 0,
    }
    return {name: count for name, count in counts.items() if count > 0}
