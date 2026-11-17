"""Tests for the rule engine."""

import pytest

from app.evaluator.rule_engine import RuleCode, RuleEngine, RuleSeverity


def codes(text: str, language: str = "en") -> set[RuleCode]:
    return {warning.code for warning in RuleEngine().evaluate_text(text, language)}


def test_clean_response_has_no_warnings():
    text = "Cloud computing delivers servers and storage over the internet. Companies pay only for usage."
    assert codes(text) == set()


def test_empty_response():
    warnings = RuleEngine().evaluate_text("   ")
    assert [w.code for w in warnings] == [RuleCode.EMPTY_RESPONSE]
    assert warnings[0].severity is RuleSeverity.CRITICAL


def test_too_short():
    assert RuleCode.TOO_SHORT in codes("Maybe so.")


def test_too_long():
    text = ("Variety keeps text interesting. " * 320).strip()
    assert RuleCode.TOO_LONG in codes(text)


def test_spam_phrase():
    assert RuleCode.SPAM in codes("Buy now and click here for a limited offer on cloud storage deals.")


def test_spam_urls():
    text = "See http://a.com and http://b.com and http://c.com for more information on this subject."
    assert RuleCode.SPAM in codes(text)


def test_repeated_punctuation():
    assert RuleCode.REPEATED_PUNCTUATION in codes("This is amazing!!! You must read all about it.")


def test_repeated_words():
    assert RuleCode.REPEATED_WORDS in codes("This is very very very good and useful.")


def test_missing_punctuation():
    assert RuleCode.MISSING_PUNCTUATION in codes("cloud computing delivers servers over the internet")


def test_capitalization_all_caps():
    assert RuleCode.CAPITALIZATION_ISSUE in codes("CLOUD COMPUTING DELIVERS SERVERS OVER THE INTERNET NOW.")


def test_capitalization_lowercase_start():
    assert RuleCode.CAPITALIZATION_ISSUE in codes("cloud computing delivers servers over the internet.")


def test_special_character_spam():
    assert RuleCode.SPECIAL_CHARACTER_SPAM in codes("Cloud ★☆★☆★ computing ♥♥♥♥♥ delivers servers ✓✓✓ over the internet.")


@pytest.mark.parametrize(
    "text",
    ["पानी का रासायनिक सूत्र H2O है।", "Le ciel est bleu à cause de la diffusion de la lumière."],
)
def test_no_false_capitalization_or_special_warnings(text):
    found = codes(text, "hi")
    assert RuleCode.CAPITALIZATION_ISSUE not in found
    assert RuleCode.SPECIAL_CHARACTER_SPAM not in found


def test_messages_are_strings():
    warnings = RuleEngine().evaluate_text("")
    assert RuleEngine.messages(warnings)[0].startswith("[CRITICAL]")
