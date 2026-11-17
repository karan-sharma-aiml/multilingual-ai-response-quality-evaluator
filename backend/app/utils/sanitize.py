"""Input sanitisation helpers."""

import re
import unicodedata

# Characters that can hide or reorder text (bidi controls, zero-width space, BOM).
# U+200C / U+200D (ZWNJ / ZWJ) are deliberately kept: Hindi needs them.
_INVISIBLE_CHARS = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_LINE_BREAKS = re.compile(r"\r\n?")
_FORMULA_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")


def sanitize_text(value: str) -> str:
    """Normalise Unicode and drop control / invisible characters.

    Newlines and tabs are preserved so multi-line answers stay intact.
    """
    normalized = unicodedata.normalize("NFC", value)
    normalized = _LINE_BREAKS.sub("\n", normalized)
    normalized = _CONTROL_CHARS.sub("", normalized)
    return _INVISIBLE_CHARS.sub("", normalized)


def escape_spreadsheet_formula(value: str) -> str:
    """Neutralise CSV/Excel formula injection by prefixing risky cells."""
    if value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value
