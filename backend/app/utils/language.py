"""Language code normalisation helpers."""

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "hi", "fr", "es", "de")
UNKNOWN_LANGUAGE: str = "und"

LANGUAGE_ALIASES: dict[str, str] = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "हिन्दी": "hi",
    "हिंदी": "hi",
    "fr": "fr",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "es": "es",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "de": "de",
    "german": "de",
    "deutsch": "de",
}


def normalize_language(value: str | None) -> str:
    """Map a language name or code to a supported ISO-639-1 code.

    Unknown values map to ``UNKNOWN_LANGUAGE`` so that language-agnostic
    processing is used instead of failing.
    """
    key = (value or "").strip().casefold()
    return LANGUAGE_ALIASES.get(key, UNKNOWN_LANGUAGE)

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}
