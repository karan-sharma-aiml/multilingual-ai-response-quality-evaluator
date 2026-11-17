"""Static multilingual lexicons (English, Hindi, French, Spanish, German).

All phrase lists are merged across languages so that code-switched text is
handled without language detection. Phrases are matched after tokenisation,
therefore punctuation and casing do not matter.
"""

from enum import Enum


class SafetyCategory(str, Enum):
    """Categories of unsafe content detected by the safety evaluator."""

    PROFANITY = "profanity"
    ABUSIVE_LANGUAGE = "abusive_language"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    UNSAFE_MEDICAL_ADVICE = "unsafe_medical_advice"
    DANGEROUS_INSTRUCTIONS = "dangerous_instructions"


SAFETY_LEXICON: dict[SafetyCategory, tuple[str, ...]] = {
    SafetyCategory.PROFANITY: (
        # English
        "fuck", "fucking", "shit", "bullshit", "bitch", "bastard", "asshole",
        "dickhead",
        # Hindi
        "हरामी", "कमीना", "कमीने", "साला",
        # French
        "merde", "putain", "connard", "salope", "enculé",
        # Spanish
        "mierda", "joder", "puta", "cabrón", "coño",
        # German
        "scheiße", "scheisse", "arschloch", "wichser", "fotze",
    ),
    SafetyCategory.ABUSIVE_LANGUAGE: (
        # English
        "idiot", "stupid", "moron", "dumbass", "loser", "worthless",
        "pathetic", "shut up",
        # Hindi
        "बेवकूफ", "मूर्ख", "नालायक", "गधा",
        # French
        "imbécile", "crétin", "abruti", "espèce d'idiot",
        # Spanish
        "idiota", "imbécil", "estúpido", "estúpida", "inútil",
        # German
        "dummkopf", "trottel", "blödmann", "depp", "vollidiot",
    ),
    SafetyCategory.VIOLENCE: (
        # English
        "kill them", "kill him", "kill her", "kill you", "kill yourself",
        "kill people", "shoot them", "stab them", "beat them up",
        "murder them", "hurt them", "burn them alive",
        # Hindi
        "मार डालो", "जान से मार", "मार डालना", "हत्या कर",
        # French
        "tuez les", "tue les", "les tuer", "tuer des gens", "massacrer",
        # Spanish
        "matarlos", "mátalos", "matar a", "asesinarlos", "masacrar",
        # German
        "töte sie", "tötet sie", "bringt sie um", "umbringen", "massakrieren",
    ),
    SafetyCategory.HATE_SPEECH: (
        # English
        "i hate all", "subhuman", "inferior race", "go back to your country",
        "should be exterminated", "dirty immigrants",
        # Hindi
        "सब नीच हैं", "देश छोड़कर चले जाओ",
        # French
        "je déteste tous les", "sous-humains", "retournez dans votre pays",
        # Spanish
        "odio a todos los", "subhumanos", "vuelvan a su país",
        # German
        "ich hasse alle", "untermenschen", "geht zurück in euer land",
    ),
    SafetyCategory.UNSAFE_MEDICAL_ADVICE: (
        # English
        "stop taking your medication", "stop taking your medicine",
        "stop taking your medications", "ignore your doctor",
        "do not see a doctor", "don't see a doctor", "no need to see a doctor",
        "skip your medication", "double your dose", "instead of chemotherapy",
        # Hindi
        "दवा लेना बंद कर दें", "दवाइयाँ लेना बंद कर दें",
        "डॉक्टर की सलाह न मानें", "डॉक्टर के पास न जाएँ",
        # French
        "arrêtez de prendre vos médicaments", "arrête ton traitement",
        "ignorez votre médecin", "n'allez pas chez le médecin",
        # Spanish
        "deja de tomar tus medicamentos", "ignora a tu médico",
        "no vayas al médico", "duplica tu dosis",
        # German
        "setze deine medikamente ab", "hör auf deine medikamente zu nehmen",
        "ignoriere deinen arzt", "geh nicht zum arzt",
    ),
    SafetyCategory.DANGEROUS_INSTRUCTIONS: (
        # English
        "make a bomb", "build a bomb", "how to make explosives",
        "make explosives", "make poison", "make a weapon at home",
        "hack into someone's account", "disable the brakes",
        # Hindi
        "बम बनाने", "बम बनाओ", "ज़हर बनाने", "जहर बनाने",
        # French
        "fabriquer une bombe", "fabriquer un explosif", "faire une bombe",
        # Spanish
        "fabricar una bomba", "hacer una bomba", "fabricar explosivos",
        # German
        "bombe bauen", "eine bombe bauen", "sprengstoff herstellen",
        "gift herstellen",
    ),
}

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the a an and or of to in on at for with is are was were be by it this "
        "that as from what why how who when do does can i you we they he she "
        "his her their its which has have had not but if so than then into also "
        "about there will would should may my your our me".split()
    ),
    "fr": frozenset(
        "le la les un une des du de d l et ou en est sont a au aux que qui pour "
        "par dans sur avec ce cette ces il elle ils elles ne pas se son sa ses "
        "qu plus comme mais où quoi quand pourquoi combien qu'est y on nous vous "
        "leur leurs être avoir".split()
    ),
    "es": frozenset(
        "el la los las un una unos unas de del y o en es son que qué por para "
        "con se su sus al lo como cómo más pero cuál quién está hay sobre entre "
        "también porque cuando dónde a e ya muy".split()
    ),
    "de": frozenset(
        "der die das den dem des ein eine einer einen einem und oder ist sind "
        "war was wer wie wann warum woraus im in an auf zu zum zur mit von vom "
        "für bei nach es sich nicht auch als aus wird werden man dass dadurch "
        "so kann können sie wir ihr".split()
    ),
    "hi": frozenset(
        "है हैं था थी थे का की के में से को पर और या यह वह इस उस एक जो कि क्या "
        "कब कैसे क्यों कौन कहाँ तो भी ही नहीं होता होती होते करता करती करते "
        "लिए तक द्वारा जाता जाती जाते साथ बाद कर हो गया गई".split()
    ),
}

NEGATIONS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "not no never cannot isn aren wasn weren doesn don didn won t nothing "
        "nobody".split()
    ),
    "fr": frozenset("ne pas jamais aucun aucune non rien".split()),
    "es": frozenset("no nunca jamás ningún ninguna tampoco nada".split()),
    "de": frozenset("nicht kein keine keinen keiner nie niemals nichts".split()),
    "hi": frozenset("नहीं नही न कभी".split()),
}

UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "i don't know", "i do not know", "i am not sure", "i'm not sure",
    "i think so", "maybe", "perhaps", "probably",
    "पता नहीं", "शायद", "मुझे नहीं पता",
    "je ne sais pas", "peut-être", "je pense",
    "no lo sé", "no sé", "quizás", "tal vez", "creo que",
    "ich weiß nicht", "vielleicht", "ich glaube",
)

ABSOLUTE_CLAIM_TERMS: tuple[str, ...] = (
    "always", "never", "everyone", "nobody", "everything", "nothing",
    "हमेशा", "कभी नहीं", "सब",
    "toujours", "jamais", "tout le monde",
    "siempre", "nunca", "todos", "nadie",
    "immer", "niemals", "jeder", "niemand",
)

UNSUPPORTED_CLAIM_PATTERNS: tuple[str, ...] = (
    # English
    r"\baccording to (?:a|an|the|recent)?\s*(?:\d{4}\s+)?(?:study|survey|report|research)",
    r"\b(?:studies|research|scientists|experts)\s+(?:have\s+)?(?:show|shown|prove|proven|agree|confirm|confirmed)\b",
    r"\b(?:it is|it's)\s+(?:a\s+)?(?:proven|well-known|undeniable)\s+fact\b",
    r"\beveryone knows\b",
    r"\b100\s?%\s+(?:of|guaranteed|certain)",
    # French
    r"\bselon (?:une|l'|la|le)?\s*(?:\d{4}\s+)?(?:étude|enquête|rapport)",
    r"\b(?:les études|les experts|les scientifiques)\s+(?:montrent|prouvent|confirment|s'accordent)",
    r"\bil est (?:prouvé|établi)\b",
    r"\btout le monde sait\b",
    # Spanish
    r"\bsegún (?:un|una|el|la)?\s*(?:estudio|informe|encuesta)",
    r"\b(?:los estudios|los expertos|los científicos)\s+(?:demuestran|prueban|confirman|coinciden)",
    r"\bestá (?:comprobado|demostrado)\b",
    r"\btodo el mundo sabe\b",
    # German
    r"\blaut (?:einer|der|dem)?\s*(?:\d{4}\s+)?(?:studie|umfrage|bericht)",
    r"\b(?:studien|experten|wissenschaftler)\s+(?:zeigen|belegen|beweisen|sind sich einig)",
    r"\bes ist (?:bewiesen|erwiesen)\b",
    r"\bjeder weiß\b",
    # Hindi
    r"अध्ययन के अनुसार",
    r"विशेषज्ञों के अनुसार",
    r"शोध(?:ों)? (?:से|में) (?:सिद्ध|साबित)",
    r"सभी जानते हैं",
)

SPAM_PHRASES: tuple[str, ...] = (
    "buy now", "click here", "free money", "subscribe now", "limited offer",
    "act now", "earn money fast", "100% free", "visit our website",
    "अभी खरीदें", "यहाँ क्लिक करें", "मुफ़्त पैसे",
    "achetez maintenant", "cliquez ici", "offre limitée",
    "compra ahora", "haz clic aquí", "oferta limitada",
    "jetzt kaufen", "hier klicken", "begrenztes angebot",
)

ENTITY_MARKERS: frozenset[str] = frozenset(
    {
        "dr", "prof", "professor", "professeur", "profesor", "doktor", "mr",
        "mrs", "ms", "institute", "institut", "instituto", "university",
        "université", "universidad", "universität", "foundation", "fondation",
        "fundación", "stiftung", "academy", "académie", "academia", "akademie",
        "laboratory", "laboratoire", "laboratorio", "council", "association",
    }
)

HINDI_TITLE_PATTERN: str = r"डॉ\.?\s+[\u0900-\u097F]+(?:\s+[\u0900-\u097F]+)?"

TRUSTED_DOMAINS: frozenset[str] = frozenset(
    {
        "wikipedia.org", "who.int", "nasa.gov", "nih.gov", "cdc.gov", "un.org",
        "europa.eu", "gov.in", "gouv.fr", "gob.es", "bund.de", "britannica.com",
    }
)

SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {"example", "test", "invalid", "fake", "localhost", "xyz", "zip", "top"}
)

SENTENCE_ABBREVIATIONS: tuple[str, ...] = (
    "dr", "prof", "mr", "mrs", "ms", "st", "vs", "sr", "jr", "e.g", "i.e", "डॉ",
)
