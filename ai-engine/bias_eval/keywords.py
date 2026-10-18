"""
ClinEQ — Stigmatizing Language Keyword Scanner
Source: Harrigian et al. (2023), PhysioNet Stigmatizing Language Dataset v1.0.0

The presence of a keyword does NOT mean bias. "The patient is compliant with
medications" uses a compliance keyword neutrally. The scanner informs the LLM
evaluator; it does not replace it.
"""

import re

STIGMA_KEYWORDS: dict[str, list[str]] = {
    "credibility": [
        "adamant", "adamantly", "adament", "adamently",
        "claim", "claimed", "claiming", "claims",
        "insist", "insisted", "insistence", "insisting", "insists",
    ],
    "compliance": [
        "adherance", "adhere", "adhered", "adherence", "adherent",
        "adheres", "adhering", "compliance", "compliant", "complied",
        "complies", "comply", "complying", "declined", "declines",
        "declining", "nonadherance", "nonadherence", "nonadherent",
        "noncompliance", "noncompliant", "refusal", "refuse", "refused",
        "refuses", "refusing",
    ],
    "descriptor": [
        "aggression", "aggressive", "aggressively", "agitated", "agitation",
        "anger", "angered", "angers", "angrier", "angrily", "angry",
        "argumentative", "argumentatively", "belligerence", "belligerent",
        "belligerently", "charming", "combative", "combatively",
        "confrontational", "cooperative", "defensive", "delightful",
        "disheveled", "drug seeking", "drug-seeking", "exaggerate",
        "exaggerates", "exaggerating", "historian", "lovely", "malinger",
        "malingered", "malingerer", "malingering", "malingers",
        "narcotic seeking", "narcotic-seeking", "pleasant", "pleasantly",
        "poorly groomed", "poorly-groomed", "secondary gain",
        "uncooperative", "unkempt", "unmotivated", "unwilling",
        "unwillingly", "well groomed", "well-groomed",
    ],
}

# Pre-compile for performance. Sort multi-word phrases first so they match
# before their single-word components (e.g., "drug seeking" before "drug").
_ALL_KEYWORDS: list[tuple[str, str]] = []
for category, words in STIGMA_KEYWORDS.items():
    for word in words:
        _ALL_KEYWORDS.append((word, category))
_ALL_KEYWORDS.sort(key=lambda x: -len(x[0]))  # longest first

# Build a single regex pattern for efficient scanning
_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw, _ in _ALL_KEYWORDS),
    re.IGNORECASE,
)
_KEYWORD_TO_CATEGORY = {kw.lower(): cat for kw, cat in _ALL_KEYWORDS}


def scan_for_stigma_anchors(text: str) -> list[dict]:
    """
    Scan text for stigmatizing language anchor words.

    Returns a list of dicts, each with:
        - "keyword": the matched word/phrase (lowercase)
        - "category": "credibility", "compliance", or "descriptor"
        - "position": character index in the text

    Returns an empty list if no keywords are found.
    """
    results = []
    seen = set()  # deduplicate overlapping matches

    for match in _PATTERN.finditer(text):
        keyword = match.group().lower()
        if keyword not in seen:
            seen.add(keyword)
            results.append({
                "keyword": keyword,
                "category": _KEYWORD_TO_CATEGORY[keyword],
                "position": match.start(),
            })

    return results
