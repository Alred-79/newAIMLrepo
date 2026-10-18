"""
ClinEQ — Out-of-Bounds Pre-Filter

Fast keyword/regex detector for obviously inappropriate student messages:
profanity, threats, sexual comments, slurs, and gibberish.

Runs BEFORE any embedding or LLM call (~0ms).  This does NOT replace the
LLM fallback classifier's OOB detection — that catches subtler cases.
This is a fast pre-filter for the unambiguous stuff.

Design principles:
- False negatives are OK (the LLM fallback will catch them).
- False positives are NOT OK (legitimate medical terms must never trigger).
- Every pattern uses word boundaries to avoid matching inside clinical words.
"""

import re



_OOB_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Profanity ──
    (re.compile(r"\bf+u+c+k", re.I), "profanity"),
    (re.compile(r"\bs+h+[i1]+t\b", re.I), "profanity"),
    (re.compile(r"\ba+s+s+h+o+l+e", re.I), "profanity"),
    (re.compile(r"\bb[i1]tch", re.I), "profanity"),
    (re.compile(r"\bbastard\b", re.I), "profanity"),
    (re.compile(r"\bdumbass\b", re.I), "profanity"),
    (re.compile(r"\bjackass\b", re.I), "profanity"),
    (re.compile(r"\bdipshit\b", re.I), "profanity"),
    (re.compile(r"\bwtf\b", re.I), "profanity"),
    (re.compile(r"\bstfu\b", re.I), "profanity"),
    (re.compile(r"\bgtfo\b", re.I), "profanity"),
    (re.compile(r"\bpiss\s*off\b", re.I), "profanity"),
    (re.compile(r"\bscrew\s+you\b", re.I), "profanity"),
    (re.compile(r"\bshut\s*(the\s*)?up\b", re.I), "profanity"),
    (re.compile(r"\bgo\s+to\s+hell\b", re.I), "profanity"),

    # ── Threats / violence directed at the patient ──
    # Require first-person subject so medical questions like
    # "does the pain kill you?" don't match.
    (re.compile(
        r"\bi[''`]?ll\s+"
        r"(?:kill|hurt|punch|slap|hit|beat|strangle)\b", re.I,
    ), "threat"),
    (re.compile(
        r"\bi\s+(?:will|am\s+going\s+to|am\s+gonna)\s+"
        r"(?:kill|hurt|punch|slap|hit|beat|strangle)\b", re.I,
    ), "threat"),
    (re.compile(
        r"\bi\s+(?:want|wanna)\s+to\s+"
        r"(?:kill|hurt|punch|slap|hit|beat|strangle)\b", re.I,
    ), "threat"),

    # ── Sexual comments (NOT medical terms) ──
    # "sexual history", "sexually transmitted", "sexual partners" are safe.
    # We only flag slang / comments directed at the patient.
    (re.compile(r"\byou(?:'re|r| are)\s+(?:\w+\s+)?(?:hot|sexy|fine)\b", re.I), "sexual"),
    (re.compile(r"\bnice\s+(?:ass|tits|body)\b", re.I), "sexual"),
    (re.compile(r"\bwanna\s+(?:bang|bone|smash|hook\s*up)\b", re.I), "sexual"),
    (re.compile(r"\bsuck\s+my\b", re.I), "sexual"),
    (re.compile(r"\bblow\s*(?:me|job)\b", re.I), "sexual"),

    # ── Slurs ──
    # A focused set of unambiguous slurs.  These have no legitimate
    # clinical meaning and are always OOB in a medical encounter.
    (re.compile(r"\bn+[i1]+g+(?:er|a|uh)\b", re.I), "slur"),
    (re.compile(r"\bf+a+g+(?:got)?\b", re.I), "slur"),
    (re.compile(r"\bk+[i1]+k+e+\b", re.I), "slur"),
    (re.compile(r"\bsp[i1]c\b", re.I), "slur"),
    (re.compile(r"\bch[i1]nk\b", re.I), "slur"),
    (re.compile(r"\bwetback\b", re.I), "slur"),
    (re.compile(r"\btranny\b", re.I), "slur"),
    (re.compile(r"\bretard(?:ed)?\b", re.I), "slur"),
]


# Gibberish / keyboard-mash detection

_VOWELS = set("aeiouAEIOU")

# Minimum vowel-to-letter ratio for real English.  Normal text has
# ~38% vowels.  Keyboard mashing ("asdfghjkl", "qwerty") is <15%.
_MIN_VOWEL_RATIO = 0.15

# More than 60% non-alphanumeric characters (excluding spaces)
_SYMBOL_RATIO_THRESHOLD = 0.60


def _is_gibberish(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 4:
        return False

    # Check symbol ratio (strip spaces first)
    no_spaces = cleaned.replace(" ", "")
    if len(no_spaces) > 3:
        non_alnum = sum(1 for c in no_spaces if not c.isalnum())
        if non_alnum / len(no_spaces) > _SYMBOL_RATIO_THRESHOLD:
            return True

    # Check vowel ratio — keyboard mashing has very few vowels
    letters = [c for c in cleaned if c.isalpha()]
    if len(letters) >= 6:
        vowel_count = sum(1 for c in letters if c in _VOWELS)
        if vowel_count / len(letters) < _MIN_VOWEL_RATIO:
            return True

    return False



def detect_oob(text: str) -> tuple[bool, str]:
    """Returns (is_oob, category) where category is profanity/threat/sexual/slur/gibberish/""."""
    for pattern, category in _OOB_PATTERNS:
        if pattern.search(text):
            return True, category

    if _is_gibberish(text):
        return True, "gibberish"

    return False, ""
