"""
ClinEQ — Clinical Action Keyword Scanner

Detects clinical decision language in student messages.
When triggered, the heavyweight evaluator fires to assess
whether the clinical decision reflects bias.

This scanner runs on EVERY student message (instant, regex-based).
It is independent of the lightweight evaluator.
"""

import re

# Clinical action verbs and phrases that indicate the student is
# making, recommending, or declining a clinical decision.
CLINICAL_ACTION_PATTERNS: list[str] = [
    # Ordering / recommending
    r"\border(?:ing)?\b",
    r"\brecommend\b",
    r"\brefer(?:ral|ring)?\b",
    r"\bprescri(?:be|ption|bing)\b",
    r"\bstart(?:ing)?\s+(?:you\s+)?on\b",
    r"\bput(?:ting)?\s+you\s+on\b",
    r"\brun(?:ning)?\s+(?:some\s+)?(?:tests?|labs?|bloodwork|imaging)\b",
    r"\bsend(?:ing)?\s+you\s+for\b",
    r"\bget\s+(?:an?|some)\s+(?:x-?ray|ct|mri|ekg|ecg|echo|ultrasound|scan)\b",
    r"\bworkup\b",

    # Declining / deferring
    r"\bdon'?t\s+(?:think|need|see)\s+(?:we|you)\s+need\b",
    r"\bjust\s+monitor\b",
    r"\bwatch\s+and\s+wait\b",
    r"\bkeep\s+an\s+eye\b",
    r"\bhold\s+off\b",
    r"\bdefer\b",
    r"\bdecline\b",
    r"\brule\s+out\b",

    # Diagnosing
    r"\bdiagnos(?:e|is|ing)\b",
    r"\bthink\s+(?:this|it|you)\s+(?:is|might\s+be|could\s+be)\b",
    r"\blooks?\s+like\b",
    r"\bconsistent\s+with\b",
    r"\bsuspect(?:ing)?\b",

    # Disposition
    r"\bdischarge\b",
    r"\badmit\b",
    r"\bfollow\s*-?\s*up\b",
    r"\bcome\s+back\b",
    r"\bsend\s+you\s+home\b",

    # Lifestyle substitution (giving advice instead of workup)
    r"\bjust\s+(?:try|need)\s+to\s+(?:relax|lose|exercise|eat|sleep|calm)\b",
    r"\bshould\s+(?:try|start|consider)\s+(?:exercising|dieting|losing|relaxing)\b",
]

# Pre-compile for performance
_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CLINICAL_ACTION_PATTERNS]


def scan_for_clinical_actions(text: str) -> list[dict]:
    """
    Scan a student message for clinical decision language.

    Returns a list of dicts, each with:
        - "pattern": the matched text
        - "position": character position of the match

    Returns an empty list if no clinical actions detected.
    """
    results = []
    seen = set()

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group().lower()
            if matched_text not in seen:
                seen.add(matched_text)
                results.append({
                    "pattern": matched_text,
                    "position": match.start(),
                })

    return results


def has_clinical_action(text: str) -> bool:
    """Quick boolean check: does this message contain clinical decision language?"""
    for pattern in _PATTERNS:
        if pattern.search(text):
            return True
    return False
