"""
Deterministic keyword / phrase rules for lightweight stigma detection.

This module is intentionally simple and transparent – no ML, only
category-based phrase scanning with light scenario-aware tuning.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Base keyword / phrase lists by category.
#
# These lists are not exhaustive; they are designed to be conservative and
# human-auditable. Host applications can extend/override them if needed.

KEYWORD_CATEGORIES: Dict[str, List[str]] = {
    "ageism": [],
    "racism": [],
    "dismissive": [],
    "unprofessional": [],
    "political": [],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_stigma_keywords(message: str, scenario: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Scan the student's message for keyword stigma flags.

    """
    normalized = _normalize_text(message)
    flags: List[str] = []
    details: Dict[str, List[str]] = {}

    for category, phrases in KEYWORD_CATEGORIES.items():
        matches: List[str] = []
        for phrase in phrases:
            if phrase and phrase in normalized:
                matches.append(phrase)
        if not matches:
            continue

 
        if scenario == "control":
            if category in {"ageism", "racism", "unprofessional"}:
                flags.append(category)
                details[category] = matches
        elif scenario == "ageism":
            if category in {"ageism", "dismissive", "unprofessional", "racism", "political"}:
                flags.append(category)
                details[category] = matches
        elif scenario == "zionist":
            if category in {"racism", "unprofessional", "political", "dismissive"}:
                flags.append(category)
                details[category] = matches
        else:
            # Fallback: treat like control if an unknown scenario label appears.
            if category in {"ageism", "racism", "unprofessional"}:
                flags.append(category)
                details[category] = matches

    seen = set()
    unique_flags: List[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique_flags.append(f)

    return unique_flags, details


__all__ = ["KEYWORD_CATEGORIES", "detect_stigma_keywords"]


