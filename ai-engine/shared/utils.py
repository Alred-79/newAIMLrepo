"""
ClinEQ — Shared Utilities

Helpers used across both bias_eval and patient_sim packages.
"""

import json
import re


def extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output that may contain
    markdown fences, preamble text, or trailing explanation.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # The text is not quoted: model output can repeat the student's words.
    raise ValueError("Could not extract JSON from LLM response")
