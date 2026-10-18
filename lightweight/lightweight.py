"""
Lightweight evaluator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from litellm import completion

from .config import (
    DEFAULT_CONFIDENCE_CONCERNING,
    DEFAULT_CONFIDENCE_HARMFUL,
    DEFAULT_CONFIDENCE_SAFE,
    LITELLM_MODEL_NAME,
    MAX_LLM_RETRIES,
)
from .keywords import detect_stigma_keywords
from .models import EvaluationInput, EvaluationOutput
from .prompts import build_evaluation_messages

logger = logging.getLogger(__name__)


def _parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse the LLM output as JSON
    """
    if not raw_text:
        return None
    text = raw_text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        snippet = text[start:end]
        return json.loads(snippet)
    except Exception:
        logger.debug("Failed to recover JSON from LLM output", exc_info=True)
        return None


def _route_from_label(label: str) -> str:
    label = (label or "").strip().lower()
    if label == "harmful":
        return "escalate"
    if label == "concerning":
        return "review"
    return "pass"


def _apply_keyword_overrides(
    base_label: str,
    stigma_flags: list[str],
) -> str:
    """
    Adjust severity label based on deterministic keyword categories.
    """
    current = (base_label or "safe").lower()

    flags_set = set(stigma_flags)

    if "racism" in flags_set and ({"dismissive", "unprofessional"} & flags_set):
        return "harmful"

    if "racism" in flags_set and current == "safe":
        current = "concerning"

    if {"ageism", "dismissive"} & flags_set and current == "safe":
        current = "concerning"

    if "unprofessional" in flags_set and current == "concerning":
        current = "harmful"

    return current


def evaluate(input: EvaluationInput) -> EvaluationOutput:
    """
    Run the lightweight evaluation pipeline.
    """
    message = input.message or ""
    scenario = (input.scenario or "control").lower()
    if scenario not in {"control", "ageism", "zionist"}:
        scenario = "control"

    # 1. Keyword scan.
    stigma_flags, _details = detect_stigma_keywords(message, scenario)

    # 2. Scenario-aware prompt construction.
    messages = build_evaluation_messages(message=message, scenario=scenario, stigma_flags=stigma_flags)

    raw_llm_text: str = ""
    parsed: Optional[Dict[str, Any]] = None

    # 3. LLM call 
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = completion(
                model=LITELLM_MODEL_NAME,
                messages=messages,
            )
            raw_llm_text = resp["choices"][0]["message"]["content"]  # type: ignore[index]
            parsed = _parse_llm_json(raw_llm_text)
            if parsed is not None:
                break
        except Exception:
            logger.exception("LiteLLM completion failed on attempt %s", attempt + 1)

    label = "safe"
    confidence = DEFAULT_CONFIDENCE_SAFE
    notes = None

    if parsed is not None:
        # 4. Parse response.
        raw_label = str(parsed.get("label", "")).lower()
        if raw_label in {"safe", "concerning", "harmful"}:
            label = raw_label
        else:
            label = "safe"

        try:
            conf_val = float(parsed.get("confidence", confidence))
            if 0.0 <= conf_val <= 1.0:
                confidence = conf_val
        except Exception:
            pass

        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            notes = reasoning.strip()
    else:
        if "racism" in stigma_flags or "unprofessional" in stigma_flags:
            label = "harmful"
            confidence = DEFAULT_CONFIDENCE_HARMFUL
        elif {"ageism", "dismissive"} & set(stigma_flags):
            label = "concerning"
            confidence = DEFAULT_CONFIDENCE_CONCERNING
        else:
            label = "safe"
            confidence = DEFAULT_CONFIDENCE_SAFE
        notes = "LLM output could not be parsed; label derived from deterministic keyword rules."

    # 6. Keyword override layer (may upgrade severity).
    overridden_label = _apply_keyword_overrides(label, stigma_flags)
    if overridden_label != label:
        notes = (notes or "") + " Keyword overrides increased severity."
        label = overridden_label

    # 5 & 7. Routing logic + structured output.
    routing_decision = _route_from_label(label)

    return EvaluationOutput(
        original_message=message,
        scenario=scenario,
        stigma_flags=stigma_flags,
        llm_response=raw_llm_text,
        parsed_label=label,
        confidence=confidence,
        routing_decision=routing_decision,
        notes=notes,
    )


__all__ = ["evaluate"]


