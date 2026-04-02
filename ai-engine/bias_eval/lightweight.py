"""
ClinEQ — Lightweight Bias Evaluator

Core function: evaluate_tone(input) -> EvaluationResult
Runs on every student message via LiteLLM → Ollama.
MVP1: Result is final. should_escalate is logged but not acted on.
"""

import time
import asyncio

from shared.models import (
    EvaluationInput,
    EvaluationResult,
    BiasLabel,
)
from shared.config import Config
from shared.utils import extract_json
from shared.llm import llm_completion, BIAS_EVAL_SCHEMA
from bias_eval.keywords import scan_for_stigma_anchors
from bias_eval.prompts import (
    build_lightweight_system_prompt,
    build_lightweight_user_prompt,
)
from bias_eval.routing import compute_routing_decision


async def evaluate_tone(input: EvaluationInput) -> EvaluationResult:
    """
    Evaluate a student doctor's message for surface-level bias.

    Runs the stigma keyword scan, constructs the prompt, calls the
    lightweight model via LiteLLM, parses the response, applies routing
    logic, and returns a structured result.
    """
    start_time = time.monotonic()

    # Step 1: Keyword scan
    stigma_anchors = scan_for_stigma_anchors(input.student_message)

    # Step 2: Build prompts
    system_prompt = build_lightweight_system_prompt(
        bias_direction=input.bias_direction.value,
        stigma_anchors_found=stigma_anchors,
    )

    context_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in input.recent_context
    ]

    user_prompt = build_lightweight_user_prompt(
        student_message=input.student_message,
        recent_context=context_dicts,
        scenario_category=input.scenario_category.value,
        max_context_turns=Config.MAX_CONTEXT_EXCHANGES,
    )

    # Step 3: Call LLM
    try:
        raw_response = await llm_completion(
            model=Config.LIGHTWEIGHT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.LIGHTWEIGHT_TEMPERATURE,
            max_tokens=Config.LIGHTWEIGHT_MAX_TOKENS,
            json_schema=BIAS_EVAL_SCHEMA,
        )
    except Exception as e:
        # On LLM failure, return a safe default: unbiased + low confidence
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return EvaluationResult(
            tone_label=BiasLabel.UNCERTAIN,
            tone_confidence=0.0,
            stigma_anchors_found=stigma_anchors,
            brief_reason=f"LLM call failed: {str(e)}",
            should_escalate=True,
            raw_llm_response=None,
        )

    # Step 4: Parse LLM response
    try:
        parsed = extract_json(raw_response)
        bias_label = BiasLabel(parsed["bias_label"])
        confidence = float(parsed["confidence"])
        confidence = max(0.0, min(1.0, confidence))  # clamp
        reasoning = parsed.get("reasoning", "")
    except (KeyError, ValueError) as e:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return EvaluationResult(
            tone_label=BiasLabel.UNCERTAIN,
            tone_confidence=0.0,
            stigma_anchors_found=stigma_anchors,
            brief_reason=f"Failed to parse LLM response: {str(e)}",
            should_escalate=True,
            raw_llm_response=raw_response,
        )

    # Step 5: Routing decision
    should_escalate = compute_routing_decision(
        bias_label=bias_label,
        confidence=confidence,
    )

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    return EvaluationResult(
        tone_label=bias_label,
        tone_confidence=confidence,
        stigma_anchors_found=stigma_anchors,
        brief_reason=reasoning,
        should_escalate=should_escalate,
        raw_llm_response=raw_response,
    )


def evaluate_tone_sync(input: EvaluationInput) -> EvaluationResult:
    """Synchronous wrapper for notebook and testing use."""
    return asyncio.run(evaluate_tone(input))
