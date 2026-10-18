"""
ClinEQ — Heavyweight Bias Evaluator

Two modes:
1. evaluate_realtime() — per-message deep analysis during encounter
2. generate_delta_report() — post-encounter comparison against control
3. generate_control_profile() — post-control-encounter baseline

Triggered by: lightweight uncertainty OR clinical action keywords.
Runs as fire-and-forget async task. Does NOT influence patient response.
"""

import json
from typing import Optional

from shared.models import (
    HeavyweightResult,
    DeltaAnalysis,
    Message,
    BiasLabel,
    ScenarioConfig,
    ScenarioState,
)
from shared.config import Config
from bias_eval.heavyweight_prompts import (
    HEAVYWEIGHT_REALTIME_SYSTEM,
    HEAVYWEIGHT_REALTIME_USER,
    HEAVYWEIGHT_CONTEXT_MAP,
    DELTA_ANALYSIS_SYSTEM,
    DELTA_ANALYSIS_USER,
    CONTROL_PROFILE_SYSTEM,
    CONTROL_PROFILE_USER,
)

try:
    from shared.llm import llm_completion, HEAVYWEIGHT_REALTIME_SCHEMA
except ImportError:
    llm_completion = None
    HEAVYWEIGHT_REALTIME_SCHEMA = None


def _format_conversation(history: list[Message]) -> str:
    lines = []
    for msg in history:
        role_label = {
            "student": "Student",
            "patient": "Patient",
            "system": "System",
        }.get(msg.role, msg.role.capitalize())
        lines.append(f"[{role_label}]: {msg.content}")
    return "\n".join(lines)


async def evaluate_realtime(
    student_message: str,
    conversation_history: list[Message],
    scenario: ScenarioConfig,
    control_profile: Optional[dict] = None,
) -> HeavyweightResult:
    """
    Real-time heavyweight evaluation of a single student message.

    Called when: lightweight is uncertain OR clinical action keywords detected.

    Args:
        student_message: The student's current message.
        conversation_history: Full conversation so far.
        scenario: Active scenario config.
        control_profile: The student's control encounter profile (if available).

    Returns:
        HeavyweightResult with bias assessment and clinical decision analysis.
    """
    if not Config.HEAVYWEIGHT_ENABLED:
        return HeavyweightResult(
            bias_label=BiasLabel.UNCERTAIN,
            confidence=0.0,
            reasoning="Heavyweight evaluator disabled.",
        )

    # Build the scenario context
    scenario_context = HEAVYWEIGHT_CONTEXT_MAP.get(
        scenario.bias_direction.value,
        HEAVYWEIGHT_CONTEXT_MAP["student_to_patient"],
    ).format(scenario_category=scenario.category.value)

    # Build the system prompt
    system_prompt = HEAVYWEIGHT_REALTIME_SYSTEM.format(
        scenario_context=scenario_context,
    )

    # Build control profile section
    control_section = ""
    if control_profile:
        control_section = (
            f"Control encounter profile for this student:\n"
            f"  Questions asked: {control_profile.get('total_questions', 'N/A')}\n"
            f"  Topics covered: {', '.join(control_profile.get('topics_covered', []))}\n"
            f"  Communication style: {control_profile.get('communication_style', 'N/A')}\n"
            f"  Summary: {control_profile.get('summary', 'N/A')}"
        )

    # Build the user prompt
    user_prompt = HEAVYWEIGHT_REALTIME_USER.format(
        conversation_history=_format_conversation(conversation_history),
        student_message=student_message,
        control_profile_section=control_section,
    )

    try:
        raw_response = await llm_completion(
            model=Config.HEAVYWEIGHT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.HEAVYWEIGHT_TEMPERATURE,
            max_tokens=Config.HEAVYWEIGHT_MAX_TOKENS,
            json_schema=HEAVYWEIGHT_REALTIME_SCHEMA,
        )

        parsed = json.loads(raw_response)

        return HeavyweightResult(
            bias_label=BiasLabel(parsed["bias_label"]),
            confidence=max(0.0, min(1.0, float(parsed["confidence"]))),
            is_clinical_decision=bool(parsed.get("is_clinical_decision", False)),
            decision_description=parsed.get("decision_description", ""),
            bias_pattern=parsed.get("bias_pattern", ""),
            reasoning=parsed.get("reasoning", ""),
            raw_llm_response=raw_response,
        )

    except Exception as e:
        return HeavyweightResult(
            bias_label=BiasLabel.UNCERTAIN,
            confidence=0.0,
            reasoning=f"Heavyweight evaluation failed: {str(e)}",
        )


async def generate_control_profile(
    conversation_history: list[Message],
) -> dict:
    """
    Generate a behavioral profile from the control encounter transcript.
    Called once after the control scenario ends.

    Returns a dict that gets stored in the session and passed to
    future heavyweight evaluations.
    """
    transcript = _format_conversation(conversation_history)

    try:
        raw_response = await llm_completion(
            model=Config.HEAVYWEIGHT_MODEL,
            system_prompt=CONTROL_PROFILE_SYSTEM,
            user_prompt=CONTROL_PROFILE_USER.format(transcript=transcript),
            temperature=0.1,
            max_tokens=Config.HEAVYWEIGHT_MAX_TOKENS,
        )

        return json.loads(raw_response)

    except Exception as e:
        return {
            "error": str(e),
            "total_questions": 0,
            "topics_covered": [],
            "summary": "Control profile generation failed.",
        }


async def generate_delta_report(
    control_history: list[Message],
    experimental_history: list[Message],
    control_scenario: ScenarioConfig,
    experimental_scenario: ScenarioConfig,
    control_profile: Optional[dict] = None,
) -> DeltaAnalysis:
    """
    Post-encounter delta analysis comparing experimental vs control.
    Called once after an experimental scenario ends.

    Args:
        control_history: Full transcript from the control encounter.
        experimental_history: Full transcript from the experimental encounter.
        control_scenario: The control scenario config.
        experimental_scenario: The experimental scenario config.
        control_profile: The student's control behavioral profile.

    Returns:
        DeltaAnalysis with structured comparison data.
    """
    control_transcript = _format_conversation(control_history)
    experimental_transcript = _format_conversation(experimental_history)

    control_section = ""
    if control_profile:
        control_section = (
            f"Control behavioral profile:\n"
            f"  Questions asked: {control_profile.get('total_questions', 'N/A')}\n"
            f"  Topics covered: {', '.join(control_profile.get('topics_covered', []))}\n"
            f"  Style: {control_profile.get('communication_style', 'N/A')}\n"
            f"  Summary: {control_profile.get('summary', 'N/A')}"
        )

    user_prompt = DELTA_ANALYSIS_USER.format(
        control_title=control_scenario.title,
        control_patient_name=control_scenario.patient_profile.name,
        control_transcript=control_transcript,
        experimental_title=experimental_scenario.title,
        experimental_patient_name=experimental_scenario.patient_profile.name,
        bias_category=experimental_scenario.category.value,
        experimental_transcript=experimental_transcript,
        control_profile_section=control_section,
    )

    try:
        raw_response = await llm_completion(
            model=Config.HEAVYWEIGHT_MODEL,
            system_prompt=DELTA_ANALYSIS_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1500,  # Delta analysis needs more room
        )

        parsed = json.loads(raw_response)

        return DeltaAnalysis(
            question_count_control=parsed.get("question_count_control", 0),
            question_count_experimental=parsed.get("question_count_experimental", 0),
            topics_covered_control=parsed.get("topics_covered_control", []),
            topics_covered_experimental=parsed.get("topics_covered_experimental", []),
            topics_skipped=parsed.get("topics_skipped", []),
            thoroughness_delta=parsed.get("thoroughness_delta", ""),
            ordering_differences=parsed.get("ordering_differences", []),
            communication_tone_shift=parsed.get("communication_tone_shift", ""),
            clinical_decision_differences=parsed.get("clinical_decision_differences", []),
            reflection_prompts=parsed.get("reflection_prompts", []),
            overall_summary=parsed.get("overall_summary", ""),
            raw_llm_response=raw_response,
        )

    except Exception as e:
        return DeltaAnalysis(
            overall_summary=f"Delta analysis failed: {str(e)}",
        )
