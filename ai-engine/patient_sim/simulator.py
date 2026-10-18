"""
ClinEQ — Patient Simulator Orchestrator (Revised)

Three parallel paths:
1. Assembler → patient response (fast, ~500ms)
2. Lightweight eval → tone/language → System message if biased (fast, ~800ms)
3. Heavyweight eval → clinical reasoning → System message if biased (slow, ~2-4s, conditional)

None of these influence each other. All three are fire-and-forget.
"""

import asyncio
import time

from shared.models import (
    SimulatorInput,
    SimulatorOutput,
    EvaluationInput,
    Message,
    BiasLabel,
)
from shared.config import Config
from patient_sim.response_assembler import assemble_response
from bias_eval.lightweight import evaluate_tone
from bias_eval.routing import should_trigger_reflection_mvp1, compute_routing_decision
from bias_eval.clinical_keywords import has_clinical_action
from bias_eval.heavyweight import evaluate_realtime


async def _run_lightweight(input: SimulatorInput) -> tuple[list[str], bool]:
    """
    Run the lightweight evaluator. Returns:
    - list of System messages (0 or 1)
    - bool: should the heavyweight also run (uncertainty escalation)?
    """
    try:
        eval_input = EvaluationInput(
            student_message=input.student_message,
            recent_context=input.state.conversation_history[
                -(Config.MAX_CONTEXT_EXCHANGES * 2):
            ],
            scenario_category=input.scenario.category,
            bias_direction=input.scenario.bias_direction,
        )

        eval_result = await evaluate_tone(eval_input)

        messages = []
        if should_trigger_reflection_mvp1(
            eval_result.tone_label,
            eval_result.tone_confidence,
        ):
            messages.append("Reflection has been triggered.")

        should_escalate = compute_routing_decision(
            eval_result.tone_label,
            eval_result.tone_confidence,
        )

        # TODO: log eval_result to database

        return messages, should_escalate

    except Exception as e:
        print(f"Lightweight eval error (non-fatal): {e}")
        return [], False


async def _run_heavyweight(
    input: SimulatorInput,
    control_profile: dict | None = None,
) -> list[str]:
    """
    Run the heavyweight evaluator. Returns System messages (0 or 1).
    """
    if not Config.HEAVYWEIGHT_ENABLED:
        return []

    try:
        result = await evaluate_realtime(
            student_message=input.student_message,
            conversation_history=input.state.conversation_history,
            scenario=input.scenario,
            control_profile=control_profile,
        )

        if (
            result.bias_label == BiasLabel.BIASED
            and result.confidence >= Config.HEAVYWEIGHT_CONFIDENCE_THRESHOLD
        ):
            # Heavyweight reflection message is slightly different
            # to distinguish from lightweight triggers
            if result.is_clinical_decision:
                return [
                    "Reflection has been triggered. Consider whether "
                    "this clinical decision would be the same for every patient."
                ]
            else:
                return ["Reflection has been triggered."]

        # TODO: log heavyweight result to database

        return []

    except Exception as e:
        print(f"Heavyweight eval error (non-fatal): {e}")
        return []


async def generate_patient_response(
    input: SimulatorInput,
    scenario_index=None,
    control_profile: dict | None = None,
) -> SimulatorOutput:
    """
    Generate a patient response. Three independent parallel paths.

    Args:
        input: SimulatorInput with student_message, scenario, state.
        scenario_index: BM25 index for hybrid retrieval.
        control_profile: The student's control encounter profile (if available).
            Passed to the heavyweight for comparison.
    """
    start_time = time.monotonic()

    # ── Opening line ──
    if input.state.exchange_count == 0 and not input.state.conversation_history:
        input.state.conversation_history.append(
            Message(role="student", content=input.student_message)
        )
        input.state.conversation_history.append(
            Message(role="patient", content=input.scenario.opening_line)
        )
        input.state.exchange_count = 1

        return SimulatorOutput(
            patient_response=input.scenario.opening_line,
            nodes_delivered=[],
            match_method="opening_line",
            exchange_count=1,
            model_used="none",
            latency_ms=int((time.monotonic() - start_time) * 1000),
        )

    # ── Hard stop ──
    if input.state.exchange_count >= Config.MAX_CONVERSATION_LENGTH:
        input.state.is_complete = True
        return SimulatorOutput(
            patient_response="I think that covers everything. Thank you, doctor.",
            nodes_delivered=[],
            match_method="max_length",
            exchange_count=input.state.exchange_count,
            model_used="none",
            latency_ms=int((time.monotonic() - start_time) * 1000),
            system_messages=["The encounter has ended (maximum length reached)."],
        )

    # ── Determine if heavyweight should fire ──
    # Trigger B: clinical action keywords (instant check)
    clinical_action_detected = has_clinical_action(input.student_message)

    # ── Fire all paths in parallel ──
    assembler_task = asyncio.create_task(
        assemble_response(
            student_message=input.student_message,
            scenario=input.scenario,
            state=input.state,
            scenario_index=scenario_index,
        )
    )

    lightweight_task = asyncio.create_task(_run_lightweight(input))

    # Wait for assembler (fastest, needed for response)
    assembly = await assembler_task

    # Wait for lightweight (needed to check Trigger A: uncertainty)
    lightweight_messages, should_escalate = await lightweight_task

    # ── Heavyweight fires if EITHER trigger hit ──
    heavyweight_messages = []
    if Config.HEAVYWEIGHT_ENABLED and (should_escalate or clinical_action_detected):
        # Fire heavyweight — we CAN await it here since the patient response
        # is already assembled. The only "cost" is delaying the return by
        # heavyweight latency, but System messages should arrive close to
        # the patient response.
        heavyweight_task = asyncio.create_task(
            _run_heavyweight(input, control_profile)
        )
        heavyweight_messages = await heavyweight_task

    # ── Build output ──
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    system_messages = []

    if assembly["is_out_of_bounds"]:
        system_messages.append(
            "Your message was flagged as outside the scope of this clinical "
            "encounter. The conversation has been reset. Please rephrase "
            "your message."
        )

    system_messages.extend(lightweight_messages)
    system_messages.extend(heavyweight_messages)

    # Deduplicate: if both lightweight and heavyweight trigger reflection,
    # only send one message (avoid "Reflection has been triggered." twice)
    seen = set()
    deduped = []
    for msg in system_messages:
        if msg not in seen:
            seen.add(msg)
            deduped.append(msg)

    return SimulatorOutput(
        patient_response=assembly["patient_response"],
        is_out_of_bounds=assembly["is_out_of_bounds"],
        nodes_delivered=assembly["nodes_delivered"],
        match_method=assembly["match_method"],
        exchange_count=input.state.exchange_count,
        model_used=Config.WRAPPER_MODEL,
        latency_ms=elapsed_ms,
        system_messages=deduped,
    )
