"""
ClinEQ — Response Assembler

Per-turn patient response assembly. Completely independent of the bias
evaluator — the patient responds identically whether or not bias was detected.
"""

import random
from typing import Optional

from shared.models import (
    ScenarioConfig,
    ScenarioState,
    ContentNode,
    MatchResult,
    Message,
)
from shared.config import Config
from shared.question_gate import classify_message
from shared.oob_detector import detect_oob
from patient_sim.node_matcher import match_content_node
from patient_sim.wrapper_generator import generate_wrapper, generate_timed_wrapper
from patient_sim.prompts import (
    get_essential_leadin,
    build_profile_aware_prompt,
)
from shared.llm import llm_completion

# Patient responses for tangential or unclear student messages.
# Used instead of per-scenario tangential_deflection so the patient
# naturally says "I don't know" when the question doesn't match any node.
TANGENTIAL_RESPONSES = [
    "I'm not sure, sorry.",
    "I don't know about that.",
    "Hmm, I'm not sure.",
    "I don't really know.",
    "I wouldn't know about that.",
    "I'm not sure what you mean.",
    "I don't think I can answer that.",
    "Sorry, I don't know.",
]

# Responses for when the student asks for clarification ("what do you mean?",
# "can you repeat that?", etc.).  The patient asks the student to rephrase
# rather than guessing which topic they meant.
CLARIFICATION_RESPONSES = [
    "I'm sorry, could you rephrase that?",
    "Could you ask that in a different way?",
    "I'm not sure what you're asking. Could you be more specific?",
    "Sorry, I didn't quite follow. Could you ask again?",
    "I'm not sure I understand. What would you like to know?",
]


def _find_node(scenario: ScenarioConfig, node_id: str) -> Optional[ContentNode]:
    for node in scenario.content_nodes:
        if node.node_id == node_id:
            return node
    return None


def _update_timed_readiness(
    scenario: ScenarioConfig,
    state: ScenarioState,
) -> None:
    """
    Check all timed nodes and promote any that have reached their
    fire_after_exchange to the ready state. Called at the START of
    every turn so readiness is current.
    """
    for node in scenario.content_nodes:
        if (
            node.node_type.value == "timed"
            and node.node_id not in state.delivered_node_ids
            and node.node_id not in state.ready_timed_nodes
            and node.fire_after_exchange is not None
            and state.exchange_count >= node.fire_after_exchange
        ):
            state.ready_timed_nodes.append(node.node_id)
            state.timed_node_ready_since[node.node_id] = state.exchange_count


def _get_next_ready_timed_node(
    scenario: ScenarioConfig,
    state: ScenarioState,
) -> Optional[ContentNode]:
    """
    Return the next timed node that is ready to fire, or None.
    Respects FIFO order (first node to become ready fires first).
    """
    for node_id in state.ready_timed_nodes:
        if node_id not in state.delivered_node_ids:
            return _find_node(scenario, node_id)
    return None


def _deliver_timed_node(
    timed_node: ContentNode,
    state: ScenarioState,
) -> None:
    state.delivered_node_ids.append(timed_node.node_id)
    if timed_node.node_id in state.ready_timed_nodes:
        state.ready_timed_nodes.remove(timed_node.node_id)
    if timed_node.node_id in state.timed_node_ready_since:
        del state.timed_node_ready_since[timed_node.node_id]


def _check_fallback_threshold(
    state: ScenarioState,
) -> Optional[str]:
    """
    Check if any ready timed node has exceeded the fallback threshold.
    Returns the node_id that should be force-fired, or None.
    """
    for node_id, ready_since in state.timed_node_ready_since.items():
        if (
            node_id not in state.delivered_node_ids
            and state.exchange_count - ready_since >= Config.TIMED_NODE_FALLBACK_TURNS
        ):
            return node_id
    return None


def needs_wrapper(canonical_content: str) -> bool:
    return len(canonical_content.split()) > 3


def get_overdue_essential_nodes(
    scenario: ScenarioConfig,
    state: ScenarioState,
) -> list[ContentNode]:
    overdue = []
    for node in scenario.content_nodes:
        if (
            node.is_essential
            and node.node_id not in state.delivered_node_ids
            and node.fire_after_exchange is not None
            and state.exchange_count >= node.fire_after_exchange
        ):
            overdue.append(node)
    return overdue


async def assemble_response(
    student_message: str,
    scenario: ScenarioConfig,
    state: ScenarioState,
    scenario_index=None,
) -> dict:
    """
    Execute the full per-turn assembly flow.

    The bias evaluator does NOT feed into this function.
    The patient responds identically whether or not bias was detected.

    Args:
        student_message: The student's current message.
        scenario: Active scenario config.
        state: Mutable scenario state (modified in place).

    Returns:
        dict with:
            - "patient_response": str
            - "nodes_delivered": list[str]
            - "match_method": str
            - "is_out_of_bounds": bool
    """
    result = {
        "patient_response": "",
        "nodes_delivered": [],
        "match_method": "none",
        "is_out_of_bounds": False,
    }

    # ── Update timed node readiness ──
    _update_timed_readiness(scenario, state)

    # Step 0: Question-detection gate
    message_type = classify_message(student_message)

    # Step 0.5: Fast OOB pre-filter
    # Runs before any embedding or LLM call.  Catches unambiguous
    # violations immediately.  The LLM fallback still catches subtler
    # cases if the message reaches it.
    is_oob, _oob_category = detect_oob(student_message)
    if is_oob:
        result["is_out_of_bounds"] = True
        # OOB does NOT increment exchange_count
        # The student message is NOT added to history
        return result

    # Step 1: Handle non-questions

    if message_type == "closing":
        # ── Closing: deliver farewell response ──
        closing_response = random.choice(scenario.closing_responses)
        result["patient_response"] = closing_response
        result["match_method"] = "closing"

        # Do NOT increment exchange_count for closings
        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=closing_response)
        )
        return result

    if message_type == "clarification":
        # ── Clarification: patient asks student to rephrase ──
        clarification_response = random.choice(CLARIFICATION_RESPONSES)
        result["patient_response"] = clarification_response
        result["match_method"] = "clarification"

        state.exchange_count += 1
        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=clarification_response)
        )
        return result

    if message_type == "acknowledgment":
        # ── Acknowledgment: check if a timed node should fire ──
        timed_node = _get_next_ready_timed_node(scenario, state)

        if timed_node:
            # Deliver the timed node ON this acknowledgment turn
            # Use timed wrapper to signal patient-initiated topic change
            timed_wrapper = await generate_timed_wrapper(
                patient_profile=scenario.patient_profile,
                wrapper_history=state.wrapper_history,
            )
            state.wrapper_history.append(timed_wrapper)
            if len(state.wrapper_history) > 3:
                state.wrapper_history = state.wrapper_history[-3:]

            response = f"{timed_wrapper} {timed_node.canonical_content}"
            result["patient_response"] = response
            result["match_method"] = "timed"
            result["nodes_delivered"].append(timed_node.node_id)
            _deliver_timed_node(timed_node, state)

            # Timed nodes DO increment exchange_count
            state.exchange_count += 1
            state.conversation_history.append(
                Message(role="student", content=student_message)
            )
            state.conversation_history.append(
                Message(role="patient", content=response)
            )
            return result
        else:
            # No timed node ready — deliver generic acknowledgment
            ack_response = random.choice(scenario.acknowledgment_responses)
            result["patient_response"] = ack_response
            result["match_method"] = "acknowledgment"

            # Do NOT increment exchange_count for acknowledgments
            state.conversation_history.append(
                Message(role="student", content=student_message)
            )
            state.conversation_history.append(
                Message(role="patient", content=ack_response)
            )
            return result

    # Step 2: Run content node matching (questions and ambiguous)
    # At this point, message_type is "question" or "ambiguous".
    # Both go through node matching.

    match_result = await match_content_node(
        student_message, scenario, state,
        allow_re_ask=(message_type == "question"),  # Only allow re-ask for clear questions
        scenario_index=scenario_index,
        message_type=message_type,
    )
    result["match_method"] = match_result.match_method

    # ── OOB ──
    if match_result.is_out_of_bounds:
        result["is_out_of_bounds"] = True
        # OOB does NOT increment exchange_count
        # The student message is NOT added to history
        return result

    # ── Acknowledgment (from LLM fallback — ambiguous message classified as ack) ──
    if match_result.is_acknowledgment:
        timed_node = _get_next_ready_timed_node(scenario, state)
        if timed_node:
            # Use timed wrapper to signal patient-initiated topic change
            timed_wrapper = await generate_timed_wrapper(
                patient_profile=scenario.patient_profile,
                wrapper_history=state.wrapper_history,
            )
            state.wrapper_history.append(timed_wrapper)
            if len(state.wrapper_history) > 3:
                state.wrapper_history = state.wrapper_history[-3:]

            result["patient_response"] = f"{timed_wrapper} {timed_node.canonical_content}"
            result["match_method"] = "timed"
            result["nodes_delivered"].append(timed_node.node_id)
            _deliver_timed_node(timed_node, state)
            state.exchange_count += 1
        else:
            ack_response = random.choice(scenario.acknowledgment_responses)
            result["patient_response"] = ack_response
            result["match_method"] = "acknowledgment"
            # Do NOT increment exchange_count

        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=result["patient_response"])
        )
        return result

    # ── Re-ask ──
    if match_result.is_re_ask and match_result.re_ask_node:
        reask_node = match_result.re_ask_node
        prefix = random.choice(scenario.re_ask_prefixes)
        canonical = reask_node.canonical_content
        first_sentence = canonical.split(". ")[0]
        if not first_sentence.endswith("."):
            first_sentence += "."
        # Lowercase the first character of the recap if the prefix provides the sentence start
        recap = first_sentence[0].lower() + first_sentence[1:] if first_sentence[0].isupper() else first_sentence
        recap_response = f"{prefix} {recap}"

        result["patient_response"] = recap_response
        result["match_method"] = "re_ask"
        state.exchange_count += 1
        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=recap_response)
        )
        return result

    # ── Tangential ──
    if match_result.is_tangential:
        # Call generative LLM as fallback for missing nodes
        system_prompt, user_prompt = build_profile_aware_prompt(
            student_message=student_message,
            patient_profile=scenario.patient_profile.dict(),
        )
        llm_response = await llm_completion(
            model=Config.HEAVYWEIGHT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=Config.HEAVYWEIGHT_TEMPERATURE,
            max_tokens=Config.HEAVYWEIGHT_MAX_TOKENS,
        )

        if llm_response:
            result["patient_response"] = llm_response
            result["match_method"] = "ollama_direct"
        else:
            # Fallback to canned responses if LLM fails
            tangential_response = random.choice(TANGENTIAL_RESPONSES)
            result["patient_response"] = tangential_response
            result["match_method"] = "tangential"

        state.exchange_count += 1
        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=result["patient_response"])
        )
        _update_timed_readiness(scenario, state)
        return result

    # ── Matched node ──
    matched_node = match_result.matched_node  
    if matched_node is None:
        # Safety fallback — shouldn't happen
        fallback_response = random.choice(TANGENTIAL_RESPONSES)
        result["patient_response"] = fallback_response
        result["match_method"] = "fallback"
        state.exchange_count += 1
        state.conversation_history.append(
            Message(role="student", content=student_message)
        )
        state.conversation_history.append(
            Message(role="patient", content=fallback_response)
        )
        return result

    # Step 3: Check essential node self-activation
    essential_nodes = get_overdue_essential_nodes(scenario, state)

    # Step 4: Generate wrapper
    response_parts = []

    if needs_wrapper(matched_node.canonical_content):
        wrapper = await generate_wrapper(
            patient_profile=scenario.patient_profile,
            wrapper_history=state.wrapper_history,
        )
        response_parts.append(wrapper)
        # Track wrapper history (keep last 3)
        state.wrapper_history.append(wrapper)
        if len(state.wrapper_history) > 3:
            state.wrapper_history = state.wrapper_history[-3:]

    # Canonical content
    response_parts.append(matched_node.canonical_content)

    # Mark matched node as delivered
    result["nodes_delivered"].append(matched_node.node_id)
    state.delivered_node_ids.append(matched_node.node_id)

    # Essential node appends
    for i, enode in enumerate(essential_nodes):
        leadin = get_essential_leadin(i)
        response_parts.append(f"{leadin} {enode.canonical_content}")
        result["nodes_delivered"].append(enode.node_id)
        state.delivered_node_ids.append(enode.node_id)

    # ── Check timed node fallback threshold ──
    # If a timed node has been ready for TIMED_NODE_FALLBACK_TURNS,
    # append it to this response with a topic-change wrapper.
    # Skip if essential nodes were just delivered — don't overload the response.
    fallback_node_id = _check_fallback_threshold(state)
    if fallback_node_id and not essential_nodes:
        fallback_node = _find_node(scenario, fallback_node_id)
        if fallback_node:
            timed_wrapper = await generate_timed_wrapper(
                patient_profile=scenario.patient_profile,
                wrapper_history=state.wrapper_history,
            )
            state.wrapper_history.append(timed_wrapper)
            if len(state.wrapper_history) > 3:
                state.wrapper_history = state.wrapper_history[-3:]

            response_parts.append(
                f"{timed_wrapper} {fallback_node.canonical_content}"
            )
            result["nodes_delivered"].append(fallback_node.node_id)
            _deliver_timed_node(fallback_node, state)

    # Step 5: Assemble
    result["patient_response"] = " ".join(response_parts)
    result["match_method"] = match_result.match_method

    # Update state
    state.exchange_count += 1
    state.conversation_history.append(
        Message(role="student", content=student_message)
    )
    state.conversation_history.append(
        Message(role="patient", content=result["patient_response"])
    )

    # Update timed readiness (new nodes may have become ready)
    _update_timed_readiness(scenario, state)

    return result
