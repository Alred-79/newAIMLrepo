"""
ClinEQ — Session Manager

Manages a single student's encounter session. Tracks the conversation
(including System messages), coordinates the simulator, and handles
special cases (opening line, OOB reset, encounter completion).

Post-encounter hooks:
- Control scenario → generates a behavioral profile
- Experimental scenario → generates a delta analysis report

This is the primary interface for the fullstack team to call.
"""

import uuid
from typing import Optional

from shared.models import (
    ScenarioConfig,
    ScenarioState,
    SimulatorInput,
    Message,
)
from shared.config import Config
from patient_sim.simulator import generate_patient_response


class SessionManager:
    """
    Manages a single encounter session.

    Usage:
        session = SessionManager(scenario)
        # First call returns the opening line
        result = await session.process_student_message("Hi there")
        # Subsequent calls return patient responses
        result = await session.process_student_message("What brings you in?")

    The result dict contains:
        - "messages": list of Message dicts to append to the chat, in order.
          Each has "role" ("student", "patient", or "system") and "content".
        - "is_complete": bool — whether the encounter has ended.
        - "metadata": dict with match_method, nodes_delivered, latency, etc.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        session_id: Optional[str] = None,
        scenario_index=None,
        control_profile: Optional[dict] = None,
        control_history: Optional[list[Message]] = None,
        control_scenario: Optional[ScenarioConfig] = None,
    ):
        self.scenario = scenario
        self.scenario_index = scenario_index
        self.session_id = session_id or str(uuid.uuid4())
        self.state = ScenarioState()
        self.message_index = 0

        # Cross-encounter state (passed in from outside)
        self.control_profile = control_profile
        self.control_history = control_history
        self.control_scenario = control_scenario

    async def process_student_message(self, student_message: str) -> dict:
        """
        Process a single student message and return all resulting chat messages.

        Returns:
            dict with "messages" (list of Message-like dicts), "is_complete",
            and "metadata".
        """
        messages_to_send = []

        # ── Hard stop check ──
        if self.state.exchange_count >= Config.MAX_CONVERSATION_LENGTH:
            self.state.is_complete = True
            return {
                "messages": [{
                    "role": "system",
                    "content": "The encounter has ended (maximum length reached).",
                }],
                "is_complete": True,
                "metadata": {
                    "exchange_count": self.state.exchange_count,
                    "total_nodes_delivered": len(self.state.delivered_node_ids),
                    "total_nodes_available": len(self.scenario.content_nodes),
                },
            }

        # Build simulator input
        sim_input = SimulatorInput(
            student_message=student_message,
            scenario=self.scenario,
            state=self.state,
        )

        # Run the simulator
        output = await generate_patient_response(
            sim_input,
            scenario_index=self.scenario_index,
            control_profile=self.control_profile,
        )

        # Always add the student message first
        messages_to_send.append({
            "role": "student",
            "content": student_message,
        })

        # Handle OOB — student message is shown but flagged
        if output.is_out_of_bounds:
            for sys_msg in output.system_messages:
                messages_to_send.append({
                    "role": "system",
                    "content": sys_msg,
                })
            return {
                "messages": messages_to_send,
                "is_complete": False,
                "metadata": {
                    "match_method": "out_of_bounds",
                    "exchange_count": self.state.exchange_count,
                },
            }

        # Patient response
        if output.patient_response:
            messages_to_send.append({
                "role": "patient",
                "content": output.patient_response,
            })

        # System messages (reflection triggers from eval)
        # These come AFTER the patient response
        for sys_msg in output.system_messages:
            messages_to_send.append({
                "role": "system",
                "content": sys_msg,
            })

        # Check if encounter is complete
        is_complete = self.state.is_complete or (
            self.state.exchange_count >= Config.MAX_CONVERSATION_LENGTH
        )

        self.message_index += 1

        return {
            "messages": messages_to_send,
            "is_complete": is_complete,
            "metadata": {
                "match_method": output.match_method,
                "nodes_delivered": output.nodes_delivered,
                "exchange_count": self.state.exchange_count,
                "latency_ms": output.latency_ms,
            },
        }

    async def end_encounter(self) -> dict:
        """
        Student-initiated encounter end.

        If this is the control scenario, generates a control profile.
        If this is an experimental scenario and control data is available,
        generates a delta analysis report.
        """
        self.state.is_complete = True

        undelivered_essential = [
            node.node_id
            for node in self.scenario.content_nodes
            if node.is_essential and node.node_id not in self.state.delivered_node_ids
        ]

        result = {
            "messages": [{
                "role": "system",
                "content": "The encounter has ended.",
            }],
            "is_complete": True,
            "metadata": {
                "exchange_count": self.state.exchange_count,
                "total_nodes_delivered": len(self.state.delivered_node_ids),
                "total_nodes_available": len(self.scenario.content_nodes),
                "undelivered_essential_nodes": undelivered_essential,
            },
        }

        # ── Post-encounter analysis ──
        from bias_eval.heavyweight import generate_control_profile, generate_delta_report

        if self.scenario.bias_direction.value == "none":
            # Control scenario — generate profile
            try:
                profile = await generate_control_profile(
                    self.state.conversation_history
                )
                result["control_profile"] = profile
                result["metadata"]["control_profile_generated"] = True
            except Exception as e:
                result["metadata"]["control_profile_error"] = str(e)

        elif self.control_history and self.control_scenario:
            # Experimental scenario — generate delta analysis
            try:
                delta = await generate_delta_report(
                    control_history=self.control_history,
                    experimental_history=self.state.conversation_history,
                    control_scenario=self.control_scenario,
                    experimental_scenario=self.scenario,
                    control_profile=self.control_profile,
                )
                result["delta_analysis"] = delta.model_dump()
                result["metadata"]["delta_analysis_generated"] = True
            except Exception as e:
                result["metadata"]["delta_analysis_error"] = str(e)

        return result
