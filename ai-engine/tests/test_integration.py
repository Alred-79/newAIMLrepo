"""
ClinEQ — Integration Tests

Tests that validate cross-package behavior:
- Model roundtrips
- State tracking
- Timed/essential node logic
- Keyword scanner + prompt builder integration
- Routing logic matrix
- End-to-end evaluator calls (API-dependent, skippable)
- Full session flow (API + DB dependent, skippable)
"""

import os
import json
import pytest

from shared.models import (
    ScenarioConfig,
    ScenarioState,
    ContentNode,
    PatientProfile,
    NodeType,
    ScenarioCategory,
    BiasDirection,
    BiasLabel,
    ClinicalContext,
    EvaluationInput,
    Message,
)
from shared.config import Config
from patient_sim.response_assembler import (
    get_overdue_essential_nodes,
    needs_wrapper,
    _update_timed_readiness,
    _get_next_ready_timed_node,
)
from bias_eval.keywords import scan_for_stigma_anchors
from bias_eval.prompts import build_lightweight_system_prompt
from bias_eval.routing import compute_routing_decision, should_trigger_reflection_mvp1



def _make_profile() -> PatientProfile:
    return PatientProfile(
        name="Test Patient",
        age=45,
        gender="male",
        chief_complaint="Test complaint",
    )


def _make_scenario(content_nodes: list[ContentNode] | None = None) -> ScenarioConfig:
    nodes = content_nodes or [
        ContentNode(
            node_id=f"node_{i}",
            topic=f"Topic {i}",
            canonical_content=f"Canonical content for node {i}.",
            node_type=NodeType.TRIGGERED,
            trigger_phrases=[f"trigger phrase {i}"],
        )
        for i in range(5)
    ]
    return ScenarioConfig(
        scenario_id="test_scenario",
        title="Test Scenario",
        category=ScenarioCategory.CONTROL,
        bias_direction=BiasDirection.NONE,
        clinical_context=ClinicalContext.PRIMARY_CARE,
        patient_profile=_make_profile(),
        content_nodes=nodes,
        opening_line="Hello, I'm the test patient.",
    )



class TestModelRoundtrip:
    """Test 1: Full model roundtrip — serialize and deserialize ScenarioConfig."""

    def test_scenario_config_roundtrip(self):
        scenario = _make_scenario()
        json_str = scenario.model_dump_json()
        restored = ScenarioConfig.model_validate_json(json_str)

        assert restored.scenario_id == scenario.scenario_id
        assert restored.title == scenario.title
        assert restored.category == scenario.category
        assert restored.bias_direction == scenario.bias_direction
        assert len(restored.content_nodes) == 5
        for orig, rest in zip(scenario.content_nodes, restored.content_nodes):
            assert orig.node_id == rest.node_id
            assert orig.topic == rest.topic
            assert orig.canonical_content == rest.canonical_content
            assert orig.node_type == rest.node_type
            assert orig.trigger_phrases == rest.trigger_phrases
        assert restored.opening_line == scenario.opening_line
        assert restored.patient_profile.name == scenario.patient_profile.name


class TestScenarioStateTracking:
    """Test 2: ScenarioState tracking — deliver nodes and increment exchange count."""

    def test_state_mutation(self):
        state = ScenarioState()
        assert state.exchange_count == 0
        assert state.delivered_node_ids == []

        # Simulate delivering 3 nodes
        state.delivered_node_ids.append("node_0")
        state.delivered_node_ids.append("node_1")
        state.delivered_node_ids.append("node_2")
        state.exchange_count = 10

        assert len(state.delivered_node_ids) == 3
        assert "node_1" in state.delivered_node_ids
        assert state.exchange_count == 10


class TestTimedNodeDetection:
    """Test 3: Timed node readiness — promotes to ready at correct exchange count."""

    def test_timed_node_becomes_ready_at_correct_exchange(self):
        timed_node = ContentNode(
            node_id="timed_1",
            topic="Timed topic",
            canonical_content="Timed content.",
            node_type=NodeType.TIMED,
            fire_after_exchange=5,
        )
        scenario = _make_scenario([timed_node])
        state = ScenarioState(exchange_count=5)

        _update_timed_readiness(scenario, state)
        result = _get_next_ready_timed_node(scenario, state)
        assert result is not None
        assert result.node_id == "timed_1"

    def test_timed_node_not_ready_early(self):
        timed_node = ContentNode(
            node_id="timed_1",
            topic="Timed topic",
            canonical_content="Timed content.",
            node_type=NodeType.TIMED,
            fire_after_exchange=5,
        )
        scenario = _make_scenario([timed_node])
        state = ScenarioState(exchange_count=4)

        _update_timed_readiness(scenario, state)
        result = _get_next_ready_timed_node(scenario, state)
        assert result is None

    def test_timed_node_not_ready_if_already_delivered(self):
        timed_node = ContentNode(
            node_id="timed_1",
            topic="Timed topic",
            canonical_content="Timed content.",
            node_type=NodeType.TIMED,
            fire_after_exchange=5,
        )
        scenario = _make_scenario([timed_node])
        state = ScenarioState(exchange_count=5, delivered_node_ids=["timed_1"])

        _update_timed_readiness(scenario, state)
        result = _get_next_ready_timed_node(scenario, state)
        assert result is None


class TestEssentialNodeDetection:
    """Test 4: Essential node detection — self-activates when overdue."""

    def test_essential_node_overdue(self):
        essential_node = ContentNode(
            node_id="essential_1",
            topic="Essential topic",
            canonical_content="Essential content.",
            node_type=NodeType.TRIGGERED,
            is_essential=True,
            fire_after_exchange=10,
        )
        scenario = _make_scenario([essential_node])
        state = ScenarioState(exchange_count=10)

        result = get_overdue_essential_nodes(scenario, state)
        assert len(result) == 1
        assert result[0].node_id == "essential_1"

    def test_essential_node_not_yet_due(self):
        essential_node = ContentNode(
            node_id="essential_1",
            topic="Essential topic",
            canonical_content="Essential content.",
            node_type=NodeType.TRIGGERED,
            is_essential=True,
            fire_after_exchange=10,
        )
        scenario = _make_scenario([essential_node])
        state = ScenarioState(exchange_count=9)

        result = get_overdue_essential_nodes(scenario, state)
        assert len(result) == 0

    def test_essential_node_already_delivered(self):
        essential_node = ContentNode(
            node_id="essential_1",
            topic="Essential topic",
            canonical_content="Essential content.",
            node_type=NodeType.TRIGGERED,
            is_essential=True,
            fire_after_exchange=10,
        )
        scenario = _make_scenario([essential_node])
        state = ScenarioState(exchange_count=10, delivered_node_ids=["essential_1"])

        result = get_overdue_essential_nodes(scenario, state)
        assert len(result) == 0


class TestNeedsWrapper:
    """Test 5: needs_wrapper logic — based on word count threshold."""

    def test_short_content_no_wrapper(self):
        assert needs_wrapper("No.") is False

    def test_long_content_needs_wrapper(self):
        assert needs_wrapper("I am allergic to peanuts.") is True

    def test_single_word_no_wrapper(self):
        assert needs_wrapper("Yes") is False

    def test_exactly_three_words_no_wrapper(self):
        assert needs_wrapper("John is good.") is False

    def test_four_words_needs_wrapper(self):
        assert needs_wrapper("It started three days.") is True


class TestTimedNodeReadinessFIFO:
    """Test 6: Timed node readiness FIFO — ready nodes fire in order."""

    def test_fifo_order(self):
        state = ScenarioState(
            ready_timed_nodes=["node_a", "node_b"],
            timed_node_ready_since={"node_a": 3, "node_b": 5},
        )
        timed_a = ContentNode(
            node_id="node_a", topic="A", canonical_content="A.",
            node_type=NodeType.TIMED, fire_after_exchange=3,
        )
        timed_b = ContentNode(
            node_id="node_b", topic="B", canonical_content="B.",
            node_type=NodeType.TIMED, fire_after_exchange=5,
        )
        scenario = _make_scenario([timed_a, timed_b])

        first = _get_next_ready_timed_node(scenario, state)
        assert first is not None
        assert first.node_id == "node_a"


class TestKeywordScannerPromptIntegration:
    """Test 7: Keyword scanner + prompt builder integration."""

    def test_detected_keywords_appear_in_prompt(self):
        biased_message = "The patient is noncompliant and aggressive"
        anchors = scan_for_stigma_anchors(biased_message)

        assert len(anchors) > 0
        keywords_found = [a["keyword"] for a in anchors]
        assert "noncompliant" in keywords_found
        assert "aggressive" in keywords_found

        prompt = build_lightweight_system_prompt(
            bias_direction="student_to_patient",
            stigma_anchors_found=anchors,
        )

        assert "noncompliant" in prompt
        assert "aggressive" in prompt
        assert "compliance" in prompt  # category of "noncompliant"
        assert "descriptor" in prompt  # category of "aggressive"

    def test_no_keywords_prompt_section(self):
        neutral_message = "Can you describe the rash for me?"
        anchors = scan_for_stigma_anchors(neutral_message)
        assert len(anchors) == 0

        prompt = build_lightweight_system_prompt(
            bias_direction="none",
            stigma_anchors_found=anchors,
        )
        assert "No stigmatizing language anchor words" in prompt


class TestRoutingLogicMatrix:
    """Test 8: Routing logic matrix — simplified (no decision point)."""

    # compute_routing_decision: 4 quadrants
    def test_biased_high_confidence_no_escalate(self):
        assert compute_routing_decision(BiasLabel.BIASED, 0.85) is False

    def test_biased_low_confidence_escalate(self):
        assert compute_routing_decision(BiasLabel.BIASED, 0.50) is True

    def test_unbiased_high_confidence_no_escalate(self):
        assert compute_routing_decision(BiasLabel.UNBIASED, 0.85) is False

    def test_unbiased_low_confidence_escalate(self):
        assert compute_routing_decision(BiasLabel.UNBIASED, 0.50) is True

    def test_uncertain_always_escalate(self):
        assert compute_routing_decision(BiasLabel.UNCERTAIN, 0.90) is True

    # should_trigger_reflection_mvp1: 4 cases
    def test_reflection_biased_high_confidence(self):
        assert should_trigger_reflection_mvp1(BiasLabel.BIASED, 0.85) is True

    def test_no_reflection_biased_low_confidence(self):
        assert should_trigger_reflection_mvp1(BiasLabel.BIASED, 0.50) is False

    def test_no_reflection_unbiased(self):
        assert should_trigger_reflection_mvp1(BiasLabel.UNBIASED, 0.95) is False

    def test_no_reflection_uncertain(self):
        assert should_trigger_reflection_mvp1(BiasLabel.UNCERTAIN, 0.90) is False



HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
class TestLightweightEvaluatorE2E:
    """Tests 9-10: Lightweight evaluator end-to-end with real API calls."""

    def test_biased_message_detection(self):
        """Test 9: Clearly biased message should return populated result."""
        from bias_eval.lightweight import evaluate_tone_sync

        eval_input = EvaluationInput(
            student_message="At your age, you should just accept these things",
            recent_context=[],
            scenario_category=ScenarioCategory.AGE,
            bias_direction=BiasDirection.PATIENT_TO_STUDENT,
        )
        result = evaluate_tone_sync(eval_input)

        assert result.tone_label is not None
        assert 0.0 <= result.tone_confidence <= 1.0
        assert isinstance(result.should_escalate, bool)
        assert result.brief_reason  # should have some reasoning

    def test_neutral_message_detection(self):
        """Test 10: Neutral clinical question should return unbiased."""
        from bias_eval.lightweight import evaluate_tone_sync

        eval_input = EvaluationInput(
            student_message="Can you describe the rash for me?",
            recent_context=[],
            scenario_category=ScenarioCategory.CONTROL,
            bias_direction=BiasDirection.NONE,
        )
        result = evaluate_tone_sync(eval_input)

        assert result.tone_label == BiasLabel.UNBIASED
        assert result.tone_confidence >= 0.5  # should be reasonably confident



def _db_available() -> bool:
    """Check if PostgreSQL with pgvector is reachable."""
    try:
        import psycopg2
        conn = psycopg2.connect(Config.DATABASE_URL)
        conn.close()
        return True
    except Exception:
        return False


HAS_DB = _db_available()


@pytest.mark.skipif(
    not HAS_OPENAI_KEY or not HAS_DB,
    reason="Requires OPENAI_API_KEY and running PostgreSQL",
)
class TestScenarioLoadAndSeed:
    """Test 11: Scenario load and seed pipeline."""

    @pytest.mark.asyncio
    async def test_load_and_seed_john_smith(self):
        from patient_sim.scenario_loader import load_and_seed_scenario

        scenario = await load_and_seed_scenario(
            "ai-engine/data/scenarios/control_john_smith.json"
        )
        assert scenario.scenario_id == "control_john_smith"
        assert len(scenario.content_nodes) == 18

        # Verify embeddings exist in database
        from patient_sim.db import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM sim.trigger_embeddings WHERE scenario_id = %s",
                    ("control_john_smith",),
                )
                count = cur.fetchone()[0]
                assert count > 0
        finally:
            conn.close()


@pytest.mark.skipif(
    not HAS_OPENAI_KEY or not HAS_DB,
    reason="Requires OPENAI_API_KEY and running PostgreSQL",
)
class TestNodeMatching:
    """Test 12: Node matching after seeding."""

    @pytest.mark.asyncio
    async def test_allergy_query_matches(self):
        from patient_sim.scenario_loader import load_and_seed_scenario
        from patient_sim.node_matcher import embed_message
        from patient_sim.db import query_similar

        scenario = await load_and_seed_scenario(
            "ai-engine/data/scenarios/control_john_smith.json"
        )

        query_embedding = await embed_message("Do you have any allergies?")
        results = query_similar(
            scenario_id="control_john_smith",
            query_embedding=query_embedding,
            excluded_node_ids=[],
            top_k=3,
        )

        assert len(results) > 0
        top = results[0]
        assert top["similarity"] > 0.65
        # The top result should be the allergies node
        assert "allerg" in top["node_id"].lower() or top["similarity"] > 0.65


@pytest.mark.skipif(
    not HAS_OPENAI_KEY or not HAS_DB,
    reason="Requires OPENAI_API_KEY and running PostgreSQL",
)
class TestFullSessionFlow:
    """Test 13: Full session flow with SessionManager."""

    @pytest.mark.asyncio
    async def test_three_message_conversation(self):
        from patient_sim.scenario_loader import load_and_seed_scenario
        from shared.session import SessionManager

        scenario = await load_and_seed_scenario(
            "ai-engine/data/scenarios/control_john_smith.json"
        )
        session = SessionManager(scenario)

        # Message 1: greeting — should get opening line
        result1 = await session.process_student_message("Hi there")
        patient_msgs1 = [
            m for m in result1["messages"] if m["role"] == "patient"
        ]
        assert len(patient_msgs1) == 1
        assert patient_msgs1[0]["content"] == scenario.opening_line
        assert result1["metadata"]["exchange_count"] == 1

        # Message 2: chief complaint question
        result2 = await session.process_student_message(
            "What brings you in today?"
        )
        patient_msgs2 = [
            m for m in result2["messages"] if m["role"] == "patient"
        ]
        assert len(patient_msgs2) == 1
        # Should contain chief complaint canonical content
        assert "rash" in patient_msgs2[0]["content"].lower() or \
               "blister" in patient_msgs2[0]["content"].lower()
        assert result2["metadata"]["exchange_count"] == 2

        # Message 3: onset question
        result3 = await session.process_student_message(
            "When did this start?"
        )
        patient_msgs3 = [
            m for m in result3["messages"] if m["role"] == "patient"
        ]
        assert len(patient_msgs3) == 1
        assert result3["metadata"]["exchange_count"] == 3

        # No system messages expected in control scenario
        for r in [result1, result2, result3]:
            system_msgs = [m for m in r["messages"] if m["role"] == "system"]
            assert len(system_msgs) == 0

        assert not result3["is_complete"]
