"""
Smoke tests: verify all shared models import, instantiate, and serialize correctly.
"""

import sys
import os

# Add ai-engine to path so shared can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.models import (
    Message,
    ScenarioCategory,
    BiasDirection,
    BiasLabel,
    NodeType,
    ClinicalContext,
    EvaluationInput,
    EvaluationResult,
    ContentNode,
    PatientProfile,
    ReferralOption,
    ReferralDecision,
    DecisionPoints,
    ScenarioConfig,
    ScenarioState,
    MatchResult,
    SimulatorInput,
    SimulatorOutput,
)
from shared.config import Config


def test_message_instantiation_and_serialization():
    msg = Message(role="student", content="Hello, how are you feeling today?")
    data = msg.model_dump()
    assert data["role"] == "student"
    assert data["content"] == "Hello, how are you feeling today?"
    json_str = msg.model_dump_json()
    assert "student" in json_str


def test_content_node_all_fields():
    node = ContentNode(
        node_id="node_1",
        topic="allergies",
        canonical_content="I am allergic to peanuts. Anaphylactic reaction.",
        node_type=NodeType.TRIGGERED,
        trigger_phrases=["any allergies", "allergic to anything"],
        fire_after_exchange=None,
        is_delivered=False,
        is_essential=False,
        prerequisite_nodes=[],
        bias_reactions=["*Patient looks uncomfortable*"],
    )
    assert node.node_id == "node_1"
    assert node.node_type == NodeType.TRIGGERED
    assert len(node.trigger_phrases) == 2
    assert len(node.bias_reactions) == 1


def test_evaluation_result_valid_values():
    result = EvaluationResult(
        tone_label=BiasLabel.BIASED,
        tone_confidence=0.85,
        stigma_anchors_found=[{"term": "at your age", "category": "ageism"}],
        brief_reason="Ageist assumption about cognitive ability",
        should_escalate=True,
        raw_llm_response='{"label": "biased"}',
    )
    assert result.tone_label == BiasLabel.BIASED
    assert result.tone_confidence == 0.85
    assert result.should_escalate is True


def test_config_loads_without_errors():
    config = Config()
    assert config.LIGHTWEIGHT_MODEL == "ollama_chat/qwen2.5:14b"
    assert config.HEAVYWEIGHT_MODEL == "ollama_chat/qwen2.5:14b"
    assert config.CONFIDENCE_THRESHOLD == 0.80
    assert config.EMBEDDING_DIMENSIONS == 1024
    assert config.NODE_MATCH_CONFIDENT == 0.60
    assert config.NODE_MATCH_AMBIGUOUS == 0.35


def test_scenario_category_enum_values():
    assert ScenarioCategory.WEIGHT == "weight"
    assert ScenarioCategory.SES == "ses"
    assert ScenarioCategory.AGE == "age"
    assert ScenarioCategory.RACE == "race"
    assert ScenarioCategory.GENDER == "gender"
    assert ScenarioCategory.RELIGION_IDENTITY == "religion_identity"
    assert ScenarioCategory.CONTROL == "control"


def test_bias_label_enum_values():
    assert BiasLabel.BIASED == "biased"
    assert BiasLabel.UNBIASED == "unbiased"
    assert BiasLabel.UNCERTAIN == "uncertain"


def test_node_type_enum_values():
    assert NodeType.TRIGGERED == "triggered"
    assert NodeType.TIMED == "timed"
    assert len(NodeType) == 2  # No ESSENTIAL — essential is a flag, not a type


def test_full_scenario_config():
    profile = PatientProfile(
        name="Adele",
        age=93,
        gender="female",
        chief_complaint="Follow-up visit for hypertension",
    )
    node = ContentNode(
        node_id="n1",
        topic="medications",
        canonical_content="I take lisinopril 10mg daily.",
    )
    config = ScenarioConfig(
        scenario_id="age_001",
        title="Elderly patient follow-up",
        category=ScenarioCategory.AGE,
        bias_direction=BiasDirection.PATIENT_TO_STUDENT,
        clinical_context=ClinicalContext.PRIMARY_CARE,
        patient_profile=profile,
        content_nodes=[node],
        opening_line="Hello doctor, I'm here for my blood pressure check.",
    )
    assert config.scenario_id == "age_001"
    assert config.patient_profile.name == "Adele"
    assert len(config.content_nodes) == 1
    assert config.max_exchanges_before_essential == 20
    assert config.decision_points is None  # Optional, defaults to None


def test_scenario_config_with_decision_points():
    profile = PatientProfile(
        name="Tehila",
        age=32,
        gender="female",
        chief_complaint="Chest pain and anxiety",
    )
    node = ContentNode(
        node_id="n1",
        topic="symptoms",
        canonical_content="I've been having chest pains.",
    )
    dp = DecisionPoints(
        referral=ReferralDecision(
            type="therapist_referral",
            options=[
                ReferralOption(name="refer", description="Refer to therapist"),
                ReferralOption(name="decline", description="Do not refer"),
            ],
            bias_note="Consider whether religion influenced this decision.",
        ),
        differential_diagnosis=["anxiety", "GERD", "cardiac"],
    )
    config = ScenarioConfig(
        scenario_id="rel_001",
        title="Zionist patient encounter",
        category=ScenarioCategory.RELIGION_IDENTITY,
        bias_direction=BiasDirection.STUDENT_TO_PATIENT,
        clinical_context=ClinicalContext.PRIMARY_CARE,
        patient_profile=profile,
        content_nodes=[node],
        opening_line="Hello doctor.",
        decision_points=dp,
    )
    assert config.decision_points is not None
    assert config.decision_points.referral.type == "therapist_referral"
    assert len(config.decision_points.referral.options) == 2
    assert len(config.decision_points.differential_diagnosis) == 3


def test_scenario_state_is_complete_flag():
    state = ScenarioState()
    assert state.is_complete is False
    state.is_complete = True
    assert state.is_complete is True


def test_config_mvp1_flags():
    config = Config()
    assert config.HEAVYWEIGHT_ENABLED is True
    assert config.COMPOUND_MATCHING_ENABLED is False
    assert config.PREREQUISITE_NODES_ENABLED is False
    assert config.FALLBACK_CLASSIFIER_MODEL == "ollama_chat/qwen2.5:14b"
    assert config.WRAPPER_TEMPERATURE == 0.5
    assert config.FALLBACK_CLASSIFIER_TEMPERATURE == 0.0
    assert config.FALLBACK_CLASSIFIER_MAX_TOKENS == 100
    assert config.MAX_CONTEXT_EXCHANGES == 4
    assert config.HEAVYWEIGHT_CONFIDENCE_THRESHOLD == 0.75
