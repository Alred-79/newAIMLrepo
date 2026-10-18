"""
ClinEQ Shared Data Models
All Pydantic models used across the AI engine.
Both bias_eval and patient_sim packages import from here.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional



class ScenarioCategory(str, Enum):
    WEIGHT = "weight"
    SES = "ses"
    AGE = "age"
    RACE = "race"
    GENDER = "gender"
    RELIGION_IDENTITY = "religion_identity"
    CONTROL = "control"


class BiasDirection(str, Enum):
    STUDENT_TO_PATIENT = "student_to_patient"
    PATIENT_TO_STUDENT = "patient_to_student"
    BIDIRECTIONAL = "bidirectional"
    NONE = "none"


class BiasLabel(str, Enum):
    BIASED = "biased"
    UNBIASED = "unbiased"
    UNCERTAIN = "uncertain"


class NodeType(str, Enum):
    TRIGGERED = "triggered"
    TIMED = "timed"


class ClinicalContext(str, Enum):
    EMERGENCY = "emergency"
    URGENT_CARE = "urgent_care"
    PRIMARY_CARE = "primary_care"
    SURGICAL = "surgical"
    INPATIENT = "inpatient"



class Message(BaseModel):
    """Single conversation turn."""
    role: str  # "student", "patient", or "system"
    content: str
    timestamp: Optional[str] = None



class EvaluationInput(BaseModel):
    """Input to the lightweight bias evaluator."""
    student_message: str
    recent_context: list[Message] = Field(
        default_factory=list,
        description="Last 2-3 exchanges for minimal context"
    )
    scenario_category: ScenarioCategory
    bias_direction: BiasDirection = BiasDirection.STUDENT_TO_PATIENT


class EvaluationResult(BaseModel):
    """Output from the lightweight bias evaluator."""
    tone_label: BiasLabel
    tone_confidence: float = Field(ge=0.0, le=1.0)
    stigma_anchors_found: list[dict] = Field(default_factory=list)
    brief_reason: str
    should_escalate: bool
    raw_llm_response: Optional[str] = None
    # is_decision_point — REMOVED. Now handled by heavyweight.
    heavyweight_triggered: bool = False  # True if this message also triggered heavyweight


class HeavyweightResult(BaseModel):
    """Output from the heavyweight bias evaluator (real-time mode)."""
    bias_label: BiasLabel
    confidence: float = Field(ge=0.0, le=1.0)
    is_clinical_decision: bool = False
    decision_description: str = ""
    bias_pattern: str = ""  # e.g., "diagnostic_anchoring", "withholding_care"
    reasoning: str = ""
    raw_llm_response: Optional[str] = None


class DeltaAnalysis(BaseModel):
    """Output from the heavyweight bias evaluator (post-encounter mode)."""
    question_count_control: int = 0
    question_count_experimental: int = 0
    topics_covered_control: list[str] = Field(default_factory=list)
    topics_covered_experimental: list[str] = Field(default_factory=list)
    topics_skipped: list[str] = Field(default_factory=list)
    thoroughness_delta: str = ""
    ordering_differences: list[str] = Field(default_factory=list)
    communication_tone_shift: str = ""
    clinical_decision_differences: list[str] = Field(default_factory=list)
    reflection_prompts: list[str] = Field(default_factory=list)
    overall_summary: str = ""
    raw_llm_response: Optional[str] = None



class ContentNode(BaseModel):
    """A single unit of canonical patient content."""
    node_id: str
    topic: str
    canonical_content: str
    node_type: NodeType = NodeType.TRIGGERED
    trigger_phrases: list[str] = Field(default_factory=list)
    fire_after_exchange: Optional[int] = None
    is_delivered: bool = False
    is_essential: bool = False
    prerequisite_nodes: list[str] = Field(default_factory=list)
    bias_reactions: list[str] = Field(
        default_factory=list,
        description="DEPRECATED in MVP1. Pre-authored reactions are unused "
        "since bias reactions were removed for canonical patient invariance. "
        "Field kept for schema compatibility with existing scenario JSONs."
    )


class PatientProfile(BaseModel):
    """Clinical and demographic information."""
    name: str
    age: int
    gender: str
    chief_complaint: str
    weight: Optional[str] = None
    medical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    vitals: Optional[dict] = None
    labs: Optional[dict] = None
    additional_context: Optional[str] = None


class ReferralOption(BaseModel):
    """A single referral choice in a decision point."""
    name: str
    description: str


class ReferralDecision(BaseModel):
    """Referral decision point with options."""
    type: str  # e.g., "therapist_referral", "cardiologist_referral"
    options: list[ReferralOption]
    bias_note: Optional[str] = None


class DecisionPoints(BaseModel):
    """Post-conversation structured choices. Not present in control scenarios."""
    referral: Optional[ReferralDecision] = None
    differential_diagnosis: list[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    """Complete scenario definition."""
    scenario_id: str
    title: str
    category: ScenarioCategory
    bias_direction: BiasDirection
    clinical_context: ClinicalContext
    patient_profile: PatientProfile
    content_nodes: list[ContentNode]
    opening_line: str
    tangential_deflection: str = (
        "I'm not sure about that. Was there anything "
        "else you wanted to ask about my symptoms?"
    )
    acknowledgment_responses: list[str] = Field(
        default_factory=lambda: [
            "Thank you, I appreciate that.",
            "That's kind of you to say.",
            "Thank you.",
            "I appreciate you saying that.",
        ]
    )
    re_ask_prefixes: list[str] = Field(
        default_factory=lambda: [
            "Like I mentioned,",
            "As I said earlier,",
            "Right, so as I was saying,",
        ]
    )
    closing_responses: list[str] = Field(
        default_factory=lambda: [
            "Thank you, doctor. I appreciate your time.",
            "Okay, thank you. I'll be here.",
            "Sounds good. Thank you for everything.",
        ]
    )
    max_exchanges_before_essential: int = 20
    decision_points: Optional[DecisionPoints] = None


class ScenarioState(BaseModel):
    """Mutable runtime state for an active scenario session."""
    exchange_count: int = 0
    delivered_node_ids: list[str] = Field(default_factory=list)
    conversation_history: list[Message] = Field(default_factory=list)
    is_complete: bool = False

    # Timed node readiness tracking (replaces queued_node_ids)
    ready_timed_nodes: list[str] = Field(
        default_factory=list,
        description="Node IDs of timed nodes that have passed their "
        "fire_after_exchange but have not yet fired."
    )
    timed_node_ready_since: dict[str, int] = Field(
        default_factory=dict,
        description="Maps node_id to the exchange_count when it became ready. "
        "Used for fallback threshold."
    )

    # Wrapper history for no-repeat constraint
    wrapper_history: list[str] = Field(
        default_factory=list,
        description="Last 3 wrappers used. Passed to the wrapper prompt "
        "to prevent repetition."
    )


class MatchResult(BaseModel):
    """Output from the content node matcher."""
    matched_node: Optional[ContentNode] = None
    match_confidence: float = 0.0
    match_method: str = "none"  # "embedding", "llm_fallback", "none"
    is_out_of_bounds: bool = False
    is_tangential: bool = False
    compound_nodes: list[ContentNode] = Field(default_factory=list)
    is_acknowledgment: bool = False
    is_re_ask: bool = False
    re_ask_node: Optional[ContentNode] = None


class SimulatorInput(BaseModel):
    """Input to the patient simulator."""
    student_message: str
    scenario: ScenarioConfig
    state: ScenarioState


class SimulatorOutput(BaseModel):
    """Output from the patient simulator."""
    patient_response: str
    is_out_of_bounds: bool = False
    nodes_delivered: list[str] = Field(default_factory=list)
    match_method: str = "none"
    exchange_count: int = 0
    model_used: str = ""
    latency_ms: Optional[int] = None
    system_messages: list[str] = Field(default_factory=list)
    # generated_reaction field REMOVED — no more bias reactions
