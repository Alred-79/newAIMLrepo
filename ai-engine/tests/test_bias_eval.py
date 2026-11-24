"""
Tests for the bias_eval package.
Covers keyword scanner, prompt builders, routing logic, and integration.
"""

import sys
import os
import pytest

# Add ai-engine to path so packages can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bias_eval.keywords import scan_for_stigma_anchors
from bias_eval.prompts import (
    build_lightweight_system_prompt,
    build_lightweight_user_prompt,
)
from bias_eval.routing import compute_routing_decision, should_trigger_reflection_mvp1
from shared.models import BiasLabel, EvaluationInput, ScenarioCategory, BiasDirection


def test_keyword_scanner_compliance_match():
    results = scan_for_stigma_anchors("The patient is noncompliant")
    assert len(results) >= 1
    keywords = [r["keyword"] for r in results]
    assert "noncompliant" in keywords
    assert results[0]["category"] == "compliance"


def test_keyword_scanner_no_match():
    results = scan_for_stigma_anchors("Hello, what brings you in today?")
    assert results == []


def test_keyword_scanner_multi_match():
    results = scan_for_stigma_anchors("Patient is drug-seeking and aggressive")
    keywords = [r["keyword"] for r in results]
    assert "drug-seeking" in keywords
    assert "aggressive" in keywords
    assert len(results) == 2
    for r in results:
        assert r["category"] == "descriptor"


def test_keyword_scanner_multi_word_phrases():
    results_hyphen = scan_for_stigma_anchors("The patient is drug-seeking")
    results_space = scan_for_stigma_anchors("The patient is drug seeking")
    assert len(results_hyphen) >= 1
    assert len(results_space) >= 1
    assert results_hyphen[0]["keyword"] == "drug-seeking"
    assert results_space[0]["keyword"] == "drug seeking"


def test_keyword_scanner_case_insensitive():
    results = scan_for_stigma_anchors("Patient is NONCOMPLIANT with meds")
    keywords = [r["keyword"] for r in results]
    assert "noncompliant" in keywords


def test_prompt_student_to_patient_direction():
    prompt = build_lightweight_system_prompt("student_to_patient", [])
    assert "STUDENT exhibits bias toward the PATIENT" in prompt


def test_prompt_patient_to_student_direction():
    prompt = build_lightweight_system_prompt("patient_to_student", [])
    assert "PATIENT who is hostile" in prompt


def test_prompt_none_direction():
    prompt = build_lightweight_system_prompt("none", [])
    assert "control scenario" in prompt


def test_prompt_with_stigma_anchors():
    anchors = [{"keyword": "noncompliant", "category": "compliance", "position": 15}]
    prompt = build_lightweight_system_prompt("student_to_patient", anchors)
    assert "noncompliant" in prompt
    assert "compliance" in prompt


def test_prompt_without_stigma_anchors():
    prompt = build_lightweight_system_prompt("student_to_patient", [])
    assert "No stigmatizing language" in prompt


def test_reflection_biased_high_confidence():
    assert should_trigger_reflection_mvp1(BiasLabel.BIASED, 0.9) is True


def test_reflection_biased_low_confidence():
    assert should_trigger_reflection_mvp1(BiasLabel.BIASED, 0.5) is False


def test_reflection_unbiased_high_confidence():
    assert should_trigger_reflection_mvp1(BiasLabel.UNBIASED, 0.9) is False


def test_reflection_uncertain():
    assert should_trigger_reflection_mvp1(BiasLabel.UNCERTAIN, 0.8) is False


def test_routing_low_confidence_escalation():
    # High confidence unbiased → no escalation (simplified routing)
    result = compute_routing_decision(BiasLabel.UNBIASED, 0.9)
    assert result is False
    # Low confidence → escalate
    result = compute_routing_decision(BiasLabel.UNBIASED, 0.5)
    assert result is True


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration test",
)
def test_evaluate_tone_biased_message():
    from bias_eval.lightweight import evaluate_tone_sync

    eval_input = EvaluationInput(
        student_message="Well at your age, these things happen. Just try to stay active.",
        recent_context=[],
        scenario_category=ScenarioCategory.AGE,
        bias_direction=BiasDirection.STUDENT_TO_PATIENT,
    )
    result = evaluate_tone_sync(eval_input)
    assert result.tone_label in (BiasLabel.BIASED, BiasLabel.UNCERTAIN)
    assert 0.0 <= result.tone_confidence <= 1.0
    assert result.raw_llm_response is not None


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration test",
)
def test_evaluate_tone_neutral_message():
    from bias_eval.lightweight import evaluate_tone_sync

    eval_input = EvaluationInput(
        student_message="Can you tell me more about when the pain started?",
        recent_context=[],
        scenario_category=ScenarioCategory.AGE,
        bias_direction=BiasDirection.STUDENT_TO_PATIENT,
    )
    result = evaluate_tone_sync(eval_input)
    assert result.tone_label == BiasLabel.UNBIASED
