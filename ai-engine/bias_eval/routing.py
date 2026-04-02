"""
ClinEQ — Bias Evaluation Routing Logic

Determines whether a lightweight evaluation result should be escalated
to the heavyweight evaluator.

Simplified: only escalates on low confidence or uncertainty.
Decision point detection is handled separately by the clinical keyword scanner.
"""

from shared.models import BiasLabel
from shared.config import Config


def compute_routing_decision(
    bias_label: BiasLabel,
    confidence: float,
) -> bool:
    """
    Determine if the lightweight result is uncertain enough to
    warrant heavyweight escalation.

    Simplified: only escalates on low confidence or uncertainty.
    Decision point detection is handled separately by the
    clinical keyword scanner.

    Returns:
        True if the lightweight result is uncertain.
    """
    threshold = Config.CONFIDENCE_THRESHOLD

    if bias_label == BiasLabel.BIASED:
        return confidence < threshold  # Low-confidence bias → escalate

    elif bias_label == BiasLabel.UNBIASED:
        return confidence < threshold  # Low-confidence unbiased → escalate

    elif bias_label == BiasLabel.UNCERTAIN:
        return True  # Always escalate uncertainty

    return True  # Fallback


def should_trigger_reflection_mvp1(
    bias_label: BiasLabel,
    confidence: float,
) -> bool:
    """
    Should the lightweight result trigger a System message?
    Only fires on: biased + confidence >= threshold.
    """
    return (
        bias_label == BiasLabel.BIASED
        and confidence >= Config.CONFIDENCE_THRESHOLD
    )
