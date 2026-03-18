from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationInput(BaseModel):
    """
    EXACT evaluator input contract.

    Fields (must not change):
    - message: str
    - scenario: str  # "control" | "ageism" | "zionist"
    - metadata: Optional[dict]
    """

    model_config = ConfigDict(extra="allow")  # allow forward-compat keys if callers send them

    message: str
    scenario: str  # "control" | "ageism" | "zionist"
    metadata: Optional[Dict[str, Any]] = None


class EvaluationOutput(BaseModel):
    """
    EXACT evaluator output contract.

    Fields (must not change):
    - original_message: str
    - scenario: str
    - stigma_flags: List[str]
    - llm_response: str
    - parsed_label: str  # "safe" | "concerning" | "harmful"
    - confidence: float
    - routing_decision: str  # "pass" | "review" | "escalate"
    - notes: Optional[str]
    """

    model_config = ConfigDict(extra="forbid")

    original_message: str
    scenario: str
    stigma_flags: List[str] = Field(default_factory=list)
    llm_response: str = ""
    parsed_label: str = "safe"
    confidence: float = 0.0
    routing_decision: str = "pass"
    notes: Optional[str] = None


__all__ = ["EvaluationInput", "EvaluationOutput"]


